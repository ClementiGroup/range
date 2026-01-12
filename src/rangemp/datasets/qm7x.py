import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, extract_tar

from mlcg.utils import download_url
from mlcg.data import AtomicData


class QM7XDataset(InMemoryDataset):
    """Extractor for the QM7X dataset described in https://doi.org/10.1038/s41597-021-00812-2.
    Units used in the dataset are:
        - pos: [A]
        - forces: [eV/A]
        - energy: [eV]
    """

    # Set names of reference dataset extracted
    set_ids = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000"]
    # Single atom energy computed using EPBE0 method for different atoms
    EPBE0_atom = {
        6: -1027.592489146,
        17: -12516.444619523,
        1: -13.641404161,
        7: -1484.274819088,
        8: -2039.734879322,
        16: -10828.707468187,
    }

    def __init__(
        self,
        root,
        load_properties: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        if load_properties:
            self.properties = np.load(self.processed_paths[1], weights_only=False)

    def set_temperature(self, temperature: float):
        """Set temperature for the dataset. Expected temperature in K"""
        self.temperature = temperature
        self.beta = 1 / (self.temperature * self.beta)

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
        host_link = "https://zenodo.org/records/3905361/files"
        for set_id in self.set_ids:
            url_set = f"{host_link}/{set_id}.xz"
            path_set = download_url(url_set, self.raw_dir)
        # extract files
        for fn in tqdm(self.raw_paths, desc="Extracting archives"):
            extract_tar(fn, self.raw_dir, mode="r")

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = [f"{set_id}.xz" for set_id in self.set_ids]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data.pt", "properties.npz"]

    def create_data_list(self):
        data_list = []
        property_dict = {
            "Eatoms": [],
            "EPBE0": [],
            "EMBD": [],
            "pbe0FOR": [],
            "vdwFOR": [],
            "DIP": [],
            "POL": [],
        }

        for set_index, set_id in enumerate(self.set_ids):
            # load HDF5 file
            set_path = os.path.join(self.raw_dir, f"{set_id}.hdf5")
            fMOL = h5py.File(set_path, "r")

            # get IDs of HDF5 files and loop through
            mol_ids = list(fMOL.keys())
            for molid in tqdm(
                mol_ids, desc=f"Process molecules in {set_index+1}000.hdf5 file"
            ):
                # get IDs of individual configurations/conformations of molecule
                conf_ids = list(fMOL[molid].keys())

                for confid in conf_ids:
                    # get atomic positions and numbers and add to molecules buffer
                    pos = np.array(fMOL[molid][confid]["atXYZ"])
                    Z = np.array(fMOL[molid][confid]["atNUM"])
                    forces = np.array(fMOL[molid][confid]["totFOR"])
                    Eatoms = sum([self.EPBE0_atom[zi] for zi in Z])
                    EPBE0 = float(list(fMOL[molid][confid]["ePBE0"])[0])  # Local energy
                    EMBD = float(
                        list(fMOL[molid][confid]["eMBD"])[0]
                    )  # Long range dispersion correction
                    energy = EPBE0 + EMBD - Eatoms

                    data = AtomicData.from_points(
                        pos=torch.as_tensor(pos, dtype=torch.float32),
                        atom_types=torch.as_tensor(Z),
                        energy=torch.as_tensor([energy], dtype=torch.float32),
                        forces=torch.as_tensor(forces, dtype=torch.float32),
                    )
                    data_list.append(data)

                    POL = float(list(fMOL[molid][confid]["mPOL"])[0])
                    DIP = float(list(fMOL[molid][confid]["DIP"])[0])
                    vdwFOR = np.array(fMOL[molid][confid]["vdwFOR"])
                    pbe0FOR = np.array(fMOL[molid][confid]["pbe0FOR"])

                    property_dict["Eatoms"].append(Eatoms)
                    property_dict["EPBE0"].append(EPBE0)
                    property_dict["EMBD"].append(EMBD)
                    property_dict["pbe0FOR"].append(pbe0FOR)
                    property_dict["vdwFOR"].append(vdwFOR)
                    property_dict["DIP"].append(DIP)
                    property_dict["POL"].append(POL)

        return data_list, property_dict

    def process(self):
        data_list, property_dict = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])

        # Add batch to property_dict
        batch = np.arange(0, len(property_dict["pbe0FOR"]))
        repeats = [ar.shape[0] for ar in property_dict["pbe0FOR"]]
        property_dict["batch"] = np.repeat(batch, repeats)
        # Concatenate forces
        property_dict["pbe0FOR"] = np.concatenate(property_dict["pbe0FOR"])
        property_dict["vdwFOR"] = np.concatenate(property_dict["vdwFOR"])
        # Save property dict
        np.savez(self.processed_paths[1], **property_dict)
