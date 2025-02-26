import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from mlcg.utils import download_url
from mlcg.data import AtomicData


class AQMBASEDataset(InMemoryDataset):
    """General base extractor for AQM dataset described in https://doi.org/10.1038/s41597-024-03521-8"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def set_temperature(self, temperature: float):
        """Set temperature for the dataset. Expected temperature in K"""
        self.temperature = temperature
        self.beta = 1 / (self.temperature * self.beta)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        raise NotImplementedError

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data.pt"]

    def create_data_list(self):
        data_list = []
        base_dataset_path = os.path.join(self.root, "raw", self.raw_file_names[0])
        fMOL = h5py.File(base_dataset_path)
        AQMmol_ids = list(fMOL.keys())
        for molid in tqdm(AQMmol_ids, desc="Processing molecules..."):
            AQMconf_ids = list(fMOL[molid].keys())
            for confid in AQMconf_ids:
                pos = np.array(fMOL[molid][confid]["atXYZ"])
                forces = np.array(fMOL[molid][confid]["totFOR"])
                Z = np.array(fMOL[molid][confid]["atNUM"])
                EAT = float(
                    list(fMOL[molid][confid]["eAT"])[0]
                )  # single atom contribution - PBE0 energy
                EMBD = float(
                    list(fMOL[molid][confid]["eMBD"])[0]
                )  # Long range dispersion correction
                energy = EMBD - EAT

                data = AtomicData.from_points(
                    pos=torch.as_tensor(pos, dtype=torch.float32),
                    atom_types=torch.as_tensor(Z),
                    energy=torch.as_tensor(energy, dtype=torch.float32),
                    forces=torch.as_tensor(forces, dtype=torch.float32),
                )
                data_list.append(data)

        return data_list

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])


class AQMgasDataset(AQMBASEDataset):
    """Extractor for AQM dataset in vacuum described in https://doi.org/10.1038/s41597-024-03521-8"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "https://zenodo.org/records/10208010/files/AQM-gas.hdf5"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["AQM-gas.hdf5"]
        return raw_set_list


class AQMsolDataset(AQMBASEDataset):
    """Extractor for AQM dataset with solvent described in https://doi.org/10.1038/s41597-024-03521-8"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "https://zenodo.org/records/10208010/files/AQM-sol.hdf5"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["AQM-sol.hdf5"]
        return raw_set_list
