from typing import List

import h5py
import os
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

# from mlcg.utils import download_url
from mlcg.data import AtomicData


class MBPolDataset(InMemoryDataset):
    """Extractor for the MBPol dataset described in https://doi.org/10.1038/s41467-023-38855-1"""

    def __init__(
        self,
        root,
        datasets: List[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.datasets = datasets
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
        # host_link = ""
        for filename in self.raw_file_names():
            # url_set = f"{host_link}/{filename}"
            # download_url(url_set, self.raw_dir)
            set_path = os.path.join(self.raw_dir, filename)
            if not os.path.exists(set_path):
                raise FileExistsError(
                    (
                        f"Set {filename} is not downloaded in the raw folder. "
                    )
                )

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        return ["dataset.hdf5"]

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data.pt"]

    def create_data_list(self):
        data_list = []

        for filename in self.raw_file_names:
            # load HDF5 file
            set_path = os.path.join(self.raw_dir, filename)
            fMOL = h5py.File(set_path, "r")

            # get IDs of HDF5 files and loop through
            if self.datasets:
                assert(set(self.datasets) < set(fMOL.keys())), F"One or more datasets not present in the datafile:\n{set(self.datasets) - set(fMOL.keys())}"
                datasets = self.datasets
            else:
                datasets = list(fMOL.keys())

            for dataset in tqdm(datasets, desc=f"Processing datasets..."):
                atnum = fMOL[dataset]["atnum"]
                box = fMOL[dataset]["box"]
                coord = fMOL[dataset]["coord"]
                energy = fMOL[dataset]["energy"]
                forces = fMOL[dataset]["forces"]
                virial = fMOL[dataset]["virial"]
                for confid in range(box.shape[0]):
                    data = AtomicData.from_points(
                        pos=torch.as_tensor(coord[confid], dtype=torch.float32),
                        cell=torch.as_tensor(box[confid], dtype=torch.float32),
                        pbc=torch.tensor([True, True, True], dtype=torch.bool),
                        atom_types=torch.as_tensor(atnum, dtype=torch.int32),
                        energy=torch.as_tensor([energy[confid]], dtype=torch.float32),
                        forces=torch.as_tensor(forces[confid], dtype=torch.float32),
                        virial=torch.as_tensor(virial[confid], dtype=torch.float32).view(-1, 3, 3),
                    )
                    data_list.append(data)

        return data_list

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])
