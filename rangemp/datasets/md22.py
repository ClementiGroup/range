import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from mlcg.utils import download_url
from mlcg.data import AtomicData


class BaseMD22Dataset(InMemoryDataset):
    """Base class for extract data from the MD22 dataset"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        for fl in self.raw_file_names:
            base_dataset_path = os.path.join(self.root, "raw", fl)
            datas = np.load(base_dataset_path)
            POSITIONS = datas["R"]
            Z = datas["z"]
            FORCES = datas["F"]
            EMEAN = np.mean(datas["E"])
            ENERGIES = datas["E"] - EMEAN

            for confid in tqdm(
                range(datas["E"].shape[0]), desc="Extracting dataset..."
            ):

                data = AtomicData.from_points(
                    pos=torch.as_tensor(POSITIONS[confid], dtype=torch.float32),
                    atom_types=torch.as_tensor(Z),
                    energy=torch.as_tensor(ENERGIES[confid], dtype=torch.float32),
                    forces=torch.as_tensor(FORCES[confid], dtype=torch.float32),
                )
                data_list.append(data)

        return data_list

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])


class DHADataset(BaseMD22Dataset):
    """Dataset for Docosahexaenoic acid MD22"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "http://www.quantum-machine.org/gdml/repo/datasets/md22_DHA.npz"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_DHA.npz"]
        return raw_set_list
