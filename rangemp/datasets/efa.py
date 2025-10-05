import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_zip

from mlcg.utils import download_url
from mlcg.data import AtomicData


class EFADataset(InMemoryDataset):
    """General base extractor for the EFA dataset described in https://doi.org/10.48550/arXiv.2412.08541"""

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
        host_url = "https://zenodo.org/records/17130729/files/source_data.zip?download=1"
        archive_path = download_url(host_url, self.raw_dir)
        extract_zip(path=archive_path, folder=self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        """List of default processed files."""
        raise NotImplementedError

    def create_data_list(self):
        data_list = []

        base_dataset_path = os.path.join(self.root, 'raw', self.raw_file_names[0])
        print(base_dataset_path)
        df = np.load(base_dataset_path)
        for idx in range(df['positions'].shape[0]):
            data = self.parse_structure(df, idx)
            if data is not None:
                data_list.append(data)

        return data_list

    def parse_structure(self, df, idx):
        raise NotImplementedError

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])


class CumuleneDataset(EFADataset):
    """Extractor for the cumulene dataset described in https://doi.org/10.48550/arXiv.2412.08541"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["source_data/datasets/cumulene/cumulene.npz"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["cumulene.pt"]

    @property
    def atom_energy(self):
        atom_energy = {1: -2.6544, 6: -5.9724}
        return atom_energy

    def parse_structure(self, df, idx):
        data = None

        pos = df['positions'][idx]
        atom_types = df['atomic_numbers'][idx]
        energy = df['energy'][idx]
        forces = df['forces'][idx]

        shift = np.sum([self.atom_energy[k] for k in atom_types])
        energy -= shift

        data = AtomicData.from_points(
            pos=torch.as_tensor(pos, dtype=torch.float32),
            atom_types=torch.as_tensor(atom_types),
            energy=torch.as_tensor(energy, dtype=torch.float32),
            forces=torch.as_tensor(forces, dtype=torch.float32),
        )

        return data


class Sn2Dataset(EFADataset):
    """Extractor for the Sn2 dataset described in https://doi.org/10.48550/arXiv.2412.08541"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["source_data/datasets/sn2/sn2_reactions_shifted.npz"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["sn2.pt"]

    def parse_structure(self, df, idx):
        data = None

        pos = df['positions'][idx][df['node_mask'][idx]]
        atom_types = df['atomic_numbers'][idx][df['node_mask'][idx]]
        energy = df['energy'][idx]
        forces = df['forces'][idx][df['node_mask'][idx]]

        data = AtomicData.from_points(
            pos=torch.as_tensor(pos, dtype=torch.float32),
            atom_types=torch.as_tensor(atom_types),
            energy=torch.as_tensor(energy, dtype=torch.float32),
            forces=torch.as_tensor(forces, dtype=torch.float32),
        )

        return data
