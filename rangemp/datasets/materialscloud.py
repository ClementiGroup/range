import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_gz, extract_tar

from mlcg.utils import download_url
from mlcg.data import AtomicData


class MaterialsCloudDataset(InMemoryDataset):
    """General base extractor for the MaterialsCloud dataset described in https://doi.org/10.1038/s41467-020-20427-2"""

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
        host_url = "https://archive.materialscloud.org/records/bk6kw-4b895/files/datasets.tar.gz"
        archive_path = download_url(host_url, self.raw_dir)
        extract_gz(path=archive_path, folder=self.raw_dir)
        extract_tar(path=archive_path[:-3], folder=self.raw_dir, mode='r')

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
        with open(base_dataset_path, 'r') as f:
            lines = f.readlines()
            idx = 0
            while idx < len(lines):
                if lines[idx].strip() == 'begin':
                    data, idx_new = self.parse_structure(lines, idx)
                    if data is not None:
                        data_list.append(data)
                        idx = idx_new
                idx += 1

        return data_list

    def parse_structure(self, lines, start_idx):
        element_map = { 'H':  1,  'C':  6,  'O':  8,
                       'Na': 11, 'Mg': 12, 'Al': 13,
                       'Cl': 17, 'Ag': 47, 'Au': 79}

        atom_energy = {1: -9.29107507032097,
                       6: -1036.5461224721375,
                       8: -18599.43617104475,
                       11: -4417.07609365649,
                       12: -8721.75974245582,
                       13: -9877.676428588728,
                       17: -12516.880649933015,
                       47: -146385.11440723907,
                       79: -688.8680063349827}

        data = None

        positions = []
        cell_vec = []
        atom_types = []
        atom_charges = []
        atom_energies = []
        forces = []
        total_energy = None
        total_charge = None

        idx = start_idx
        line = lines[idx].strip()
        while line != 'end':
            if line.startswith('lattice'):
                parts = line.split()
                cell_vec.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('atom'):
                parts = line.split()
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                atom_types.append(element_map[parts[4]])
                atom_charges.append(float(parts[5]))
                atom_energies.append(float(parts[6]))
                forces.append([float(parts[7]), float(parts[8]), float(parts[9])])
            elif line.startswith('energy'):
                parts = line.split()
                total_energy = float(parts[1])
            elif line.startswith('charge'):
                parts = line.split()
                total_charge = float(parts[1])

            idx += 1
            line = lines[idx].strip()

        shift = np.sum([atom_energy[k] for k in atom_types])/27.211386245981  # Convert eV to Hartree
        total_energy -= shift

        data = AtomicData.from_points(
            pos=torch.as_tensor(positions, dtype=torch.float32),
            cell=torch.as_tensor(cell_vec, dtype=torch.float32),
            pbc=torch.as_tensor([True, True, True], dtype=torch.bool),
            atom_types=torch.as_tensor(atom_types),
            energy=torch.as_tensor([total_energy], dtype=torch.float32),
            forces=torch.as_tensor(forces, dtype=torch.float32),
            charges=torch.as_tensor([total_charge], dtype=torch.float32),
        )

        return data, idx

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])


class AgClusterDataset(MaterialsCloudDataset):
    """Extractor for the Ag cluster dataset described in https://doi.org/10.1038/s41467-020-20427-2"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["datasets/Ag_cluster/input.data"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data_Ag-cluster.pt"]


class AuMgODataset(MaterialsCloudDataset):
    """Extractor for the AuMgO dataset described in https://doi.org/10.1038/s41467-020-20427-2"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["datasets/AuMgO/input.data"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data_AuMgO.pt"]


class CarbonChainDataset(MaterialsCloudDataset):
    """Extractor for the Carbon chain dataset described in https://doi.org/10.1038/s41467-020-20427-2"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["datasets/Carbon_chain/input.data"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data_carbon-chain.pt"]


class NaClDataset(MaterialsCloudDataset):
    """Extractor for the NaCl dataset described in https://doi.org/10.1038/s41467-020-20427-2"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["datasets/NaCl/input.data"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data_NaCl.pt"]
