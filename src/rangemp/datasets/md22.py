import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from mlcg.utils import download_url
from mlcg.data import AtomicData


class BaseMD22Dataset(InMemoryDataset):
    """Base class for extract data from the MD22 dataset.
    Units used in all the MD22 datasets are:
        - pos: [A]
        - forces: [kcal/mol/A]
        - energy: [kcal/mol]

    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
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
                    energy=torch.as_tensor([ENERGIES[confid]], dtype=torch.float32),
                    forces=torch.as_tensor(FORCES[confid], dtype=torch.float32),
                )
                data_list.append(data)

        return data_list

    def process(self):
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])


class AcAla3NHMeDataset(BaseMD22Dataset):
    """Ac-Ala3-NHMe dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = (
            "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz"
        )
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_Ac-Ala3-NHMe.npz"]
        return raw_set_list


class DHADataset(BaseMD22Dataset):
    """Docosahexaenoic acid dataset as part of MD22."""

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


class StachyoseDataset(BaseMD22Dataset):
    """Stachyose dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "http://www.quantum-machine.org/gdml/repo/datasets/md22_stachyose.npz"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_stachyose.npz"]
        return raw_set_list


class ATATDataset(BaseMD22Dataset):
    """DNA base pair (AT-AT) dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT.npz"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_AT-AT.npz"]
        return raw_set_list


class ATATCGCGDataset(BaseMD22Dataset):
    """DNA base pair (AT-AT-CG-CG) dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = (
            "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz"
        )
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_AT-AT-CG-CG.npz"]
        return raw_set_list


class BuckyballCatcherDataset(BaseMD22Dataset):
    """Buckyball catcher dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_buckyball-catcher.npz"]
        return raw_set_list


class DoubleWallNanotubeDataset(BaseMD22Dataset):
    """Double wall nanotube dataset as part of MD22."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        """Download the datas and store them in self.raw_dir directory"""
        url_set = "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz"
        download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["md22_double-walled_nanotube.npz"]
        return raw_set_list
