import numpy as np
import torch
import re
import os
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset

from mlcg.utils import download_url
from mlcg.data import AtomicData


class DimersDataset(InMemoryDataset):
    """General base extractor for the Dimers dataset used in LODE https://doi.org/10.1038/s41597-024-03521-8"""

    _mapping = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
    }

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def set_temperature(self, temperature: float):
        """Set temperature for the dataset. Expected temperature in K"""
        self.temperature = temperature
        self.beta = 1 / (self.temperature * self.beta)

    def download(self):
        """Download the data and store them in self.raw_dir directory"""
        url_sets = [
            "https://archive.materialscloud.org/records/405an-d8183/files/bio_dimers.xyz?download=1",
            "https://archive.materialscloud.org/records/405an-d8183/files/bio_dimers_monomers.xyz?download=1",
        ]
        for url_set in url_sets:
            download_url(url_set, self.raw_dir)

    @property
    def raw_file_names(self):
        """Return a list of local downloaded files processed to create the dataset."""
        raw_set_list = ["bio_dimers.xyz", "bio_dimers_monomers.xyz"]
        return raw_set_list

    @property
    def processed_file_names(self):
        """List of default processed files."""
        return ["data.pt", "atomic_enrgy.pt"]

    @staticmethod
    def iterate_extended_xyz(file_path):
        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break  # End of file
                if not line.strip().isdigit():
                    continue  # Skip until we find a number of atoms

                natoms = int(line.strip())
                header = f.readline().strip()

                # Parse metadata
                metadata = {}
                for keyval in re.findall(r'(\w+)=(".*?"|[^\s]+)', header):
                    key, val = keyval
                    val = val.strip('"')
                    try:
                        metadata[key] = float(val)
                    except ValueError:
                        metadata[key] = val

                elements = []
                positions = []
                forces = []

                for _ in range(natoms):
                    atom_line = f.readline().strip()
                    if not atom_line:
                        break  # malformed file
                    tokens = atom_line.split()
                    elements.append(tokens[0])
                    positions.append([float(x) for x in tokens[1:4]])
                    forces.append([float(x) for x in tokens[4:7]])
                if metadata.get("pbc") is None:
                    pbc = torch.tensor([False, False, False], dtype=torch.bool)
                else:
                    pbc = torch.tensor(
                        [bool(id) for id in metadata["pbc"].split(" ")],
                        dtype=torch.bool,
                    )
                if metadata.get("Lattice") is None:
                    lattice = torch.eye(3, dtype=torch.float32)
                else:
                    lattice = (
                        torch.tensor(
                            [float(id) for id in metadata.get("Lattice").split(" ")],
                            dtype=torch.float32,
                        )
                    )

                yield {
                    "elements": elements,
                    "positions": torch.tensor(positions, dtype=torch.float32),
                    "forces": torch.tensor(forces, dtype=torch.float32),
                    "cell": lattice,
                    "pbc": pbc,
                    "metadata": metadata,
                }

    def create_data_list(self):
        atom_energy = {1:  -16.3878, 6: -1037.1395, 7: -1489.3353, 8: -2048.7069}

        data_list = []
        for filename in self.raw_file_names:
            filepath = os.path.join(self.raw_dir, filename)
            for i, structure in tqdm(
                enumerate(self.iterate_extended_xyz(filepath)),
                desc=f"Processing structures {filename}...",
            ):

                pos = structure["positions"]
                atom_types = [self._mapping[el] for el in structure["elements"]]
                energy = structure["metadata"].get("energy")
                forces = structure["forces"]
                pbc = structure["pbc"]
                cell = structure["cell"]

                shift = np.sum([atom_energy[k] for k in atom_types])
                energy -= shift

                data = AtomicData.from_points(
                    pos=pos,
                    atom_types=torch.tensor(atom_types),
                    energy=torch.tensor([energy], dtype=torch.float32),
                    forces=forces,
                    pbc=pbc,
                    cell=cell,
                )
                data_list.append(data)

        return data_list

    def process(self):
        print("Creating data list from raw files...")
        data_list = self.create_data_list()

        # Store data as .pt file
        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])
