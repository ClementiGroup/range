from typing import List, Final

import torch

from mlcg.nn import MLP
from mlcg.geometry import compute_distance_vectors
from mlcg.nn.angular_basis.spherical_harmonics import SphericalHarmonics

from .base import RANGE
from ..blocks import (
    RANGEInteractionBlock,
    AttentionBlock,
    SelfAttentionBlock,
    So3kratesBlock,
)
from ..system import System
from ...utils import create_instance


class RANGESo3krates(RANGE):
    name: Final[str] = "RANGESo3krates"

    def __init__(self,
                 embedding_layer: torch.nn.Module,
                 virt_embedding_layer: torch.nn.Module,
                 interaction_blocks: List[torch.nn.Module],
                 basis_instance: torch.nn.Module,
                 cutoff: float,
                 num_virt_nodes: int,
                 virt_basis_instance: torch.nn.Module,
                 regularization_instance: torch.nn.Module,
                 layer_norm: torch.nn.Module,
                 output_network: torch.nn.Module,
                 degrees: List[int],
                 max_num_neighbors: int,
                 normalize_sph: bool):
        super().__init__(embedding_layer,
                         virt_embedding_layer,
                         interaction_blocks,
                         basis_instance,
                         cutoff,
                         num_virt_nodes,
                         virt_basis_instance,
                         regularization_instance,
                         layer_norm,
                         output_network,
                         max_num_neighbors)

        self.degrees = degrees
        self.max_num_neighbors = max_num_neighbors

        # Spherical harmonics computation
        self.sph_harmonics = SphericalHarmonics(
            lmax=max(degrees), normalize=normalize_sph
        )

        # Total number of spherical harmonic coefficients
        self.m_tot = sum(2 * l + 1 for l in degrees)

    def MakeRealSystem(self, data):
        device = data.pos.device
        dtype = data.pos.dtype

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.pos.shape[0],
                                     dtype=torch.long,
                                     device=device)

        system = System()

        system.embedding.append(self.embedding_layer(data.atom_types))  # (n_atoms, hidden_channels)

        # Initialize spherical harmonics features. They can also be initalized with neighbor dependent embeddings
        system.embedding.append(
                torch.zeros(
                    (system.embedding[0].shape[0], self.m_tot), device=device, dtype=dtype
                    )
                )

        system.batch = data.batch

        if nl_goodness := hasattr(data, "neighbor_list"):
            neighbor_list = data.neighbor_list.get(self.name)
            nl_goodness = self.is_nl_compatible(neighbor_list)

        if not nl_goodness:
            neighbor_list = self.neighbor_list(
                    self.name,
                    data,
                    self.cutoff,
                    self.max_num_neighbors)[self.name]

        system.edge_indices = neighbor_list["index_mapping"]

        system.edge_weights, edge_versors = compute_distance_vectors(
            data.pos,
            system.edge_indices,
            neighbor_list["cell_shifts"]
        )

        system.edge_attrs = self.basis(
                system.edge_weights.squeeze(-1)
                )

        sph_ij = self.sph_harmonics(edge_versors)
        # Concatenate only the required degrees
        system.edge_versors = torch.cat([sph_ij[l] for l in self.degrees], dim=-1)  # (n_pairs, sum_l 2l + 1 = m_tot)

        return system


class StandardRANGESo3krates(RANGESo3krates):
    def __init__(
        self,
        num_virt_nodes: int = 1,
        radial_basis_fn: str = 'mlcg.nn.radial_basis.GaussianBasis',
        basis_dim: int = 128,
        virt_basis_dim: int = 10,
        edge_dim: int = 128,
        cutoff_fn: str = 'mlcg.nn.cutoff.CosineCutoff',
        cutoff: float = 3.0,
        output_channels: List[int] = [64,],
        hidden_channels: int = 132,
        num_heads: int = 8,
        num_heads_so3: int = 4,
        regularization_fn: str = 'rangemp.nn.regularization.LinearReg',
        min_num_atoms: int = 5,
        max_num_atoms: int = 100,
        embedding_size: int = 100,
        num_interactions: int = 3,
        degrees: List[int] = [1, 2, 3],
        activation: torch.nn.Module = torch.nn.SiLU(),
        max_num_neighbors: int = 100,
        normalize_sph: bool = True,
        aggr: str = "add",
        epsilon: float = 1e-8,
    ):
        if num_interactions < 1:
            raise ValueError(
                "At least one interaction block must be specified"
                )

        assert hidden_channels % len(degrees) == 0, "hidden_channels must be divisible by len(degrees)."
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads."

        cutoff_instance = create_instance(cutoff_fn, 0.0, cutoff)

        basis_instance = create_instance(radial_basis_fn,
                                         cutoff=cutoff_instance,
                                         num_rbf=basis_dim,
                                         trainable=False)

        virt_basis_instance = create_instance(radial_basis_fn,
                                              cutoff=1.0,
                                              num_rbf=virt_basis_dim,
                                              trainable=False)

        regularization_instance = create_instance(regularization_fn,
                                                  num_virt_nodes=num_virt_nodes,
                                                  min_num_atoms=min_num_atoms,
                                                  max_num_atoms=max_num_atoms)

        fb_rad_filter_features = [hidden_channels, hidden_channels]
        fb_sph_filter_features = [hidden_channels // 4, hidden_channels]
        gb_rad_filter_features = [hidden_channels, hidden_channels]
        gb_sph_filter_features = [hidden_channels // 4, hidden_channels]

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)
        virt_embedding_layer = torch.nn.Embedding(num_virt_nodes, hidden_channels)
        # virt_embedding_layer = torch.nn.utils.parametrizations.orthogonal(virt_embedding_layer)

        attention_block_kwargs = {
            "channels": hidden_channels,
            "activation": activation,
            "basis_dim": virt_basis_dim,
            "n_heads": num_heads
        }

        prop_block_kwargs = {
            "hidden_channels": hidden_channels,
            "degrees": degrees,
            "edge_attr_dim": basis_dim,
            "fb_rad_filter_features": fb_rad_filter_features,
            "fb_sph_filter_features": fb_sph_filter_features,
            "gb_rad_filter_features": gb_rad_filter_features,
            "gb_sph_filter_features": gb_sph_filter_features,
            "n_heads": num_heads_so3,
            "activation": activation,
            "cutoff": cutoff_instance,
            "aggr": aggr,
            "epsilon": epsilon
        }

        interaction_blocks = []
        for _ in range(num_interactions):
            propagation_block = So3kratesBlock(**prop_block_kwargs)

            aggregation_block = AttentionBlock(**attention_block_kwargs)
            broadcast_block = SelfAttentionBlock(**attention_block_kwargs)

            activation_block = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_channels),
                    activation,
                    )

            output_block = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels, bias=False),
                    torch.nn.LayerNorm(hidden_channels),
                    activation,
                    torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
                    )

            block = RANGEInteractionBlock(
                    propagation_block,
                    aggregation_block,
                    broadcast_block,
                    activation_block,
                    output_block
            )
            interaction_blocks.append(block)

        output_layer_widths = (
            [hidden_channels] + output_channels + [1]
        )

        layer_norm = torch.nn.LayerNorm(hidden_channels)

        output_network = MLP(
                output_layer_widths,
                activation_func=activation,
                last_bias=False
                )

        super().__init__(
            embedding_layer,
            virt_embedding_layer,
            interaction_blocks,
            basis_instance,
            cutoff,
            num_virt_nodes,
            virt_basis_instance,
            regularization_instance,
            layer_norm,
            output_network,
            degrees,
            max_num_neighbors,
            normalize_sph,
        )
