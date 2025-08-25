from typing import List, Final, Optional, Dict, Any, Union

import torch

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY

from .base import RANGE
from ..blocks import (
    RANGEInteractionBlock,
    AttentionBlock,
    SelfAttentionBlock,
    MACEBlock,
)
from ..system import System
from ...utils import create_instance

from e3nn import o3

try:
    from mace.modules.radial import ZBLBasis
    from mace.tools.scatter import scatter_sum
    from mace.tools import to_one_hot

    from mace.modules.blocks import (
        LinearNodeEmbeddingBlock,
        RadialEmbeddingBlock,
    )
    from mace.modules.utils import get_edge_vectors_and_lengths

except ImportError as e:
    print(e)
    print(
        "Please install or set mace to your path before using this interface. "
        + "To install you can either run 'pip install git+https://github.com/ACEsuit/mace.git@v0.3.12', "
        + "or clone the repository and add it to your PYTHONPATH."
        ""
    )
from e3nn.util.jit import compile_mode


class RANGEMACE(RANGE):
    name: Final[str] = "RANGEMACE"

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        node_embeddig: torch.nn.Module,
        radial_embeddig: torch.nn.Module,
        spherical_harmonics: torch.nn.Module,
        virt_embedding_layer: torch.nn.Module,
        interaction_blocks: List[torch.nn.Module],
        cutoff: float,
        num_virt_nodes: int,
        virt_basis_instance: torch.nn.Module,
        regularization_instance: torch.nn.Module,
        max_num_neighbors: int,
        pair_repulsion_fn: torch.nn.Module,
    ):

        super().__init__(
            None,
            virt_embedding_layer,
            interaction_blocks,
            None,
            cutoff,
            num_virt_nodes,
            virt_basis_instance,
            regularization_instance,
            None,
            None,
            max_num_neighbors,
        )
        delattr(self, "layer_norm")
        delattr(self, "embedding_layer")
        delattr(self, "output_network")
        delattr(self, "basis")

        self.register_buffer("atomic_numbers", atomic_numbers)
        self.node_embedding = node_embeddig
        self.radial_embedding = radial_embeddig
        self.spherical_harmonics = spherical_harmonics
        self.r_max = cutoff
        self.pair_repulsion_fn = pair_repulsion_fn

        self.register_buffer(
            "types_mapping",
            -1 * torch.ones(atomic_numbers.max() + 1, dtype=torch.long),
        )
        self.types_mapping[atomic_numbers] = torch.arange(atomic_numbers.shape[0])

    def reset_parameters(self) -> None:
        for block in self.interaction_blocks:
            block.reset_parameters()

    def MakeRealSystem(self, data):
        device = data.pos.device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long, device=device)

        types_ids = self.types_mapping[data.atom_types].view(-1, 1)
        node_attrs = to_one_hot(types_ids, self.atomic_numbers.shape[0])

        system = System()
        system.batch = data.batch
        system.embedding = [self.node_embedding(node_attrs)]

        if nl_goodness := hasattr(data, "neighbor_list"):
            neighbor_list = data.neighbor_list.get(self.name)
            nl_goodness = self.is_nl_compatible(neighbor_list)

        if not nl_goodness:
            neighbor_list = self.neighbor_list(
                self.name, data, self.cutoff, self.max_num_neighbors
            )[self.name]

        system.edge_indices = neighbor_list["index_mapping"]

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=system.edge_indices,
            shifts=neighbor_list["cell_shifts"],
        )

        system.edge_attrs = self.spherical_harmonics(vectors)
        system.edge_weights = self.radial_embedding(
            lengths, node_attrs, system.edge_indices, self.atomic_numbers
        )

        system.extra_propagation_args["energy_list"] = []
        system.extra_propagation_args["batch"] = system.batch
        system.extra_propagation_args["node_attrs"] = node_attrs

        return system, lengths

    def forward(self, data: AtomicData) -> AtomicData:
        system, lengths = self.MakeRealSystem(data)
        system = self.MakeVirtSystem(data, system)

        if self.pair_repulsion_fn:
            pair_node_energy = self.pair_repulsion_fn(
                lengths,
                system.extra_propagation_args["node_attrs"],
                system.edge_index,
                self.atomic_numbers,
            )
            pair_energy = scatter_sum(
                src=pair_node_energy,
                index=data.batch,
                dim=-1,
                dim_size=data.batch.max(),
            )  # [n_graphs,]
        else:
            pair_energy = torch.zeros(
                data.batch.max() + 1,
                device=data.pos.device,
                dtype=data.pos.dtype,
            )

        system.extra_propagation_args["energy_list"].append(pair_energy)

        for block_update in self.interaction_blocks:
            block_update(system)

        # Sum over energy contributions
        contributions = torch.stack(
            system.extra_propagation_args["energy_list"], dim=-1
        )
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        data.out[self.name] = {
            ENERGY_KEY: total_energy,
            "regL1": self.regularization.regularized_params(),
        }

        return data


@compile_mode("script")
class StandardRANGEMACE(RANGEMACE):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: str,
        interaction_cls_first: str,
        num_interactions: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Dict[str, Any]] = None,
        num_virt_nodes: int = 1,
        radial_basis_fn: str = "mlcg.nn.radial_basis.GaussianBasis",
        virt_basis_dim: int = 10,
        num_virt_heads: int = 8,
        regularization_fn: str = "rangemp.nn.regularization.LinearReg",
        min_num_atoms: int = 5,
        max_num_atoms: int = 100,
    ):
        if num_interactions < 1:
            raise ValueError("At least one interaction block must be specified")

        atomic_numbers.sort()
        atomic_numbers = torch.as_tensor(atomic_numbers)
        num_elements = atomic_numbers.shape[0]

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")

        pair_repulsion_fn = None
        if pair_repulsion:
            pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Consider only scalar embedding for RANGE
        hidden_channels = hidden_irreps[0].dim

        attention_block_kwargs = {
            "channels": hidden_channels,
            "activation": gate,
            "basis_dim": virt_basis_dim,
            "n_heads": num_virt_heads,
        }

        prop_block_kwargs = {
            "interaction_cls": interaction_cls_first,
            "node_attr_irreps": node_attr_irreps,
            "node_feats_irreps": node_feats_irreps,
            "edge_attrs_irrep": sh_irreps,
            "edge_feats_irreps": edge_feats_irreps,
            "target_irreps": interaction_irreps,
            "num_elements": num_elements,
            "avg_num_neighbors": avg_num_neighbors,
            "radial_MLP": radial_MLP,
            "cueq_config": None,
            "gate": gate,
            "MLP_irreps": MLP_irreps,
        }

        interaction_blocks = []
        for i in range(num_interactions):
            if i == num_interactions - 1:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
                linear_readout = False
            else:
                hidden_irreps_out = hidden_irreps
                linear_readout = True
            prop_block_kwargs["hidden_irreps"] = hidden_irreps_out
            prop_block_kwargs["linear_readout"] = linear_readout

            propagation_block = MACEBlock(
                **prop_block_kwargs, correlation=correlation[i]
            )
            prop_block_kwargs["interaction_cls"] = interaction_cls
            prop_block_kwargs["node_feats_irreps"] = hidden_irreps

            aggregation_block = AttentionBlock(**attention_block_kwargs)
            broadcast_block = SelfAttentionBlock(**attention_block_kwargs)

            activation_block = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_channels),
                gate,
            )

            output_block = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels, bias=False),
                torch.nn.LayerNorm(hidden_channels),
                gate,
                torch.nn.Linear(hidden_channels, hidden_channels, bias=False),
            )

            block = RANGEInteractionBlock(
                propagation_block,
                aggregation_block,
                broadcast_block,
                activation_block,
                output_block,
            )
            interaction_blocks.append(block)

        virt_embedding_layer = torch.nn.Embedding(num_virt_nodes, hidden_channels)
        virt_basis_instance = create_instance(
            radial_basis_fn, cutoff=1.0, num_rbf=virt_basis_dim, trainable=False
        )

        regularization_instance = create_instance(
            regularization_fn,
            num_virt_nodes=num_virt_nodes,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms,
        )

        super().__init__(
            atomic_numbers,
            node_embedding,
            radial_embedding,
            spherical_harmonics,
            virt_embedding_layer,
            interaction_blocks,
            r_max,
            num_virt_nodes,
            virt_basis_instance,
            regularization_instance,
            max_num_neighbors,
            pair_repulsion_fn,
        )
