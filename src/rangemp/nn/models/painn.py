from typing import List, Final

import torch

from mlcg.nn import MLP
from mlcg.geometry import compute_distance_vectors

from .base import RANGE
from ..blocks import (
    RANGEInteractionBlock,
    AggregationBlock,
    BroadcastBlock,
    PaiNNBlock,
)
from ..system import System
from ...utils import create_instance


class RANGEPaiNN(RANGE):
    """
    RANGE adapter using the PaiNN equivariant interaction and mixing modules.

    Wraps PaiNN building blocks so they can be executed inside the RANGE
    virtual-node framework. Converts AtomicData into the System structure used by
    RANGE, computes edge distances/versors and radial features for PaiNN interaction.

    Initialization parameters
    -------------------------
    embedding_layer : torch.nn.Module
        Embedding module mapping atomic types -> initial node features.
    virt_embedding_layer : torch.nn.Module
        Embedding module producing per-virtual-node features.
    interaction_blocks : List[torch.nn.Module]
        Sequence of RANGEInteractionBlock instances containing PaiNN propagation,
        aggregation and broadcast sub-blocks.
    basis_instance : torch.nn.Module
        Radial basis instance used to embed edge distances.
    cutoff : float
        Cutoff radius used for neighbor list construction and radial basis.
    num_virt_nodes : int
        Number of virtual nodes per graph / sample.
    virt_basis_instance : torch.nn.Module
        Basis instance for virtual->node edges.
    regularization_instance : torch.nn.Module
        Regularizer producing per-graph statistics for training.
    layer_norm : torch.nn.Module
        LayerNorm instance applied before final output MLP.
    output_network : torch.nn.Module
        MLP mapping node embeddings to per-node energy contributions.
    max_num_neighbors : int
        Hard cap for neighbor list size per node.
    """

    name: Final[str] = "RANGEPaiNN"

    def __init__(
        self,
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
        max_num_neighbors: int,
    ):
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
            max_num_neighbors,
        )

    def MakeRealSystem(self, data):
        device = data.pos.device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long, device=device)

        system = System()
        system.embedding.append(self.embedding_layer(data.atom_types))
        system.embedding.append(
            torch.zeros(
                (system.embedding[0].shape[0], 3, system.embedding[0].shape[1]),
                device=system.embedding[0].device,
            )
        )  # (n_atoms, 3, n_features)

        system.batch = data.batch

        if nl_goodness := hasattr(data, "neighbor_list"):
            neighbor_list = data.neighbor_list.get(self.name)
            nl_goodness = self.is_nl_compatible(neighbor_list)

        if not nl_goodness:
            neighbor_list = self.neighbor_list(
                self.name, data, self.cutoff, self.max_num_neighbors
            )[self.name]

        system.edge_indices = neighbor_list["index_mapping"]

        system.edge_weights, system.edge_versors = compute_distance_vectors(
            data.pos, system.edge_indices, neighbor_list["cell_shifts"]
        )

        system.edge_weights = system.edge_weights.squeeze(1)

        system.edge_attrs = self.basis(system.edge_weights)

        return system


class StandardRANGEPaiNN(RANGEPaiNN):
    """
    Helper factory to construct a typical RANGEPaiNN configuration from high-level
    hyperparameters.

    Parameters
    ----------
    num_virt_nodes : int
        Number of virtual nodes per graph.
    radial_basis_fn : str
        Path to radial basis constructor.
    basis_dim : int
        Number of radial basis functions for real edges.
    virt_basis_dim : int
        Radial basis size for virtual edges.
    edge_dim : int
        Hidden dimension inside propagation blocks.
    cutoff_fn : str
        Cutoff constructor path.
    cutoff : float
        Cutoff radius used for neighbor list construction.
    output_channels : List[int]
        MLP hidden widths for the output network (final layer produces scalar).
    hidden_channels : int
        Node hidden channel size.
    virt_hidden_channels : int
        Virtual node hidden channel size.
    num_virt_heads : int
        Number of attention heads for aggregation/broadcasting.
    regularization_fn : str
        Regularizer constructor path.
    min_num_atoms, max_num_atoms : int
        Min/max atoms used to configure regularizer statistics.
    embedding_size : int
        Number of embedding indices (vocabulary) for node types.
    num_interactions : int
        Number of PaiNN interaction layers to stack.
    activation : torch.nn.Module
        Activation function used across blocks.
    max_num_neighbors : int
        Hard cap for neighbor count per node.
    aggr : str
        Aggregation mode passed to PaiNN interaction ('add'/'mean').
    epsilon : float
        Small value used by PaiNN mixing block where applicable.

    """

    def __init__(
        self,
        num_virt_nodes: int = 1,
        radial_basis_fn: str = "mlcg.nn.radial_basis.GaussianBasis",
        basis_dim: int = 128,
        virt_basis_dim: int = 10,
        edge_dim: int = 128,
        cutoff_fn: str = "mlcg.nn.cutoff.CosineCutoff",
        cutoff: float = 3.0,
        output_channels: List[int] = [
            64,
        ],
        hidden_channels: int = 144,
        virt_hidden_channels: int = 432,
        num_virt_heads: int = 8,
        regularization_fn: str = "rangemp.nn.regularization.LinearReg",
        min_num_atoms: int = 5,
        max_num_atoms: int = 100,
        embedding_size: int = 100,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 100,
        aggr: str = "add",
        epsilon: float = 1e-8,
    ):
        if num_interactions < 1:
            raise ValueError("At least one interaction block must be specified")

        cutoff_instance = create_instance(cutoff_fn, 0.0, cutoff)

        basis_instance = create_instance(
            radial_basis_fn, cutoff=cutoff_instance, num_rbf=basis_dim, trainable=False
        )

        virt_basis_instance = create_instance(
            "mlcg.nn.GaussianBasis", cutoff=1.0, num_rbf=virt_basis_dim, trainable=False
        )

        regularization_instance = create_instance(
            regularization_fn,
            num_virt_nodes=num_virt_nodes,
            min_num_atoms=min_num_atoms,
            max_num_atoms=max_num_atoms,
        )

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)
        virt_embedding_layer = torch.nn.Embedding(num_virt_nodes, virt_hidden_channels)
        # virt_embedding_layer = torch.nn.utils.parametrizations.orthogonal(virt_embedding_layer)

        aggr_block_kwargs = {
            "in_channels": hidden_channels,
            "out_channels": virt_hidden_channels,
            "activation": activation,
            "basis_dim": virt_basis_dim,
            "n_heads": num_virt_heads,
        }

        bcast_block_kwargs = {
            "in_channels": virt_hidden_channels,
            "out_channels": hidden_channels,
            "activation": activation,
            "basis_dim": virt_basis_dim,
            "n_heads": num_virt_heads,
        }

        prop_block_kwargs = {
            "channels": hidden_channels,
            "basis_dim": basis_dim,
            "activation": activation,
            "cutoff": cutoff_instance,
            "aggr": aggr,
            "epsilon": epsilon,
        }

        interaction_blocks = []
        for _ in range(num_interactions):
            propagation_block = PaiNNBlock(**prop_block_kwargs)

            aggregation_block = AggregationBlock(**aggr_block_kwargs)
            broadcast_block = BroadcastBlock(**bcast_block_kwargs)

            block = RANGEInteractionBlock(
                propagation_block,
                aggregation_block,
                broadcast_block,
            )
            interaction_blocks.append(block)

        output_layer_widths = [hidden_channels] + output_channels + [1]

        layer_norm = torch.nn.LayerNorm(hidden_channels)

        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=False
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
            max_num_neighbors,
        )
