from dataclasses import dataclass, field

import torch


@dataclass
class System:
    """
    A data structure representing a system with various tensors and embeddings,
    used in graph-based computations.

    Attributes:
        batch (torch.Tensor, optional):
            A tensor indicating the batch assignments of nodes.
        embedding (list[torch.Tensor]):
            A list of tensors representing node embeddings.
        virt_embedding (torch.Tensor, optional):
            A tensor representing virtual node embeddings.
        edge_indices (torch.Tensor, optional):
            A tensor containing edge index information.
        edge_weights (torch.Tensor, optional):
            A tensor storing the edge weights.
        edge_versors (torch.Tensor, optional):
            A tensor containing directional unit edge vectors.
        edge_attrs (torch.Tensor, optional):
            A tensor storing additional edge features.
        aggregation_edge_indices (torch.Tensor, optional):
            A tensor defining indices for aggregation edges in RANGE.
        aggregation_edge_attrs (torch.Tensor, optional):
            A tensor storing edge features for aggregation.
        broadcast_edge_indices (torch.Tensor, optional):
            A tensor defining indices for broadcast edges in RANGE.
        broadcast_edge_attrs (torch.Tensor, optional):
            A tensor storing edge features for broadcast.
        regularization_weights (torch.Tensor, optional):
            A tensor storing weights for regularization.
        extra_propagation_args: dict
            Dictionary of eventual extra argument for propagation block.
    """

    batch: torch.Tensor = None

    embedding: list[torch.Tensor] = field(default_factory=list)
    virt_embedding: torch.Tensor = None

    edge_indices: torch.Tensor = None
    edge_weights: torch.Tensor = None
    edge_versors: torch.Tensor = None
    edge_attrs: torch.Tensor = None

    aggregation_edge_indices: torch.Tensor = None
    aggregation_edge_attrs: torch.Tensor = None

    broadcast_edge_indices: torch.Tensor = None
    broadcast_edge_attrs: torch.Tensor = None

    regularization_weights: torch.Tensor = None
    extra_propagation_args: dict = field(default_factory=dict)
