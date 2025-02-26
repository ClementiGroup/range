from dataclasses import dataclass, field

import torch


@dataclass
class System:
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
