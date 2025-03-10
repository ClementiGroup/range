from typing import List

import torch
from torch_scatter import scatter

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist
)

from ..blocks import RANGEInteractionBlock
from ..utils import calc_weights, calc_weights_pbc


class RANGE(torch.nn.Module):
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
                 max_num_neighbors: int):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.virt_embedding_layer = virt_embedding_layer
        self.basis = basis_instance
        self.virt_basis = virt_basis_instance
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_virt_nodes = num_virt_nodes
        self.regularization = regularization_instance

        # Check interaction_blocks
        if isinstance(interaction_blocks, List):
            self.interaction_blocks = torch.nn.Sequential(*interaction_blocks)
        elif isinstance(interaction_blocks, RANGEInteractionBlock):
            self.interaction_blocks = torch.nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )

        self.layer_norm = layer_norm
        self.output_network = output_network

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding_layer.reset_parameters()
        self.basis.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def MakeRealSystem(self, data):
        raise NotImplementedError

    def MakeVirtSystem(self, data, system):
        device = data.pos.device

        virt_embedding = []
        for level_index in range(self.num_virt_nodes):
            virt_embedding.append(
                    self.virt_embedding_layer(torch.tensor([level_index], device=device)).repeat(data.batch.max() + 1, 1)
                    )
        system.virt_embedding = torch.cat(virt_embedding)

        range_edge_indices = torch.stack(
                (torch.arange(data.pos.shape[0], device=device),
                 data.batch)
                )

        if "pbc" in data and data.pbc.any():
            weights = calc_weights_pbc(data)
        else:
            weights = calc_weights(data)

        range_edge_attrs = self.virt_basis(weights)

        index_0 = range_edge_indices[0].repeat(self.num_virt_nodes)
        index_1 = (range_edge_indices[1].repeat(self.num_virt_nodes).view(self.num_virt_nodes, -1) +
                   (torch.arange(self.num_virt_nodes, device=device)*(range_edge_indices[1].max() + 1)).unsqueeze(1)).flatten()

        system.aggregation_edge_indices = torch.stack((index_0, index_1))
        system.aggregation_edge_attrs = range_edge_attrs.repeat(self.num_virt_nodes, 1)

        index_0 = torch.cat((system.aggregation_edge_indices[1],
                             torch.arange(system.aggregation_edge_indices[1].max() + 1,
                                          system.aggregation_edge_indices[1].max() + 1 + system.embedding[0].shape[0],
                                          device=device)))

        index_1 = torch.arange(data.pos.shape[0], device=device).repeat(self.num_virt_nodes + 1)

        system.broadcast_edge_indices = torch.stack((index_0, index_1))

        system.broadcast_edge_attrs = torch.cat(
                (system.aggregation_edge_attrs,
                 torch.zeros((system.embedding[0].shape[0],
                              range_edge_attrs.shape[1]), device=device))
                )

        system.regularization_weights = self.regularization(data.n_atoms)

        return system

    def forward(self, data: AtomicData) -> AtomicData:
        system = self.MakeRealSystem(data)
        system = self.MakeVirtSystem(data, system)

        for block_update in self.interaction_blocks:
            block_update(system)

        energies = self.output_network(
                self.layer_norm(
                    system.embedding[0]
                    )
                )

        energy = scatter(
                energies,
                system.batch,
                dim=0,
                reduce='add'
                ).flatten()

        data.out[self.name] = {ENERGY_KEY: energy, 'regL1': self.regularization.regularized_params()}

        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.cutoff
            ):
                is_compatible = True
        return is_compatible

    @staticmethod
    def neighbor_list(name: str,
                      data: AtomicData,
                      rcut: float,
                      max_num_neighbors: int = 1000) -> dict:
        return {
            name:
                atomic_data2neighbor_list(data,
                                          rcut,
                                          self_interaction=False,
                                          max_num_neighbors=max_num_neighbors)
            }
