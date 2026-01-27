from typing import List

import torch
from ..scatter import scatter

from mlcg.data import AtomicData
from mlcg.data._keys import ENERGY_KEY
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from ..blocks import RANGEInteractionBlock
from ..utils import calc_weights, calc_weights_pbc


class RANGE(torch.nn.Module):
    """
    RANGE model coordinating embedding, interaction and output networks for graph systems.

    This module is a wrapper that constructs a system with virtual system used in RANGE,
    applies a sequence of interaction blocks, and reduces per-node contributions
    to per-graph energies.

    To apply with a specific baseline model the method MakeRealSystem has to be implemented.

    Initialization parameters
    -------------------------
    embedding_layer : nn.Module
        Module creating initial node embeddings from atomic attributes.
    virt_embedding_layer : nn.Module
        Module generating virtual-node embeddings (one per virtual level).
    interaction_blocks : List[nn.Module] | RANGEInteractionBlock
        One or many interaction blocks applied sequentially.
    basis_instance : nn.Module
        Basis (radial/edge) used by interaction modules for real edges.
    cutoff : float
        Cutoff distance used to construct neighbor lists / edge weighting.
    num_virt_nodes : int
        Number of virtual nodes per sample.
    virt_basis_instance : nn.Module
        Basis used to compute virtual->node edge attributes.
    regularization_instance : nn.Module
        Module providing regularization weights and statistics.
    layer_norm : nn.Module
        Layer normalization applied before the output network.
    output_network : nn.Module
        Module mapping node embeddings to per-node energies.
    max_num_neighbors : int
        Maximum allowed neighbors per node.
    nls_distance_method:
        Method for computing a neighbor list. Supported values are
        `torch`, `nvalchemi_naive`, `nvalchemi_cell` and custom.
    """

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
        nls_distance_method: str = "torch",
    ):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.virt_embedding_layer = virt_embedding_layer
        self.basis = basis_instance
        self.virt_basis = virt_basis_instance
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_virt_nodes = num_virt_nodes
        self.regularization = regularization_instance
        self.nls_distance_method = nls_distance_method

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
        """
        Initialize model parameters.
        """
        self.embedding_layer.reset_parameters()
        self.virt_embedding_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def MakeRealSystem(self, data):
        """
        Build the System object for the 'real' (atom) graph.

        Expected to construct `system.embedding`, `system.edge_*` tensors,
        indices and other bookkeeping for real atoms from AtomicData.

        Parameters
        ----------
        data : AtomicData
            Input atomic data containing at least positions, atomic numbers and batch.

        Returns
        -------
        System
            A System instance populated with real/atom graph structures.
        """
        raise NotImplementedError

    def MakeVirtSystem(self, data, system):
        """
        Extend the provided System with virtual-node connectivity and features.

        Populates:
          - system.virt_embedding : per-virtual-node embeddings
          - system.aggregation_edge_indices / aggregation_edge_attrs : edges from atoms -> virtuals
          - system.broadcast_edge_indices / broadcast_edge_attrs : edges from virtuals -> atoms
          - system.regularization_weights : per-edge regularization scalars

        Parameters
        ----------
        data : AtomicData
            Input atomic data used to compute per-graph virtual connections.
        system : System
            System created by MakeRealSystem to be augmented.

        Returns
        -------
        System
            The mutated system (same instance) with virtual nodes added.
        """
        device = data.pos.device

        virt_embedding = []
        for level_index in range(self.num_virt_nodes):
            virt_embedding.append(
                self.virt_embedding_layer(
                    torch.tensor([level_index], device=device)
                ).repeat(data.batch.max() + 1, 1)
            )
        system.virt_embedding = torch.cat(virt_embedding)

        range_edge_indices = torch.stack(
            (torch.arange(data.pos.shape[0], device=device), data.batch)
        )

        if "pbc" in data and data.pbc.any():
            weights = calc_weights_pbc(data)
        else:
            weights = calc_weights(data)

        range_edge_attrs = self.virt_basis(weights)

        index_0 = range_edge_indices[0].repeat(self.num_virt_nodes)
        index_1 = (
            range_edge_indices[1]
            .repeat(self.num_virt_nodes)
            .view(self.num_virt_nodes, -1)
            + (
                torch.arange(self.num_virt_nodes, device=device)
                * (range_edge_indices[1].max() + 1)
            ).unsqueeze(1)
        ).flatten()

        system.aggregation_edge_indices = torch.stack((index_0, index_1))
        system.aggregation_edge_attrs = range_edge_attrs.repeat(self.num_virt_nodes, 1)

        index_0 = torch.cat(
            (
                system.aggregation_edge_indices[1],
                torch.arange(
                    system.aggregation_edge_indices[1].max() + 1,
                    system.aggregation_edge_indices[1].max()
                    + 1
                    + system.embedding[0].shape[0],
                    device=device,
                ),
            )
        )

        index_1 = torch.arange(data.pos.shape[0], device=device).repeat(
            self.num_virt_nodes + 1
        )

        system.broadcast_edge_indices = torch.stack((index_0, index_1))

        system.broadcast_edge_attrs = torch.cat(
            (
                system.aggregation_edge_attrs,
                torch.zeros(
                    (system.embedding[0].shape[0], range_edge_attrs.shape[1]),
                    device=device,
                ),
            )
        )

        system.regularization_weights = self.regularization(data.n_atoms)

        return system

    def forward(self, data: AtomicData) -> AtomicData:
        """
        Run a forward pass of RANGE on AtomicData.

        Performs:
          - real system construction (MakeRealSystem)
          - virtual node addition (MakeVirtSystem)
          - sequential application of interaction blocks
          - reduction of node energies to per-graph energies and logging to data.out

        Parameters
        ----------
        data : AtomicData
            Input atomic data. Must contain pos, batch and other keys expected by embedding/basis modules.

        Returns
        -------
        AtomicData
            Same AtomicData instance with `data.out[self.name]` set to a dict containing:
              - ENERGY_KEY : per-graph energies (Tensor)
              - 'regL1' : regularization statistics
        """
        system = self.MakeRealSystem(data)
        system = self.MakeVirtSystem(data, system)

        for block_update in self.interaction_blocks:
            block_update(system)

        energies = self.output_network(self.layer_norm(system.embedding[0]))

        energy = scatter(energies, system.batch, dim=0, reduce="add").flatten()

        data.out[self.name] = {
            ENERGY_KEY: energy,
            "regL1": self.regularization.regularized_params(),
        }

        return data

    def is_nl_compatible(self, nl):
        """
        Check if a given neighbor-list dictionary is compatible with this model.

        A compatible neighbor list for RANGE must:
          - be validated by validate_neighborlist()
          - have order == 2
          - not include self-interactions
          - use the same rcut as this model's cutoff

        Parameters
        ----------
        nl : dict
            Neighbor-list dictionary.

        Returns
        -------
        bool
            True if the neighbor list suits the model, False otherwise.
        """
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.cutoff
            ):
                is_compatible = True
        return is_compatible

    def neighbor_list(
        self,
        name: str,
        data: AtomicData,
        rcut: float,
        max_num_neighbors: int = 1000,
    ) -> dict:
        """
        Helper to construct a neighbor-list mapping.

        Parameters
        ----------
        name : str
            Key name for the returned mapping.
        data : AtomicData
            Input atomic data to build the neighbor list from.
        rcut : float
            Cutoff radius for neighbor search.
        max_num_neighbors : int
            Maximum number of neighbors per atom.

        Returns
        -------
        dict
            A dictionary { name: neighbor_list_dict } where neighbor_list_dict is
            the structure produced by atomic_data2neighbor_list(...).
        """
        if not hasattr(self, "nls_distance_method"):
            self.nls_distance_method = "torch"
        return {
            name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
                nls_distance_method=self.nls_distance_method,
            )
        }
