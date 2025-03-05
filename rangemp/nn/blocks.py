import torch
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.inits import glorot

from mlcg.nn import MLP
from mlcg.nn._module_init import init_xavier_uniform
from mlcg.nn.painn import PaiNNInteraction, PaiNNMixing

from .system import System


class AttentionBlock(torch.nn.Module):
    """
    Implements a multi-head attention mechanism for graph-based models.
    """

    def __init__(self,
                 channels: int,
                 n_heads: int,
                 basis_dim: int,
                 **kwargs):
        super().__init__()

        if channels % n_heads != 0:
            raise ValueError(
                    "The number of hidden channels must be divisible by the number of heads"
                    )

        self.n_heads = n_heads
        self.channels = channels
        self.hidden_channels = channels // self.n_heads

        self.lin_Q = torch.nn.Linear(channels, channels, bias=False)
        self.lin_K = torch.nn.Linear(channels, channels, bias=False)
        self.lin_V = torch.nn.Linear(channels, channels, bias=False)

        self.activation = torch.nn.LeakyReLU()

        self.attention = torch.nn.Parameter(torch.empty(1, n_heads, self.hidden_channels))

        self.basis_dim = basis_dim
        self.lin_E = torch.nn.Linear(basis_dim, channels, bias=False)

    def forward(self,
                senders: torch.Tensor,
                receivers: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_attrs: torch.Tensor,
                *args) -> torch.Tensor:
        """
        Forward pass for the attention block.

        Args:
            senders (torch.Tensor): Feature matrix of sender nodes.
            receivers (torch.Tensor): Feature matrix of receiver nodes.
            edge_indices (torch.Tensor): Edge index tensor.
            edge_attrs (torch.Tensor): Edge feature tensor.

        Returns:
            torch.Tensor: Updated node embeddings.
        """
        E = self.lin_E(edge_attrs)

        Q = self.lin_Q(receivers)[edge_indices[1]]
        K = self.lin_K(senders)[edge_indices[0]]
        V = self.lin_V(senders)[edge_indices[0]]

        weights = torch.sum(
                self.attention*
                self.activation(
                    (Q + K + E).view(-1, self.n_heads, self.hidden_channels)
                    ),
                dim=2)
        weights = scatter_softmax(weights,
                                  edge_indices[1],
                                  dim=0)
        weights = weights.unsqueeze(-1)

        V = V.view(-1, self.n_heads, self.hidden_channels)

        embedding = scatter(
                (weights*V).view(-1, self.channels),
                edge_indices[1],
                reduce='add',
                dim=0
                )

        return embedding

    def reset_parameters(self):
        """
        Reinitializes the model parameters.
        """
        init_xavier_uniform(self.lin_E)
        init_xavier_uniform(self.lin_Q)
        init_xavier_uniform(self.lin_K)
        init_xavier_uniform(self.lin_V)
        glorot(self.attention)


class SelfAttentionBlock(AttentionBlock):
    """
    Implements a self-attention mechanism, extending AttentionBlock with additional transformations.
    """

    def __init__(self,
                 channels: int,
                 n_heads: int,
                 basis_dim: int,
                 **kwargs):

        super().__init__(channels, n_heads, basis_dim, **kwargs)

        self.weights_K = torch.nn.Parameter(torch.empty(n_heads,
                                                        channels//n_heads,
                                                        channels//n_heads))
        self.weights_V = torch.nn.Parameter(torch.empty(n_heads,
                                                        channels//n_heads,
                                                        channels//n_heads))

    def forward(self,
                senders: torch.Tensor,
                senders_self: torch.Tensor,
                receivers: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_attrs: torch.Tensor,
                regularization_weights: torch.Tensor,
                *args) -> torch.Tensor:
        """
        Forward pass for the attention block.

        Args:
            senders (torch.Tensor): Feature matrix of sender nodes.
            senders_self (torch.Tensor): Feature matrix of sender nodes in self-loops.
            receivers (torch.Tensor): Feature matrix of receiver nodes.
            edge_indices (torch.Tensor): Edge index tensor.
            edge_attrs (torch.Tensor): Edge feature tensor.
            regularization_weights (torch.Tensor): .

        Returns:
            torch.Tensor: Updated node embeddings.
        """
        K = torch.vmap(torch.matmul, in_dims=(1, 0), out_dims=1)(senders.view(-1,
                                                                              self.n_heads,
                                                                              self.hidden_channels),
                                                                 self.weights_K).reshape(-1, self.channels)

        K_self = self.lin_K(senders_self)
        K = torch.cat([K, K_self], dim=0)[edge_indices[0]]

        V = torch.vmap(torch.matmul, in_dims=(1, 0), out_dims=1)(senders.view(-1,
                                                                              self.n_heads,
                                                                              self.hidden_channels),
                                                                 self.weights_V).reshape(-1, self.channels)
        V_self = self.lin_V(senders_self)
        V = torch.cat([V, V_self], dim=0)[edge_indices[0]]

        E = self.lin_E(edge_attrs)
        Q = self.lin_Q(receivers)[edge_indices[1]]

        weights = torch.sum(
                self.attention*
                self.activation(
                    (Q + K + E).view(-1, self.n_heads, self.hidden_channels)
                    ),
                dim=2)
        weights = scatter_softmax(weights,
                                  edge_indices[1],
                                  dim=0)

        weights = weights.unsqueeze(-1)

        V = V.view(-1, self.n_heads, self.hidden_channels)

        embedding = scatter(
                regularization_weights.unsqueeze(1)*(weights*V).view(-1, self.channels),
                edge_indices[1],
                reduce='add',
                dim=0
                )

        return embedding

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weights_K)
        glorot(self.weights_V)


class SchnetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 basis_dim: int,
                 edge_dim: int,
                 activation: torch.nn.Module,
                 cutoff: torch.nn.Module,
                 **kwargs):
        super().__init__()

        self.lin1 = torch.nn.Linear(in_channels, edge_dim, bias=False)

        self.cutoff = cutoff

        self.filter_network = MLP(
                layer_widths=[basis_dim,
                              edge_dim,
                              edge_dim],
                activation_func=activation,
                last_bias=False
                )

        self.lin2 = torch.nn.Linear(edge_dim, out_channels)
        self.layer_norm = torch.nn.LayerNorm(out_channels)
        self.activation = activation
        self.lin3 = torch.nn.Linear(out_channels, out_channels)

    def forward(self,
                senders: torch.Tensor,
                receivers: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor,
                edge_versors: torch.Tensor,
                edge_attrs: torch.Tensor,
                *args) -> torch.Tensor:

        C = self.cutoff(edge_weights)
        weights = self.filter_network(edge_attrs)*C.view(-1, 1)

        V = self.lin1(senders[0])[edge_indices[0]]

        embedding_update = scatter(
                weights*V,
                edge_indices[1],
                reduce='add',
                dim=0
                )

        embedding_update = self.lin3(
                self.activation(
                    self.layer_norm(
                        self.lin2(
                            embedding_update
                            )
                        )
                    )
                )
        return [receivers[0] + embedding_update]

    def reset_parameters(self):
        self.filter_network.reset_parameters()
        init_xavier_uniform(self.lin1)
        init_xavier_uniform(self.lin2)
        init_xavier_uniform(self.lin3)


class PaiNNBlock(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 basis_dim: int,
                 activation: torch.nn.Module,
                 cutoff: torch.nn.Module,
                 aggr: str,
                 epsilon: float,
                 **kwargs):
        super().__init__()

        self.interaction = PaiNNInteraction(channels,
                                            basis_dim,
                                            cutoff,
                                            activation,
                                            aggr)
        self.mixing = PaiNNMixing(channels,
                                  activation,
                                  epsilon)

        self.reset_parameters()

    def forward(self,
                senders: torch.Tensor,
                receivers: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor,
                edge_versors: torch.Tensor,
                edge_attrs: torch.Tensor,
                *args) -> torch.Tensor:

        q = senders[0].unsqueeze(1)  # (n_atoms, 1, n_features)
        mu = senders[1]
        q, mu = self.interaction(
            q, mu, edge_versors, edge_indices, edge_weights.unsqueeze(1), edge_attrs.unsqueeze(1)
        )
        q, mu = self.mixing(q, mu)
        q = q.squeeze(1)

        return [q, mu]

    def reset_parameters(self):
        self.interaction.reset_parameters()
        self.mixing.reset_parameters()


class RANGEInteractionBlock(torch.nn.Module):
    def __init__(self,
                 propagation_block: torch.nn.Module,
                 aggregation_block: torch.nn.Module,
                 broadcast_block: torch.nn.Module,
                 activation_block: torch.nn.Module,
                 output_block: torch.nn.Module):
        super().__init__()
        self.propagation_block = propagation_block
        self.aggregation_block = aggregation_block
        self.broadcast_block = broadcast_block
        self.activation_block = activation_block
        self.output_block = output_block

    def forward(self, system: System) -> None:
        system.embedding = self.propagation_block(
                system.embedding,
                system.embedding,
                system.edge_indices,
                system.edge_weights,
                system.edge_versors,
                system.edge_attrs
                )

        system.virt_embedding = self.aggregation_block(
                system.embedding[0],
                system.virt_embedding,
                system.aggregation_edge_indices,
                system.aggregation_edge_attrs
                )

        system.virt_embedding = self.activation_block(
                system.virt_embedding
                )

        system.embedding[0] = self.broadcast_block(
                system.virt_embedding,
                system.embedding[0],
                system.embedding[0],
                system.broadcast_edge_indices,
                system.broadcast_edge_attrs,
                system.regularization_weights
                )

        system.embedding[0] = self.output_block(system.embedding[0])

        return None

    def reset_parameters(self):
        self.propagation_block.reset_parameters()
        self.aggregation_block.reset_parameters()
        self.broadcast_block.reset_parameters()
        for block in self.output_block:
            init_xavier_uniform(block)
