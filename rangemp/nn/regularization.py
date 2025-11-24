import torch
from typing import Optional
from mlcg.nn.losses import _Loss
from mlcg.data import AtomicData


class LinearReg(torch.nn.Module):
    """
    A linear regularization module that computes a scaling factor based on
    the number of atoms in a system, using learnable parameters.

    Attributes:
        param1 (torch.nn.Parameter):
            A learnable parameter affecting the scaling factor.
        param2 (torch.nn.Parameter):
            A learnable parameter controlling the influence of atom count.
        param3 (torch.nn.Parameter):
            A learnable parameter modifying the transformation of atom count.
        first_node (torch.Tensor):
            A buffer representing the first virtual node.
        n_min (torch.Tensor):
            The minimum number of atoms in the system.
        n_max (torch.Tensor):
            The maximum number of atoms in the system.
        self_loop (torch.Tensor):
            A buffer for self-loop regularization.

    Args:
        num_virt_nodes (int): Number of virtual nodes in the system.
        min_num_atoms (int): Minimum number of atoms in the dataset.
        max_num_atoms (int): Maximum number of atoms in the dataset.
    """

    def __init__(self, num_virt_nodes: int, min_num_atoms: int, max_num_atoms: int):
        super().__init__()
        self.param1 = torch.nn.Parameter(
            torch.normal(
                torch.ones(num_virt_nodes - 1) * 0.5,
                torch.ones(num_virt_nodes - 1) * 0.1,
            )
        )
        self.param2 = torch.nn.Parameter(torch.ones(num_virt_nodes - 1) * 4)
        self.param3 = torch.nn.Parameter(torch.ones(num_virt_nodes - 1) * 2)
        self.register_buffer("first_node", torch.ones(1))
        self.register_buffer("n_min", torch.tensor(min_num_atoms))
        self.register_buffer("n_max", torch.tensor(max_num_atoms))
        self.register_buffer("self_loop", torch.tensor(1.0))

    def forward(self, num_atoms: torch.Tensor):
        """
        Computes the regularization scaling factors for a batch of systems.

        Args:
            num_atoms (torch.Tensor): Tensor containing the number of atoms in each system.

        Returns:
            torch.Tensor: A tensor of regularization factors.
        """
        A = 1 + torch.abs(self.param2).unsqueeze(1)
        B = A * torch.tanh(torch.abs(self.param3)).unsqueeze(1)
        N = (num_atoms.unsqueeze(0) - self.n_min) / (self.n_max - self.n_min)
        alpha = torch.abs(A * (1 - N) + B * N)

        reg = torch.cat(
            (
                self.first_node.repeat(num_atoms.shape[0]).unsqueeze(0),
                torch.pow(torch.abs(self.param1.unsqueeze(1)), alpha),
            )
        )
        reg = reg.repeat_interleave(num_atoms, dim=1).flatten()

        reg = torch.cat([reg, self.self_loop.repeat(num_atoms.sum())])

        return reg

    def regularized_params(self):
        """
        Returns the parameters subject to regularization.

        Returns:
            torch.nn.Parameter: Regularized parameters.
        """
        return self.param1


class IdentityReg(torch.nn.Module):
    """
    A regularization module that applies an identity transformation,
    ensuring uniform regularization across nodes.

    Attributes:
        num_virt_nodes (torch.Tensor):
            Number of virtual nodes in the system.
        self_loop (torch.Tensor):
            A buffer for self-loop regularization.

    Args:
        num_virt_nodes (int): Number of virtual nodes in the system.
        min_num_atoms (int): Minimum number of atoms in the dataset.
        max_num_atoms (int): Maximum number of atoms in the dataset.
    """

    def __init__(self, num_virt_nodes: int, min_num_atoms: int, max_num_atoms: int):
        super().__init__()
        self.register_buffer("num_virt_nodes", torch.tensor(num_virt_nodes))
        self.register_buffer("self_loop", torch.tensor(1.0))

    def forward(self, num_atoms: torch.Tensor):
        """
        Computes a uniform regularization factor.

        Args:
            num_atoms (torch.Tensor): Tensor containing the number of atoms in each system.

        Returns:
            torch.Tensor: A tensor of uniform regularization factors.
        """
        reg = (
            torch.ones(num_atoms.shape[0], self.num_virt_nodes, device=num_atoms.device)
            .repeat_interleave(num_atoms, dim=0)
            .flatten()
        )
        reg = torch.cat([reg, self.self_loop.repeat(num_atoms.sum())])

        return reg

    def regularized_params(self):
        """
        Returns zero tensors, indicating no learnable parameters for regularization.

        Returns:
            torch.Tensor: A tensor of zeros.
        """
        return torch.zeros_like(self.self_loop)


class RegL1(_Loss):
    """
    Computes the L1 regularization loss for a given property.

    Attributes:
        reg_kwd (str):
            The key for retrieving the regularization property from `data.out`.

    Args:
        reg_kwd (str, optional): The target regularization keyword. Defaults to "regL1".
        size_average (Optional[bool], optional): Whether to average the loss. Defaults to None.
        reduce (Optional[bool], optional): Whether to reduce the loss. Defaults to None.
        reduction (str, optional): The reduction method to apply to the loss. Defaults to "mean".
    """

    def __init__(
        self,
        reg_kwd: str = "regL1",
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

        self.reg_kwd = reg_kwd

    def forward(self, data: AtomicData) -> torch.Tensor:
        """
        Computes the L1 loss for the specified regularization property.

        Args:
            data (AtomicData): The input data containing computed regularization values.

        Returns:
            torch.Tensor: The computed L1 loss.

        Raises:
            RuntimeError: If the target property is not found in `data.out`.
        """
        if self.reg_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.reg_kwd} has not been computed in data.out {list(data.out.keys())}"
            )

        return torch.nn.functional.l1_loss(
            data.out[self.reg_kwd],
            torch.zeros_like(data.out[self.reg_kwd]),
            reduction=self.reduction,
        )
