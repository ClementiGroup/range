import torch
from typing import Optional
from mlcg.nn.losses import _Loss
from mlcg.data import AtomicData


class LinearReg(torch.nn.Module):
    def __init__(self,
                 num_virt_nodes: int,
                 min_num_atoms: int,
                 max_num_atoms: int):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.normal(torch.ones(num_virt_nodes - 1)*0.5,
                                                      torch.ones(num_virt_nodes - 1)*0.1))
        self.param2 = torch.nn.Parameter(torch.ones(num_virt_nodes - 1)*4)
        self.param3 = torch.nn.Parameter(torch.ones(num_virt_nodes - 1)*2)
        self.register_buffer('first_node', torch.ones(1))
        self.register_buffer('n_min', torch.tensor(min_num_atoms))
        self.register_buffer('n_max', torch.tensor(max_num_atoms))
        self.register_buffer('self_loop', torch.tensor(1.))

    def forward(self,
                num_atoms: torch.Tensor):
        A = 1 + torch.abs(self.param2).unsqueeze(1)
        B = A*torch.tanh(torch.abs(self.param3)).unsqueeze(1)
        N = (num_atoms.unsqueeze(0) - self.n_min)/(self.n_max - self.n_min)
        alpha = torch.abs(A*(1 - N) + B*N)

        reg = torch.cat((self.first_node.repeat(num_atoms.shape[0]).unsqueeze(0), torch.pow(torch.abs(self.param1.unsqueeze(1)), alpha)))
        reg = reg.repeat_interleave(num_atoms, dim=1).flatten()

        reg = torch.cat([reg, self.self_loop.repeat(num_atoms.sum())])

        return reg

    def regularized_params(self):
        return self.param1


class IdentityReg(torch.nn.Module):
    def __init__(self,
                 num_virt_nodes: int,
                 min_num_atoms: int,
                 max_num_atoms: int):
        super().__init__()
        self.register_buffer('num_virt_nodes', torch.tensor(num_virt_nodes))
        self.register_buffer('self_loop', torch.tensor(1.))

    def forward(self,
                num_atoms: torch.Tensor):
        reg = torch.ones(num_atoms.shape[0], self.num_virt_nodes, device=num_atoms.device).repeat_interleave(num_atoms, dim=0).flatten()
        reg = torch.cat([reg, self.self_loop.repeat(num_atoms.sum())])

        return reg

    def regularized_params(self):
        return torch.zeros_like(self.self_loop)
    
class RegL1(_Loss):
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
        if self.reg_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.reg_kwd} has not been computed in data.out {list(data.out.keys())}"
            )

        return torch.nn.functional.l1_loss(
            data.out[self.reg_kwd],
            torch.zeros_like(data.out[self.reg_kwd]),
            reduction=self.reduction,
        )
