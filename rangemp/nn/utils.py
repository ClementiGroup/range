import torch
from torch_scatter import scatter

from mlcg.data import AtomicData


def calc_weights(data: AtomicData) -> torch.Tensor:
    virt_pos = scatter(data.pos,
                       data.batch,
                       reduce='mean',
                       dim=0)

    weights = (data.pos - virt_pos[data.batch]).norm(p=2, dim=1)
    norm = scatter(weights, data.batch, reduce='max')[data.batch]
    norm = norm + (norm == 0.0)
    weights = weights/norm

    return weights


def calc_weights_pbc(data: AtomicData) -> torch.Tensor:
    cell = data.cell.view((-1, 3, 3)).transpose(2, 1)
    pbc = data.pbc.view((-1, 3, 1))

    reciprocal_cell = torch.linalg.inv(cell)
    scaled_pos = reciprocal_cell[data.batch]@data.pos.unsqueeze(-1)
    scaled_virt_pos = scatter(scaled_pos,
                              data.batch,
                              reduce='mean',
                              dim=0)
    diff = scaled_pos - scaled_virt_pos[data.batch]
    diff = diff - pbc[data.batch]*torch.round(diff)

    weights = (cell[data.batch]@diff).norm(p=2, dim=1).squeeze()
    norm = scatter(weights, data.batch, reduce='max')[data.batch]
    norm = norm + (norm == 0.0)
    weights = weights/norm

    return weights
