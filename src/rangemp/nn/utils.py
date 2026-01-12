import torch

from mlcg.data import AtomicData
from typing import Optional
from .scatter import scatter


def calc_weights(data: AtomicData) -> torch.Tensor:
    """
    Compute per-atom radial weights relative to each molecule's geometric center.

    The weight for each atom is the Euclidean distance from the atom to the
    geometric center (mean position) of its molecule, normalized by the maximum
    distance within that molecule. Normalization ensures weights are in [0, 1].
    If an atom lays on the geometric center, it's weights is 0.

    Parameters
    ----------
    data : AtomicData
        AtomicData containing `pos` (tensor of shape [N, 3]) and `batch`
        (tensor of length N) mapping atoms to molecule indices.

    Returns
    -------
    torch.Tensor
        Tensor of shape [N] with a normalized radial weight for each atom.
    """
    virt_pos = scatter(data.pos, data.batch, reduce="mean", dim=0)

    weights = (data.pos - virt_pos[data.batch]).norm(p=2, dim=1)
    norm = scatter(weights, data.batch, reduce="max")[data.batch]
    norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    weights = weights / norm

    return weights


def calc_weights_pbc(data: AtomicData) -> torch.Tensor:
    """
    Compute per-atom radial weights with periodic boundary conditions (PBC).

    This function accounts for periodic images when computing distances to the
    geometric center. Positions are transformed to fractional coordinates using
    the inverse cell, the mean (geometric center) is computed in fractional
    space, and distances are corrected using minimal image convention. Final
    distances are converted back to Cartesian space and normalized by the
    per-molecule maximum distance to yield values in [0, 1].

    Requirements on `data`
    ---------------------
    - data.cell indicating cell vectors.
    - data.pbc indicating periodicity along each axis.
    - data.pos and data.batch as in calc_weights.

    Parameters
    ----------
    data : AtomicData
        AtomicData providing `pos`, `batch`, `cell` and `pbc`.

    Returns
    -------
    torch.Tensor
        Tensor of shape [N] with normalized radial weights for each atom that
        respect periodic boundary conditions.
    """
    cell = data.cell.view((-1, 3, 3)).transpose(2, 1)
    pbc = data.pbc.view((-1, 3, 1))

    reciprocal_cell = torch.linalg.inv(cell)
    scaled_pos = reciprocal_cell[data.batch] @ data.pos.unsqueeze(-1)
    scaled_virt_pos = scatter(scaled_pos, data.batch, reduce="mean", dim=0)
    diff = scaled_pos - scaled_virt_pos[data.batch]
    diff = diff - pbc[data.batch] * torch.round(diff)

    weights = (cell[data.batch] @ diff).norm(p=2, dim=1).squeeze()
    norm = scatter(weights, data.batch, reduce="max")[data.batch]
    norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    weights = weights / norm

    return weights
