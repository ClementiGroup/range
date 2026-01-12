import torch
import pytest
from mlcg.data import AtomicData
from rangemp.nn.utils import calc_weights, calc_weights_pbc  # adjust import as needed


def get_dummy_data(one_mol=False):
    pos = torch.tensor(
        [[1.1, 2.2, 0.3], [0.4, 0.2, 0.0], [0.1, 0.8, 0.5], [0.3, 1.7, 0.6]],
        dtype=torch.float32,
    )

    if not one_mol:
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        molecules = 2
    else:
        batch = torch.zeros(4, dtype=torch.long)
        molecules = 1

    cell = torch.eye(3).repeat(molecules, 1, 1)
    pbc = torch.ones((molecules, 3), dtype=torch.bool).view(-1, 3)
    atom_types = torch.ones(4)

    data = AtomicData(
        pos=pos,
        atom_types=atom_types,
        batch=batch,
        cell=cell,
        pbc=pbc,
    )
    return data


@pytest.mark.parametrize(
    "data, periodic",
    [
        (get_dummy_data(False), True),
        (get_dummy_data(True), True),
        (get_dummy_data(False), False),
        (get_dummy_data(True), False),
    ],
)
def test_calc_weights_basic(data, periodic):
    print(data)

    if periodic:
        weights = calc_weights_pbc(data)
    else:
        weights = calc_weights(data)

    assert weights.shape == data.batch.shape
    assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0)
    for batch in range(data.batch.max() + 1):
        max_per_batch = weights[batch == batch].max()
        torch.testing.assert_close(max_per_batch, torch.tensor(1.0))
