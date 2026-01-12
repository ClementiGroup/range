import pytest
import torch

from rangemp.nn.scatter import scatter, scatter_softmax
from torch_geometric.utils import scatter as pyg_scatter
from torch_scatter import scatter as ts_scatter
from torch_scatter import scatter_softmax as ts_scatter_softmax


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "x_size, batch_size, dim, reduce_op",
    [
        ([128, 32], 1, 1, "sum"),
        ([128, 32], 1, 1, "mean"),
        ([128, 32], 1, 1, "min"),
        ([128, 32], 1, 1, "max"),
        ([128, 32], 1, 1, "mul"),
        ([128, 32], 8, 1, "sum"),
        ([128, 32], 8, 1, "mean"),
        ([128, 32], 8, 1, "min"),
        ([128, 32], 8, 1, "max"),
        ([128, 32], 8, 1, "mul"),
        ([32], 1, 0, "sum"),
        ([32], 1, 0, "mean"),
        ([32], 1, 0, "min"),
        ([32], 1, 0, "max"),
        ([32], 1, 0, "mul"),
        ([32], 8, 0, "sum"),
        ([32], 8, 0, "mean"),
        ([32], 8, 0, "min"),
        ([32], 8, 0, "max"),
        ([32], 8, 0, "mul"),
    ],
)
def test_scatter_out(
    x_size: tuple, batch_size: int, dim: int, reduce_op: str, compiled: bool
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(x_size, device=device, dtype=torch.float32)
    batch = torch.arange(
        batch_size, dtype=torch.int64, device=device
    ).repeat_interleave(x_size[dim] // batch_size)

    s_ts = ts_scatter(x, batch, dim=dim, reduce=reduce_op)
    s_pyg = pyg_scatter(x, batch, dim=dim, reduce=reduce_op)

    if compiled and device == "cpu":
        pytest.skip("Skip compiled test on CPU")
    if compiled:
        fn = torch.compile(scatter, dynamic=True)
    else:
        fn = scatter
    s_custom = fn(x, batch, dim=dim, reduce=reduce_op)

    torch.testing.assert_close(
        s_pyg, s_ts, atol=1e-5, rtol=1e-5, msg="TS and PYG results differ"
    )
    torch.testing.assert_close(
        s_pyg, s_custom, atol=1e-5, rtol=1e-5, msg="PYG and Custom results differ"
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "x_size, batch_size, dim, reduce_op",
    [
        ([128, 32], 1, 1, "sum"),
        ([128, 32], 1, 1, "mean"),
        ([128, 32], 1, 1, "min"),
        ([128, 32], 1, 1, "max"),
        ([128, 32], 1, 1, "mul"),
        ([128, 32], 8, 1, "sum"),
        ([128, 32], 8, 1, "mean"),
        ([128, 32], 8, 1, "min"),
        ([128, 32], 8, 1, "max"),
        ([128, 32], 8, 1, "mul"),
        ([32], 1, 0, "sum"),
        ([32], 1, 0, "mean"),
        ([32], 1, 0, "min"),
        ([32], 1, 0, "max"),
        ([32], 1, 0, "mul"),
        ([32], 8, 0, "sum"),
        ([32], 8, 0, "mean"),
        ([32], 8, 0, "min"),
        ([32], 8, 0, "max"),
        ([32], 8, 0, "mul"),
    ],
)
def test_gradient_scatter_out(
    x_size: tuple, batch_size: int, dim: int, reduce_op: str, compiled: bool
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(x_size, device=device, dtype=torch.float32)
    batch = torch.arange(
        batch_size, dtype=torch.int64, device=device
    ).repeat_interleave(x_size[dim] // batch_size)
    compiled_scatter = torch.compile(scatter, dynamic=True)
    x = x.requires_grad_(True)
    gs_ts = torch.autograd.grad(
        ts_scatter(x, batch, dim=dim, reduce=reduce_op).sum(), x
    )[0]
    x = x.detach().requires_grad_(True)
    gs_pyg = torch.autograd.grad(
        pyg_scatter(x, batch, dim=dim, reduce=reduce_op).sum(), x
    )[0]
    x = x.detach().requires_grad_(True)

    if compiled and device == "cpu":
        pytest.skip("Skip compiled test on CPU")
    if compiled:
        gs_custom = torch.autograd.grad(
            compiled_scatter(x, batch, dim=dim, reduce=reduce_op).sum(), x
        )[0]
    else:
        gs_custom = torch.autograd.grad(
            scatter(x, batch, dim=dim, reduce=reduce_op).sum(), x
        )[0]
    x = x.detach()

    torch.testing.assert_close(
        gs_pyg, gs_ts, atol=1e-5, rtol=1e-5, msg="TS and PYG results differ"
    )
    torch.testing.assert_close(
        gs_pyg, gs_custom, atol=1e-5, rtol=1e-5, msg="PYG and Custom results differ"
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "x_size, batch_size, dim",
    [
        ([128, 32], 1, 1),
        ([128, 32], 8, 1),
        ([32], 1, 0),
        ([32], 8, 0),
    ],
)
def test_scatter_softmax_out(x_size: tuple, batch_size: int, dim: int, compiled: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(x_size, device=device, dtype=torch.float32)
    batch = torch.arange(
        batch_size, dtype=torch.int64, device=device
    ).repeat_interleave(x_size[dim] // batch_size)

    s_ts = ts_scatter_softmax(x, batch, dim=dim)
    if compiled and device == "cpu":
        pytest.skip("Skip compiled test on CPU")
    if compiled:
        fn = torch.compile(scatter_softmax, dynamic=True)
    else:
        fn = scatter_softmax
    s_custom = fn(x, batch, dim=dim)

    torch.testing.assert_close(
        s_ts, s_custom, atol=1e-5, rtol=1e-5, msg="PYG and Custom results differ"
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "x_size, batch_size, dim",
    [
        ([128, 32], 1, 1),
        ([128, 32], 8, 1),
        ([32], 1, 0),
        ([32], 8, 0),
    ],
)
def test_gradient_scatter_softmax_out(
    x_size: tuple, batch_size: int, dim: int, compiled: bool
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(x_size, device=device, dtype=torch.float32)
    batch = torch.arange(
        batch_size, dtype=torch.int64, device=device
    ).repeat_interleave(x_size[dim] // batch_size)

    x = x.requires_grad_(True)
    gs_ts = torch.autograd.grad(ts_scatter_softmax(x, batch, dim=dim).sum(), x)[0]
    x = x.detach().requires_grad_(True)

    if compiled and device == "cpu":
        pytest.skip("Skip compiled test on CPU")
    if compiled:
        compiled_scatter_softmax = torch.compile(scatter_softmax, dynamic=True)
        gs_custom = torch.autograd.grad(
            compiled_scatter_softmax(x, batch, dim=dim).sum(), x
        )[0]
    else:
        gs_custom = torch.autograd.grad(scatter_softmax(x, batch, dim=dim).sum(), x)[0]

    x = x.detach()

    torch.testing.assert_close(
        gs_ts, gs_custom, atol=1e-5, rtol=1e-5, msg="PYG and Custom results differ"
    )
