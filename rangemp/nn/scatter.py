import torch
from typing import Optional


def initialize_tensor_like(
    src: torch.Tensor, shape: int, fill_value: float = 0.0
) -> torch.Tensor:
    """
    Initialize a tensor with provided shape and same dtype and device as src.
    By default initializes with zeros.
    """
    out = torch.full(
        shape,
        fill_value,
        dtype=src.dtype,
        device=src.device,
        requires_grad=src.requires_grad,
    )
    return out


def _get_canonical_key(reduce: str) -> str:
    """
    Maps various reduction operation aliases to canonical keys.
    """
    _mapping_sets = {
        "sum": "sum",
        "add": "sum",
        "mean": "mean",
        "avg": "mean",
        "prod": "prod",
        "mul": "prod",
        "amax": "amax",
        "max": "amax",
        "amin": "amin",
        "min": "amin",
    }
    return _mapping_sets[reduce.lower()]


class SafeScatterMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, dim=0, dim_size=None):
        # Compute max normally in forward pass
        if dim_size is None:
            dim_size = int(index.max()) + 1
        out_shape = list(src.shape)
        out_shape[dim] = dim_size

        out = initialize_tensor_like(src, out_shape, fill_value=float("-inf"))
        out = out.index_reduce(dim, index, src, "amax", include_self=False)
        ctx.save_for_backward(src, index, out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        src, index, out = ctx.saved_tensors
        dim = ctx.dim
        # Identify which src positions contributed to the max
        mask = (src == out.index_select(dim, index)).float()

        # If multiple maxima exist, split gradients evenly
        count = torch.zeros_like(out).index_add(dim, index, mask)
        count = count.index_select(dim, index).clamp(min=1)
        grad_src = mask * (grad_output.index_select(dim, index) / count)

        return grad_src, None, None, None


def scatter_max(src, index, dim=0, dim_size=None):
    return SafeScatterMax.apply(src, index, dim, dim_size)


class SafeScatterMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, dim=0, dim_size=None):
        # Compute max normally in forward pass
        if dim_size is None:
            dim_size = int(index.max()) + 1
        out_shape = list(src.shape)
        out_shape[dim] = dim_size

        out = initialize_tensor_like(src, out_shape, fill_value=float("+inf"))
        out = out.index_reduce(dim, index, src, "amin", include_self=False)
        ctx.save_for_backward(src, index, out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        src, index, out = ctx.saved_tensors
        dim = ctx.dim
        # Identify which src positions contributed to the max
        mask = (src == out.index_select(dim, index)).float()

        # If multiple maxima exist, split gradients evenly
        count = torch.zeros_like(out).index_add(dim, index, mask)
        count = count.index_select(dim, index).clamp(min=1)
        grad_src = mask * (grad_output.index_select(dim, index) / count)

        return grad_src, None, None, None


def scatter_min(src, index, dim=0, dim_size=None):
    return SafeScatterMin.apply(src, index, dim, dim_size)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    reduce: str = "sum",
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Implementation of scatter function compatible with torch.compile.
    Operations are fully based on pytorch native functions.
    Args:
    ----
    src (torch.Tensor):
        Source tensor.
    index (torch.Tensor):
        Index tensor. Specifies the indices on the source tensor scatter
        operation is applied to along given dimension.
    dim (int, optional):
        Dimension along which scatter operation is applied. Defaults is 0.
    reduce (str, optional):
        Reduction operation to apply. Availables are "sum", "mean", "prod",
        "amax", "amin". For compatibility with torch_scatter, "add", "avg", "mul",
        "max", "min" are also accepted as aliases. Defaults is "sum".
    dim_size (Optional[int], optional):
        Size of the output tensor along the scattering dimension.
        If None, it is inferred from the maximum index value. Defaults is None.

    Returns:
    -------
    torch.Tensor:
        Resulting tensor after scatter operation.
    """
    reduce = _get_canonical_key(reduce)
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = initialize_tensor_like(src, out_shape, fill_value=0)
    if reduce == "sum":
        return out.index_add(dim, index, src, alpha=1)
    elif reduce == "amax":
        return scatter_max(src, index, dim, dim_size)
    elif reduce == "amin":
        return scatter_min(src, index, dim, dim_size)
    else:
        return out.index_reduce(
            dim,
            index,
            src,
            reduce,
            include_self=False,
        )


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    for _ in range(dim):
        src = src.unsqueeze(0)
    while src.dim() < other.dim():
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_softmax` requires floating-point input tensors.")

    max_value_per_index = scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")

    expanded_index = broadcast(index, src, dim)

    max_per_src_element = max_value_per_index.gather(dim, expanded_index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter(
        recentered_scores_exp, index, dim=dim, dim_size=dim_size, reduce="sum"
    )
    normalizing_constants = sum_per_index.gather(dim, expanded_index)

    return recentered_scores_exp / normalizing_constants
