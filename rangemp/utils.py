import importlib
from typing import Tuple

import torch


def create_instance(class_path: str, *args, **kwargs):
    """
    Creates class instance from string of the class path.

    Parameters
    ----------
    class_path : str
        Absolute path of the class to instatiate.

    Returns
    -------
    cls
        Class instantiation.
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)


# FIXME: this only works when used to create batches because they are already sorted
# there is a bug in pytorch that always gives a sorted array with unique
def uniquen(*tensor_list: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    zipped_list = torch.stack(tensor_list, dim=0).T
    indices, indices_inverse = torch.unique(zipped_list,
                                            return_inverse=True,
                                            dim=0)
    return (indices, indices_inverse)
