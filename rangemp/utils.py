import importlib


def create_instance(class_path: str, *args, **kwargs):
    """
    Dynamically creates an instance of a class given its full path.

    Args:
        class_path (str): The full path to the class, e.g., 'module.submodule.ClassName'.
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)
