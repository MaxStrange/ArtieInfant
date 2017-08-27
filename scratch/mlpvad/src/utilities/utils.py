"""
Random utilities for this project.
"""
import inspect

def log(*args, **kwargs):
    """
    Simply does this:

    print("  " * depth, "|->", *args)
    """
    return None
    depth = len(inspect.stack()) + kwargs.pop("increase_depth", 0)
    print("  " * depth, "|->", *args, **kwargs)

