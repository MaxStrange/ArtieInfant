"""
Random utilities for this project.
"""
import inspect
import itertools

def grouper(n, iterable):
    """
    Taken from Stack Overflow:
       [https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python]

    Yields n items from the iterable at a time.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def log(*args, **kwargs):
    """
    Simply does this:

    print("  " * depth, "|->", *args)
    """
#    return None
    depth = len(inspect.stack()) + kwargs.pop("increase_depth", 0)
    print("  " * depth, "|->", *args, **kwargs)

