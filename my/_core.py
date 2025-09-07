import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from time import time, sleep
from typing import Callable


def time_it(func: Callable) -> Callable: # Decorator function for timing.
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1): .4f}s')
        return result
    return wrapper


def comp(f: Callable, g: Callable) -> Callable: # Return composite function f(g())
    def wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))
        
    return wrapper
