from mpi4py import MPI
from functools import wraps

def is_rank_zero():
    return MPI.COMM_WORLD.Get_rank() == 0

def rank_zero(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper
    

