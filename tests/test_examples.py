import pytest
from runpy import run_module
import sys
from pathlib import Path
import os


this_path = Path(__file__).parent.resolve()
parent_path = Path(__file__).parent.resolve().parent
try:
    python_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    python_paths = [str(parent_path)]
for path in python_paths:
    if os.path.isdir(path+'/fitsnap3'):
        if path != str(parent_path):
            raise RuntimeError("PythonPath Path {} comes before Path being tested on {}".format(path, str(parent_path)))
        else:
            break

example_path = parent_path / 'examples'

try:
    # stubs = 0 MPI is active
    stubs = 0
    from mpi4py import MPI
except ModuleNotFoundError:
    stubs = 1

if stubs == 0:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
if stubs == 1:
    rank = 0
    size = 1
    comm = None
    sub_rank = 0
    sub_size = 1
    sub_comm = None
    sub_head_proc = 0
    node_index = 0
    number_of_nodes = 1


@pytest.fixture
def ta_linear_input():
    ta_linear_example_file = example_path / 'Ta_Linear_JCP2014' / 'Ta-example.in'
    return str(ta_linear_example_file)


def test_pathing(ta_linear_input):
    original_arguments = sys.argv
    new_arguments = ['fitsnap3', ta_linear_input, '--overwrite']
    sys.argv = new_arguments
    run_module(sys.argv[0], run_name='__main__', alter_sys=True)
    sys.argv = original_arguments
