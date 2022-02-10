import pytest
from runpy import run_module
import sys
from pathlib import Path
import os
import importlib.util


"""Resolve Pathing and load ExampleChecker"""
this_path = Path(__file__).parent.resolve()
parent_path = Path(__file__).parent.resolve().parent
try:
    python_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    python_paths = [str(parent_path)]
for path in python_paths:
    if os.path.isdir(path+'/fitsnap3'):
        if path != str(parent_path) and path != str(parent_path) + '/':
            raise RuntimeError("PythonPath Path {} comes before Path being tested on {}".format(path, str(parent_path)))
        else:
            break
assert importlib.util.find_spec("fitsnap3") is not None
spec = importlib.util.spec_from_file_location("module.name", str(parent_path / "tests/example_checker.py"))
example_checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_checker)

example_path = parent_path / 'examples'

"""Examples"""
ta_linear_example_file = example_path / 'Ta_Linear_JCP2014' / 'Ta-example.in'
ta_quadratic_example_file = example_path / 'Ta_Quadratic_JCP2018' / 'Ta-example.in'
ta_xyz_example_file = example_path / 'Ta_XYZ' / 'Ta-example.in'
wbe_linear_example_file = example_path / 'WBe_PRB2019' / 'WBe-example.in'
inp_eme_example_file = example_path / 'InP_JPCA2020' / 'InP-example.in'
fe_spin_example_file = example_path / 'Fe_Linear_NPJ2021' / 'Fe-example.in'


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "env(stubs): mark test to run only on named environment"
    )


def snap_test(input_file, mpi_comm=None):
    """Generic formula for testing snap examples"""
    assert importlib.util.find_spec("fitsnap3") is not None
    original_arguments = sys.argv
    new_arguments = ['fitsnap3', str(input_file), '--overwrite', '-r']
    sys.argv = new_arguments
    run_module(sys.argv[0], run_name='__main__', alter_sys=True)
    sys.argv = original_arguments
    ec = example_checker.ExampleChecker(input_file)
    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        ec.snapcoeff_diff()


@pytest.mark.parametrize(
    "stubs_input",
    [ta_linear_example_file]
)
def test_stubs(stubs_input):
    """Test stubs is working on Ta Linear Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 1
    snap_test(stubs_input)


@pytest.mark.parametrize(
    "basic_input",
    [ta_linear_example_file]
)
def test_fitsnap_basic(basic_input):
    """Test FitSNAP 1 proc non stubs is working on Ta Linear Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    snap_test(basic_input)


@example_checker.mpi_run(3)
def test_fitsnap_mpi():
    """Test FitSNAP 3 proc non stubs is working on Ta Linear Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(ta_linear_example_file, mpi.comm)


@example_checker.mpi_run(2)
def test_fitsnap_quad():
    """Test FitSNAP 4 proc is working on Ta Quadratic Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(ta_quadratic_example_file, mpi.comm)


@example_checker.mpi_run(2)
def test_fitsnap_eme():
    """Test FitSNAP 8 proc is working on InP EME Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(inp_eme_example_file, mpi.comm)


@example_checker.mpi_run(2)
def test_fitsnap_xyz():
    """Test FitSNAP 4 proc XYZ scraper on Ta Linear Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(ta_xyz_example_file, mpi.comm)


@example_checker.mpi_run(2)
def test_fitsnap_neme():
    """Test FitSNAP multi element non-explicit WBe Linear Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(wbe_linear_example_file, mpi.comm)


@example_checker.mpi_run(2)
def test_fitsnap_spin():
    """Test FitSNAP Fe Linear Spin Example"""
    mpi = example_checker.MPICheck()
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()
    snap_test(wbe_linear_example_file, mpi.comm)
