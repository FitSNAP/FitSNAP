import pytest
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
    if os.path.isdir(path+'/fitsnap3lib'):
        if path != str(parent_path) and path != str(parent_path) + '/':
            raise RuntimeError("PythonPath Path {} comes before Path being tested on {}".format(path, str(parent_path)))
        else:
            break
assert importlib.util.find_spec("fitsnap3lib") is not None
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
pace_example_file = example_path / 'Ta_PACE' / 'Ta.in'


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "env(stubs): mark test to run only on named environment"
    )


def mpi_assert(mpi):
    assert mpi.stubs == 0
    assert mpi.size >= 1
    mpi.set_added_comm()


@pytest.mark.parametrize(
    "stubs_input",
    [ta_linear_example_file]
)
def test_stubs(stubs_input):
    """Test stubs is working on Ta Linear Example"""
    ec = example_checker.ExampleChecker(stubs_input)
    ec.assert_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@pytest.mark.parametrize(
    "basic_input",
    [ta_linear_example_file]
)
def test_fitsnap_basic(basic_input):
    #Test FitSNAP 1 proc non stubs is working on Ta Linear Example
    ec = example_checker.ExampleChecker(basic_input)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()

@example_checker.mpi_run(3)
def test_fitsnap_mpi():
    #Test FitSNAP 3 proc non stubs is working on Ta Linear Example
    ec = example_checker.ExampleChecker(ta_linear_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_quad():
    #Test FitSNAP 2 proc is working on Ta Quadratic Example
    ec = example_checker.ExampleChecker(ta_quadratic_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_eme():
    #Test FitSNAP 2 proc is working on InP EME Example
    ec = example_checker.ExampleChecker(inp_eme_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_xyz():
    #Test FitSNAP 2 proc XYZ scraper on Ta Linear Example
    ec = example_checker.ExampleChecker(ta_xyz_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_neme():
    #Test FitSNAP multi element non-explicit WBe Linear Example
    ec = example_checker.ExampleChecker(wbe_linear_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_spin():
    #Test FitSNAP Fe Linear Spin Example
    ec = example_checker.ExampleChecker(fe_spin_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.snapcoeff_diff()


@example_checker.mpi_run(2)
def test_fitsnap_pace():
    #Test FitSNAP Fe Linear Spin Example
    ec = example_checker.ExampleChecker(pace_example_file)
    ec.assert_not_stubs()
    ec.run_fitsnap()
    ec.pacecoeff_diff()