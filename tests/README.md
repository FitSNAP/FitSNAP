### Requirements

The purpose of these tests is to verify that fitting results are consistent. We also verify that 
neural network forces are properly coded via finite difference tests. To test the parallel 
capabilities of FitSNAP, some of our linear fitting tests use `mpirun`. We therefore require 
`OpenMPI` for these tests.

### Automated testing with pytest

`pytest` find files and functions containing the `test` phrase, and executes them. 

For example, the functions in `test_examples.py` are run via GitHub Actions with
`FitSNAP/.github/workflows/pytests.yaml` when pulling or pushing to the master branch. As seen in 
`pytests.yaml`, the name of this action is `tests`, which is why we see a "tests passing" badge on 
our main repo page if the tests were successful.

### Testing locally

We can manually perform these tests locally. 

**You may have to ensure that your `DYLD_LIBRARY_PATH` and PYTHONPATH are properly set**.

For example:

    python -m pytest -s test_examples.py::test_fitsnap_basic

will run the single proc non stubs test for the Ta linear example. Adding the `-s` python will allow 
you to see screen output from prints, for the purpose of debugging or designing your own tests. 

To perform PyTorch tests locally:

    python -m pytest -s test_pytorch.py --disable-pytest-warning
