# Examples using FitSNAP library

### `example1.py`

Here we import all the configs in the tantalum example with FitSNAP's `scrape_configs` function.
Then we use the `Calculator` class to compute bispectrums, bispectrum derivatives, and virials for
all configs.

### `example2.py`

Use the FitSNAP library to calculate errors on a test set, without performing a fit. Run this 
example like:

`python example2.py --fitsnap_in /path/to/fitsnap.in --test_dir /path/to/test_dir`

The `--fitsnap_in` precedes the path to the FitSNAP input script that was used for fitting. This 
ensures that we use exactly the same settings that we used when training, like units. **Nothing in
the input script needs to be modified to calculate test errors, we just need to supply it.**

The `--test_dir` precedes the path to a directory containing test data.

**Before running on your own example, change the `pairstyle` variable in this script.**
