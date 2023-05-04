## FitSnap3 NN Tungten - Beryllium example

This example will generate a custom pairwise  NN potential for WBe using the `WBe_PRB2019` dataset. 

### Running this example:

To run this example, use the following command in this directory:

python3 -m fitsnap3 WBe-example.in --overwrite # use --overwrite if you already have fitsnap files
saved.

### Running MD with custom pairwise NN potential in MLIAP Unified.

Deploy the model to MLIAP Unified with:

    python deploy-script.py

Then run MD with:

    mpirun -np P lmp -in in.run
