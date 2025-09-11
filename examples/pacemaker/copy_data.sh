#!/bin/bash

# Script to copy pacemaker data files to FitSNAP examples
# Run this script from the FitSNAP examples/pacemaker directory

echo "Copying pacemaker data files..."

# Source directory (adjust path as needed)
PACEMAKER_EXAMPLES="/Users/mitch/github/python-ace-alphataubio/examples"

# Copy Cu-I data
if [ -f "$PACEMAKER_EXAMPLES/Cu-I/Cu_df1_A1_A2_A3_EV_elast_phon.pckl.gzip" ]; then
    cp "$PACEMAKER_EXAMPLES/Cu-I/Cu_df1_A1_A2_A3_EV_elast_phon.pckl.gzip" Cu-I/
    echo "✓ Copied Cu-I data file"
else
    echo "✗ Cu-I data file not found"
fi

# Copy Cu-II data  
if [ -f "$PACEMAKER_EXAMPLES/Cu-II/Cu_df2_1k.pkl.gzip" ]; then
    cp "$PACEMAKER_EXAMPLES/Cu-II/Cu_df2_1k.pkl.gzip" Cu-II/
    echo "✓ Copied Cu-II data file"
else
    echo "✗ Cu-II data file not found"
fi

# Copy Ethanol data
if [ -f "$PACEMAKER_EXAMPLES/Ethanol/ethanol.pckl.gzip" ]; then
    cp "$PACEMAKER_EXAMPLES/Ethanol/ethanol.pckl.gzip" Ethanol/
    echo "✓ Copied Ethanol data file"
else
    echo "✗ Ethanol data file not found"
fi

# Copy HEA data
if [ -f "$PACEMAKER_EXAMPLES/HEA/HEA_randII_example.pckl.gzip" ]; then
    cp "$PACEMAKER_EXAMPLES/HEA/HEA_randII_example.pckl.gzip" HEA/
    echo "✓ Copied HEA data file"
else
    echo "✗ HEA data file not found"
fi

echo "Data file copying complete!"
echo ""
echo "Note: Make sure to install required dependencies:"
echo "  pip install pyace pandas"
echo ""
echo "To run examples:"
echo "  cd [example_directory]"
echo "  mpirun -np 4 python -m fitsnap3 [input_file].in"
