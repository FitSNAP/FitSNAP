pair_style	hybrid/overlay snap zbl 4.0 4.8 spin/exchange/biquadratic 5.0
pair_coeff	* * snap Fe_pot_snappy.snapcoeff Fe_pot_snappy.snapparam Fe
pair_coeff	* * zbl 26 26
pair_coeff      * * spin/exchange/biquadratic biquadratic 5.0 0.2827 -4.747 0.7810 -0.03619 -2.973 0.5273 1 offset yes

#pair_style     hybrid/overlay snap zbl 4.0 4.8
#pair_coeff     * * snap Fe_pot_snappy.snapcoeff Fe_pot_snappy.snapparam Fe
#pair_coeff     * * zbl 26 26

# Setup neighbor style
neighbor 1.0    bin
neigh_modify    every 1 check yes delay 1
