# Definition of SNAP+ZBL potential.
variable zblcutinner equal 4
variable zblcutouter equal 4.8
variable zblz1 equal 73

# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff 1 1 zbl ${zblz1} ${zblz1}
pair_coeff * * snap Ta.snapcoeff Ta.snapparam Ta  

