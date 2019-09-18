# Definition of SNAP+ZBL potential.
variable zblcutinner equal 4
variable zblcutouter equal 4.8
variable zblz equal 73

# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff 1 1 zbl ${zblz} ${zblz}
pair_coeff * * snap TaJCP2014.snapcoeff TaJCP2014.snapparam Ta

