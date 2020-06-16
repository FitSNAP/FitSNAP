# Definition of SNAP+ZBL potential.
set type 1 charge 1e-08
set type 2 charge -1e-08
variable zblcutinner equal 4
variable zblcutouter equal 4.2
variable zblz1 equal 49
variable zblz2 equal 15
variable rcoul equal 10

# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay coul/long ${rcoul} &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff * * coul/long
pair_coeff 1 1 zbl ${zblz1} ${zblz1}
pair_coeff 1 2 zbl ${zblz1} ${zblz2}
pair_coeff 2 2 zbl ${zblz2} ${zblz2}
pair_coeff * * snap InP.snapcoeff In P InP.snapparam In P
kspace_style ewald 1.0e-5

