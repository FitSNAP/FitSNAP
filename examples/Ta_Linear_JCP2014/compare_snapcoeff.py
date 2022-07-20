import numpy as np

data = np.loadtxt('Ta_pot.snapcoeff', max_rows=31, delimiter=',', skiprows=4)

data_old = np.loadtxt('20May21_Standard/Ta_pot.snapcoeff', max_rows=31, delimiter=',', skiprows=4)

diff = np.abs(data-data_old)

print(f"Max abs difference between new coeff and standard coeff: {np.max(diff)}")
