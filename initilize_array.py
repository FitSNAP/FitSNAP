import numpy as np
a=np.zeros([1,31])
with open('A_mat_before.csv', 'wb') as abc:
    np.savetxt(abc, a, delimiter=",")
