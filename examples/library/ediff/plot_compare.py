import numpy as np
import matplotlib.pyplot as plt

dat1 = np.loadtxt("e_vs_b_os.dat")
dat2 = np.loadtxt("e_vs_b.dat")

l1 = np.shape(dat1)[0]
l2 = np.shape(dat2)[0]

x1 = np.linspace(0,100, l1)
x2 = np.linspace(0,100, l2)

plt.plot(x1, dat1[:,2], 'b', linewidth=5.0)
plt.plot(x2, dat2[:,2], 'r', linewidth=5.0)

plt.xlabel("Epochs")
plt.ylabel("|ΔE| (eV)")
#plt.yscale("log")

plt.legend(["With OS", "Without OS"])

plt.grid(axis="both")
plt.savefig("oversample_compare.png", dpi=500)