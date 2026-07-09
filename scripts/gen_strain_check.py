"""Generate 3 strained Ge diamond structures for OVITO inspection (eps=0.05)."""
import numpy as np
from ase.build import bulk
from ase.io import write

a0 = 5.76
atoms0 = bulk("Ge", "diamond", a=a0, cubic=True)
cell0 = np.array(atoms0.get_cell())
eps = 0.05

# 1. Hydrostatic: scale all vectors uniformly
hc = (1.0 + eps) * cell0
at = atoms0.copy()
at.set_cell(hc, scale_atoms=True)
write("strain_hydrostatic.xyz", at)
print(f"hydrostatic   a = {at.get_cell()[0,0]:.4f} Å  (expected {a0*(1+eps):.4f})")

# 2. Uniaxial [100]: scale only the a vector
uc = cell0.copy()
uc[0] = (1.0 + eps) * cell0[0]
at = atoms0.copy()
at.set_cell(uc, scale_atoms=True)
write("strain_uniaxial_100.xyz", at)
print(f"uniaxial_100  a = {at.get_cell()[0,0]:.4f} Å  b = {at.get_cell()[1,1]:.4f} Å")

# 3. Shear: add eps*|a0| to the y-component of the a vector
sc = cell0.copy()
sc[0, 1] += eps * a0
at = atoms0.copy()
at.set_cell(sc, scale_atoms=True)
write("strain_shear.xyz", at)
print(f"shear         a_xy = {at.get_cell()[0,1]:.4f} Å  (expected {eps*a0:.4f})")