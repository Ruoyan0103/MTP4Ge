# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style hybrid/overlay mtp nlh
pair_coeff * * mtp ${POT_PATH} 0
pair_coeff 1 1 nlh 32 32 0.05121 0.50355 0.44524 49.82156 12.56998 4.13684 1.28 2.0


# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol fmax fnorm
thermo_modify norm no
