# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
#pair_style hybrid/overlay table linear 10000 mlip load_from=${POT_PATH}
#pair_coeff * * table /scratch/project_2012355/Paper_3/00-subsets/07-short_range/01-SW_joining/tables/nlh-MTP.table NLH_GE
#pair_coeff * * mlip

pair_style  mlip load_from=${POT_PATH}
pair_coeff  * *

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
