### Aim: generate lammps input file which is used for producing liquid structure
### Format: python code
### Protocol: Three-stage NVT → NPT → NVT
### Timestep: 0.0005 ps (0.5 fs)

### Stage 0 — NVT Pre-heat
- Ensemble: NVT (Nose-Hoover thermostat)
- Temperature: 2000 K (liquid, default)
- Purpose: melt the diamond-cubic seed gently at constant volume before the barostat engages
- Steps: 2000 (1 ps at dt=0.5 fs)

### Stage 1 — NPT Equilibration
- Ensemble: NPT (Nose-Hoover thermostat + isotropic barostat)
- Temperature: 2000 K (liquid, default)
- Pressure: 0 bar
- Purpose: let the density relax to the true liquid density at 2000 K / 0 bar
- Steps: 20000 (10 ps at dt=0.5 fs)

### Stage 2 — NVT Production
- Ensemble: NVT (Nose-Hoover thermostat)
- Temperature: 2000 K
- Purpose: collect dump frames for RDF computation at the equilibrated density
- Steps: 2000 (1 ps at dt=0.5 fs)
- Dump: custom dump (id type x y z) every 100 steps

### Step3: After simulation, compute rdf from dump file, output rdf figure

### Note: For computing rdf, functions in /scratch/project_2012355/Paper_3/MTP4Ge/tests/utils.py can be used. Generate python code in this folder.
### AFS/NFS: all file writes must use tempfile + shutil.move pattern (write to /tmp, then move)
