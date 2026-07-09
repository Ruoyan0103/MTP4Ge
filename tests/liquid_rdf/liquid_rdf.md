### Aim: generate lammps input file which is used for producing liquid structure
### Format: python code
### Protocol: Two-stage NPT → NVT

### Stage 1 — NPT Equilibration
- Ensemble: NPT (Nose-Hoover thermostat + barostat)
- Temperature: 1500 K (liquid)
- Pressure: 1 bar
- Purpose: let the density relax to the true liquid density at 1500 K / 1 bar
- Steps: 20000 (20 ps at dt=1 fs)

### Stage 2 — NVT Production
- Ensemble: NVT (Nose-Hoover thermostat)
- Temperature: 1500 K
- Purpose: collect dump frames for RDF computation at the equilibrated density
- Steps: 10000 (10 ps)
- Dump: custom dump (id type x y z) every 100 steps

### Step3: After simulation, compute rdf from dump file, output rdf figure

### Note: For computing rdf, functions in /scratch/project_2012355/Paper_3/MTP4Ge/tests/utils.py can be used. Generate python code in this folder.
### AFS/NFS: all file writes must use tempfile + shutil.move pattern (write to /tmp, then move)
