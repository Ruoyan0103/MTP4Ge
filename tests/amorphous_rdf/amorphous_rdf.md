### Aim: generate lammps input file which is used for producing amorphous structures
### Format: python code
### Protocol: NPT Equilibration → NPT Quench → Energy Minimization

### Stage 1 — NPT Equilibration at 2000 K
- Ensemble: NPT (Nose-Hoover thermostat + barostat), zero pressure (hydrostatic, variable cell volume)
- Temperature: 2000 K
- Pressure: 0 bar (zero-pressure variable cell volume)
- Purpose: equilibrate the liquid configuration with the current potential before cooling
- Steps: 100000 (100 ps at dt=1 fs)
- supercell: 3x3x3 unit cells (216 atoms for diamond cubic Ge)
- Initial config: crystalline structure (e.g., diamond cubic for Ge) with random velocities corresponding to 2000 K

### Stage 2 — NPT Quenching from 2000 K to 500 K
- Ensemble: NPT (Nose-Hoover thermostat + barostat), zero pressure (hydrostatic, variable cell volume)
- Temperature: ramp linearly from 2000 K → 500 K
- Pressure: 0 bar
- Cooling rate: 10^12 K/s (1 K per ps, 1 fs timestep → 1 mK/step)
- Steps: 1500000 (1500 ps at dt=1 fs)

### Stage 3 — Energy Minimization
- Relax atomic positions and cell size/shape to local energy minimum
- Use LAMMPS minimize (conjugate gradient or FIRE)
- Purpose: obtain the inherent structure of the quenched amorphous sample
- Dump file: output atomic positions and cell parameters after minimization for RDF analysis, frequency: every 1 step

### Step 4: After simulation, compute rdf from dump file, output rdf figure

### Note: For computing rdf, functions in /scratch/project_2012355/Paper_3/MTP4Ge/tests/utils.py can be used. Generate python code in this folder.
### AFS/NFS: all file writes must use tempfile + shutil.move pattern (write to /tmp, then move)

