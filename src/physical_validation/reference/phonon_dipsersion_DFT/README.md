# Phonon dispersion DFT reference

DFT phonon dispersion reference data for diamond-cubic Ge, used by
`src/physical_validation/phonon_dispersion.py` to overlay against the MTP-predicted band
structure and total DOS.

- **Supercell size**: 5×5×5
- **Lattice constant**: experimental value at 80 K

## Files
- `band.conf` — phonopy band-structure configuration
- `band.yaml` — phonopy band-structure output
- `raw-data.txt` — dispersion data with segment end points; read directly by `src/physical_validation/phonon_dispersion.py`
- `phonon_dos.dat` — total DOS data; read directly by `src/physical_validation/phonon_dispersion.py`
- `phonon_dispersion_EXP.dat` — experimental dispersion overlay; read directly by `src/physical_validation/phonon_dispersion.py`
