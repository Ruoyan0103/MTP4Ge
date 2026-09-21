"""Shared VASP single-point calculator settings for the structure-generation scripts."""

from ase.calculators.vasp import Vasp


def set_cal():
    """Static single-point VASP calculator shared by all structure-generation scripts."""
    return Vasp(
        command='srun vasp_std',
        setups={'Ge': '_d'},

        # Initialisation:
        istart=0,          # (Default = 0; from scratch)

        # Ionic:
        ibrion=-1,         # (Static calculation, default for nsw=0)
        nsw=0,             # (Max ionic steps)

        # Electronic:
        ediff=1E-07,       # (SCF energy convergence; eV)
        ismear=0,          # (Electronic temperature, Gaussian smearing)
        sigma=0.05,        # (Smearing value in eV)
        nelm=100,          # (Max SCF steps)
        gga='PE',          # (Exchange-correlation functional)

        # Plane wave basis set:
        encut=500,         # (Default = largest ENMAX in the POTCAR file)
        prec='Accurate',   # (Accurate forces are required)
        lasph=True,        # (Non-spherical elements included)

        # Reciprocal space
        kspacing=0.15,     # (Smallest spacing between k points in 1/A)
        kgamma=True,       # (Default = Gamma centered)

        # Parallelisation:
        ncore=4,

        # Output features:
        lwave=False,       # (WAVECAR is NOT written out)
        lcharg=False,      # (CHGCAR is NOT written out)
    )
