"""Shared VASP single-point calculator settings.

Single source of truth for VASP DFT physics parameters, used by both the
structure-generation scripts and the active-learning DFT tagging step
(src/active_learning_structs/active_learning.py, tagging_engine: vasp).
Cluster/job settings (command, ncore, kpar, pp path/version, output
directory) vary per caller and per cluster, so they stay overridable via
keyword arguments rather than being baked in here.
"""

from ase.calculators.vasp import Vasp


def set_cal(**overrides):
    """Static single-point VASP calculator shared by all VASP tagging callers.

    Physics settings (ENCUT, EDIFF, smearing, functional, precision, k-point
    spacing, ...) are fixed here so every VASP tagging run uses identical DFT
    parameters. Pass keyword arguments to override or add settings (e.g.
    directory=..., ncore=..., kpar=..., pp_version=...).
    """
    kwargs = dict(
        command='srun vasp_std',
        setups={'Ge': '_d'},

        # Initialisation:
        istart=0,          # (Default = 0; from scratch)

        # Ionic:
        ibrion=-1,         # (Static calculation, default for nsw=0)
        nsw=0,             # (Max ionic steps)
        isif=2,            # (Stress/relaxation flag; forces only)

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
        kpar=1,

        # Output features:
        lwave=False,       # (WAVECAR is NOT written out)
        lcharg=False,      # (CHGCAR is NOT written out)
    )
    kwargs.update(overrides)
    return Vasp(**kwargs)
