"""Tests for liquid_rdf.py — validate input generation and RDF computation.

These tests exercise the LAMMPS input writer and RDF-from-dump logic
using synthetic data, without requiring an actual LAMMPS run.
"""

import tempfile
import sys
from pathlib import Path

import numpy as np

# Ensure the parent tests/ directory is on sys.path for utils imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import functions from the module under test
from liquid_rdf.liquid_rdf import (
    write_liquid_input,
    compute_rdf_from_dump,
    _GE_MASS_AMU,
)
from utils import compute_rdf, read_lammps_dump, coordination_number


def test_write_liquid_input_basic():
    """Verify that the LAMMPS input file is written with expected content."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        pot_path = "/fake/path/pot.almtp"
        n_atoms = write_liquid_input(
            workdir, pot_path,
            nx=2, ny=2, nz=2, a0=5.658,
            T=1500.0, steps=5000,
        )

        # 8 atoms per diamond conventional cell × 2×2×2 = 64
        assert n_atoms == 64, f"Expected 64 atoms, got {n_atoms}"

        inp = workdir / "in.liquid"
        assert inp.exists(), "Input file not created"
        content = inp.read_text()

        # Check key elements
        assert "units       metal" in content
        assert "pair_style  mlip" in content
        assert "load_from=/fake/path/pot.almtp" in content
        assert "lattice     diamond 5.658" in content
        assert "fix         nvt all nvt temp 1500.0 1500.0 0.1" in content
        assert "dump        dmp all custom" in content
        assert "id type x y z" in content
        assert f"mass        1 {_GE_MASS_AMU}" in content
        assert "run         5000" in content


def test_write_liquid_input_supercell_dimensions():
    """Verify that nx, ny, nz produce correct supercell and atom count."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        nx, ny, nz = 3, 4, 5
        n_atoms = write_liquid_input(
            workdir, "/fake/pot.almtp",
            nx=nx, ny=ny, nz=nz,
        )
        expected = 8 * nx * ny * nz  # 8 * 60 = 480
        assert n_atoms == expected, f"Expected {expected} atoms, got {n_atoms}"

        content = (workdir / "in.liquid").read_text()
        assert f"block 0 {nx} 0 {ny} 0 {nz}" in content


def test_write_liquid_input_custom_a0_and_temperature():
    """Verify custom lattice constant and temperature are propagated."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        write_liquid_input(
            workdir, "/fake/pot.almtp",
            a0=5.500, T=2000.0,
        )
        content = (workdir / "in.liquid").read_text()
        assert "lattice     diamond 5.5" in content
        assert "temp 2000.0 2000.0 0.1" in content


def test_compute_rdf_from_dump_with_synthetic_data():
    """Test RDF computation using a synthetic LAMMPS dump file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Create a minimal synthetic dump with 2 frames (2-atom system)
        dump_path = tmpdir / "dump.lammpstrj"
        dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.000000 5.000000
0.000000 5.000000
0.000000 5.000000
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 2.0 0.0 0.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.000000 5.000000
0.000000 5.000000
0.000000 5.000000
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 2.5 0.0 0.0
"""
        dump_path.write_text(dump_content)

        r, g_r, cells = compute_rdf_from_dump(
            dump_path, n_atoms=2, rmax=5.0, nbins=50,
        )

        assert len(r) == 50, f"Expected 50 bins, got {len(r)}"
        assert len(g_r) == 50
        assert cells.shape == (2, 3, 3)
        # Both frames averaged: atoms at (0,0,0)–(2.0,0,0) and (0,0,0)–(2.5,0,0)
        # should produce a peak in the averaged g(r)
        peak_val = g_r.max()
        assert peak_val > 0, "g(r) should have non-zero values"


def test_utils_read_lammps_dump():
    """Verify that utils.read_lammps_dump correctly parses dump files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        dump_path = tmpdir / "dump.lammpstrj"
        dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.0 0.0 0.0
3 1 0.0 1.0 0.0
"""
        dump_path.write_text(dump_content)

        frames = read_lammps_dump(dump_path, n_atoms=3)
        assert len(frames) == 1, f"Expected 1 frame, got {len(frames)}"
        cell, positions = frames[0]
        assert cell.shape == (3, 3)
        assert positions.shape == (3, 3)
        # Check the box is diagonal 10×10×10
        np.testing.assert_array_almost_equal(cell, np.diag([10.0, 10.0, 10.0]))


def test_utils_compute_rdf():
    """Verify that compute_rdf returns sensible g(r) for a simple system."""
    # Two atoms at fixed distance in a box
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cell = np.diag([10.0, 10.0, 10.0])

    r, g_r = compute_rdf(positions, cell, rmax=8.0, nbins=200)
    assert len(r) == 200
    assert len(g_r) == 200
    # There should be a peak near 2.5 Å
    idx = np.argmax(g_r)
    assert 2.0 < r[idx] < 3.5, f"Peak at unexpected r={r[idx]:.2f}"


def test_utils_coordination_number():
    """Verify coordination number computation."""
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cell = np.diag([10.0, 10.0, 10.0])
    r, g_r = compute_rdf(positions, cell, rmax=8.0, nbins=200)

    volume = abs(np.linalg.det(cell))
    rho = 2 / volume
    cn = coordination_number(r, g_r, rho, r_cutoff=3.5)
    # With 2 atoms and one neighbor at 2.5 Å, CN should be ~1
    assert 0.5 < cn < 2.0, f"Unexpected coordination number: {cn:.2f}"


def test_load_ref_rdf():
    """Verify reference RDF loading."""
    from utils import load_ref_rdf

    # None path
    r, g = load_ref_rdf(None)
    assert r is None
    assert g is None

    # Non-existent path
    r, g = load_ref_rdf("/nonexistent/path.dat")
    assert r is None
    assert g is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
