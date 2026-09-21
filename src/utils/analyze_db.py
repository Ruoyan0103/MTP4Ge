"""
Analyze an MLIP-3 CFG database and save one figure (named after the input
CFG, e.g. train-tgap.png for train-tgap.cfg) with stacked histograms of
energy per atom, per-atom force magnitude, and per-structure mean normal
stress (columns), grouped by the `Feature type` tag already present in the
file. The number of rows depends on whether the CFG contains any
othertypes/defects members (bcc, bc8, fcc, hcp, hd, betaTin, st12, V, X, T,
H, B):
  - if present: a 2x3 grid -
      row 1 (coarse) - thermal, elastic, othertypes (bcc/bc8/fcc/hcp/hd/
                       betaTin/st12 combined), defects (V/X/T/H/B combined),
                       liquids.
      row 2 (detail) - only the othertypes/defects members, each kept
                       separate; thermal/elastic/liquids structures are
                       dropped, not shown.
  - otherwise: a single 1x3 row using only the coarse grouping.
`FTYPE_GROUP_COARSE`/`FTYPE_GROUP_DETAIL` map the tag values used in
data/train-tgap.cfg (vac, phonon, diamond, bcc, bc8, fcc, hcp, hd, betaTin,
st12, tet, split, hex, bond, liquid) to a display group; anything unmapped
falls into "other" (coarse row) or is dropped (detail row).

PlusStress in the CFG is the virial [xx, yy, zz, yz, xz, xy] in eV — an
extensive quantity that scales with cell volume, not directly comparable
across structures of different size. Each structure is reduced to one
intensive scalar, the mean normal stress (sxx+syy+szz)/3 in GPa (using
stress = -virial / volume, the same sign convention as
src/utils/convert.py's `_parse_virial_from_atoms`) — the stress analogue of
dividing energy by atom count — so its histogram is "number of structures"
per bin, rather than "number of components". The energy histogram's y-axis
is "number of atoms" instead: each structure contributes its own atom count
to its bin, i.e. atoms-in-bin = structures-in-bin * atoms-per-structure for
a given type.

Usage:
    python src/utils/analyze_db.py data/train.cfg
    python src/utils/analyze_db.py data/train.cfg --outdir results/dataset_analysis
    python src/utils/analyze_db.py data/train.cfg --energy-bin-width 0.02
    python src/utils/analyze_db.py data/train.cfg --linear-scale
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

EV_A3_TO_GPA = 160.21766  # 1 eV/Å³ = 160.21766 GPa


class Grouping:
    """A `Feature type` -> display-group mapping, plus its display order and colors.

    drop_unmapped=False (default): a `Feature type` not in `ftype_group` falls
    back to the "other" group. drop_unmapped=True: it is excluded from the
    histogram entirely instead (used by GROUPING_DETAIL to show only the
    othertypes/defects members, dropping thermal/elastic/liquids structures).
    """

    def __init__(self, ftype_group, group_order, label, drop_unmapped=False):
        self.ftype_group = ftype_group
        self.group_order = group_order
        self.label = label
        self.drop_unmapped = drop_unmapped
        # Fixed, distinct color per group (tab20 has 20 distinct colors),
        # shared across all subplots using this grouping.
        self.colors = {g: plt.cm.tab20(i % 20) for i, g in enumerate(group_order)}

    def classify(self, features):
        group = self.ftype_group.get(features.get("type", ""))
        if group is not None:
            return group
        return None if self.drop_unmapped else "other"


# -- Fine grouping: every structure type kept separate -----------------------
GROUP_ORDER_FINE = [
    "phonon", "elastic", "bcc", "bc8", "fcc", "hcp", "hd", "betaTin", "st12",
    "V", "X", "T", "H", "B", "liquids", "other",
]

FTYPE_GROUP_FINE = {
    "vac": "V",
    "phonon": "phonon",
    "diamond": "elastic",
    "bcc": "bcc",
    "bc8": "bc8",
    "fcc": "fcc",
    "hcp": "hcp",
    "hd": "hd",
    "betaTin": "betaTin",
    "st12": "st12",
    "tet": "T",
    "split": "X",
    "hex": "H",
    "bond": "B",
    "liquid": "liquids",
}

# -- Coarse grouping: bcc/bc8/fcc/hcp/hd/betaTin/st12 -> othertypes; ---------
# -- V/X/T/H/B -> defects ----------------------------------------------------
GROUP_ORDER_COARSE = ["phonon", "elastic", "othertypes", "defects", "liquids", "other"]

FTYPE_GROUP_COARSE = {
    "vac": "defects",
    "phonon": "phonon",
    "diamond": "elastic",
    "bcc": "othertypes",
    "bc8": "othertypes",
    "fcc": "othertypes",
    "hcp": "othertypes",
    "hd": "othertypes",
    "betaTin": "othertypes",
    "st12": "othertypes",
    "tet": "defects",
    "split": "defects",
    "hex": "defects",
    "bond": "defects",
    "liquid": "liquids",
}

GROUPING_FINE = Grouping(FTYPE_GROUP_FINE, GROUP_ORDER_FINE, "by type")
GROUPING_COARSE = Grouping(FTYPE_GROUP_COARSE, GROUP_ORDER_COARSE, "by category")

# -- Detail grouping: only the othertypes/defects members, each kept separate;
# -- thermal/elastic/liquids structures are dropped (not "other") ------------
GROUP_ORDER_DETAIL = ["bcc", "bc8", "fcc", "hcp", "hd", "betaTin", "st12", "V", "X", "T", "H", "B"]

FTYPE_GROUP_DETAIL = {
    "bcc": "bcc",
    "bc8": "bc8",
    "fcc": "fcc",
    "hcp": "hcp",
    "hd": "hd",
    "betaTin": "betaTin",
    "st12": "st12",
    "vac": "V",
    "tet": "T",
    "split": "X",
    "hex": "H",
    "bond": "B",
}

GROUPING_DETAIL = Grouping(
    FTYPE_GROUP_DETAIL, GROUP_ORDER_DETAIL, "othertypes & defects, by type",
    drop_unmapped=True,
)


# ---------------------------------------------------------------------------
# CFG parsing (from analyze_structs._parse_cfg)
# ---------------------------------------------------------------------------

def _parse_cfg(cfg_path):
    with open(cfg_path) as f:
        lines = f.readlines()

    structures = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "BEGIN_CFG":
            i += 1
            continue

        struct = {
            "lattice": None,
            "positions": None,
            "types": None,
            "forces": None,
            "energy": None,
            "stress": None,
            "features": {},
        }
        i += 1
        while lines[i].strip() != "END_CFG":
            key = lines[i].strip()
            if key == "Supercell":
                struct["lattice"] = np.array(
                    [[float(x) for x in lines[i + 1 + k].split()] for k in range(3)]
                )
                i += 4
            elif key.startswith("AtomData"):
                header = key.split(":", 1)[1].split()
                col = {name: idx for idx, name in enumerate(header)}
                has_forces = all(c in col for c in ("fx", "fy", "fz"))
                i += 1
                positions, types, forces = [], [], []
                while lines[i].strip() and lines[i].strip()[0].isdigit():
                    parts = lines[i].split()
                    types.append(int(parts[col["type"]]))
                    positions.append(
                        [
                            float(parts[col["cartes_x"]]),
                            float(parts[col["cartes_y"]]),
                            float(parts[col["cartes_z"]]),
                        ]
                    )
                    if has_forces:
                        forces.append(
                            [
                                float(parts[col["fx"]]),
                                float(parts[col["fy"]]),
                                float(parts[col["fz"]]),
                            ]
                        )
                    i += 1
                struct["positions"] = np.array(positions)
                struct["types"] = types
                struct["forces"] = np.array(forces) if has_forces else None
            elif key == "Energy":
                struct["energy"] = float(lines[i + 1].strip())
                i += 2
            elif key.startswith("PlusStress"):
                struct["stress"] = [float(x) for x in lines[i + 1].split()]
                i += 2
            elif key.startswith("Feature"):
                parts = lines[i].split(None, 2)
                if len(parts) >= 3:
                    struct["features"][parts[1]] = parts[2].strip()
                i += 1
            else:
                i += 1
        structures.append(struct)
        i += 1

    return structures


def classify_group(features, grouping):
    return grouping.classify(features)


def _stacked_group_bar(bin_centers, counts, group_totals, bin_width, ax, grouping):
    groups_present = [g for g in grouping.group_order if group_totals[g] > 0]
    bottom = [0] * len(bin_centers)
    for group in groups_present:
        heights = [counts[b][group] for b in bin_centers]
        ax.bar(
            bin_centers, heights, width=bin_width * 0.9, bottom=bottom,
            label=f"{group} ({group_totals[group]})", color=grouping.colors[group],
        )
        bottom = [b + h for b, h in zip(bottom, heights)]


# ---------------------------------------------------------------------------
# Energy per atom
# ---------------------------------------------------------------------------

def plot_grouped_energy_histogram(
    cfg_path, bin_width=0.01, output_path=None, log_scale=False, xlim=None, show_mean=True,
    ax=None, grouping=GROUPING_FINE,
):
    structures = _parse_cfg(cfg_path)

    counts = defaultdict(Counter)
    group_totals = Counter()
    weighted_sum, weight_total = 0.0, 0
    for struct in structures:
        if struct["energy"] is None:
            continue
        n_atoms = len(struct["positions"])
        e_per_atom = struct["energy"] / n_atoms
        weighted_sum += e_per_atom * n_atoms
        weight_total += n_atoms
        bin_center = round(round(e_per_atom / bin_width) * bin_width, 6)
        group = classify_group(struct["features"], grouping)
        if group is None:
            continue
        # Each structure contributes n_atoms to its bin/group, i.e. number of
        # atoms = number of structures * atoms per structure, for that type.
        counts[bin_center][group] += n_atoms
        group_totals[group] += n_atoms

    bin_centers = sorted(counts)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()
    _stacked_group_bar(bin_centers, counts, group_totals, bin_width, ax, grouping)

    ax.set_xlabel("Energy per atom (eV/atom)")
    ax.set_ylabel("Number of atoms")
    if log_scale:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)

    mean_energy = weighted_sum / weight_total if weight_total else None
    if show_mean and mean_energy is not None:
        ax.axvline(mean_energy, color="black", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_energy:.4f}")

    ax.set_title(f"Energy per atom ({grouping.label})\n{os.path.basename(cfg_path)} ({len(structures)} structures)")
    ax.legend()

    if standalone:
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    return counts, group_totals, mean_energy


# ---------------------------------------------------------------------------
# Force magnitude
# ---------------------------------------------------------------------------

def plot_grouped_force_histogram(
    cfg_path, bin_width=0.05, output_path=None, log_scale=False, xlim=None, show_mean=True,
    ax=None, grouping=GROUPING_FINE,
):
    structures = _parse_cfg(cfg_path)

    counts = defaultdict(Counter)
    group_totals = Counter()
    all_forces = []
    n_skipped = 0
    for struct in structures:
        if struct["forces"] is None:
            n_skipped += 1
            continue
        group = classify_group(struct["features"], grouping)
        magnitudes = np.linalg.norm(struct["forces"], axis=1)
        for f_mag in magnitudes:
            all_forces.append(f_mag)
            if group is None:
                continue
            bin_center = round(round(f_mag / bin_width) * bin_width, 6)
            counts[bin_center][group] += 1
            group_totals[group] += 1

    bin_centers = sorted(counts)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()
    _stacked_group_bar(bin_centers, counts, group_totals, bin_width, ax, grouping)

    ax.set_xlabel("Force magnitude (eV/Å)")
    ax.set_ylabel("Number of atoms")
    if log_scale:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)

    mean_force = sum(all_forces) / len(all_forces) if all_forces else None
    if show_mean and mean_force is not None:
        ax.axvline(mean_force, color="black", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_force:.4f}")

    title = f"Force magnitude ({grouping.label})\n{os.path.basename(cfg_path)} ({len(structures)} structures"
    if n_skipped:
        title += f", {n_skipped} without forces skipped"
    title += ")"
    ax.set_title(title)
    ax.legend()

    if standalone:
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    return counts, group_totals, mean_force


# ---------------------------------------------------------------------------
# Mean normal stress (new)
# ---------------------------------------------------------------------------

def plot_grouped_stress_histogram(
    cfg_path, bin_width=0.5, output_path=None, log_scale=False, xlim=None, show_mean=True,
    ax=None, grouping=GROUPING_FINE,
):
    structures = _parse_cfg(cfg_path)

    counts = defaultdict(Counter)
    group_totals = Counter()
    all_stress = []
    n_skipped = 0
    for struct in structures:
        if struct["stress"] is None or struct["lattice"] is None:
            n_skipped += 1
            continue
        volume = abs(np.linalg.det(struct["lattice"]))
        sxx, syy, szz = (-np.array(struct["stress"][:3]) / volume) * EV_A3_TO_GPA
        mean_normal = (sxx + syy + szz) / 3.0
        all_stress.append(mean_normal)
        bin_center = round(round(mean_normal / bin_width) * bin_width, 6)
        group = classify_group(struct["features"], grouping)
        if group is None:
            continue
        counts[bin_center][group] += 1
        group_totals[group] += 1

    bin_centers = sorted(counts)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()
    _stacked_group_bar(bin_centers, counts, group_totals, bin_width, ax, grouping)

    ax.set_xlabel("Mean normal stress (GPa)")
    ax.set_ylabel("Number of structures")
    if log_scale:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim)

    mean_stress = sum(all_stress) / len(all_stress) if all_stress else None
    if show_mean and mean_stress is not None:
        ax.axvline(mean_stress, color="black", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_stress:.4f}")

    title = f"Mean normal stress ({grouping.label})\n{os.path.basename(cfg_path)} ({len(structures)} structures"
    if n_skipped:
        title += f", {n_skipped} without stress skipped"
    title += ")"
    ax.set_title(title)
    ax.legend()

    if standalone:
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    return counts, group_totals, mean_stress


def _has_detail_groups(cfg_path):
    """True if the CFG has any structures classified into GROUPING_DETAIL
    (i.e. defects or othertypes members), meaning the detail row is non-empty."""
    structures = _parse_cfg(cfg_path)
    return any(
        GROUPING_DETAIL.classify(struct["features"]) is not None
        for struct in structures
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cfg_path", help="Path to an MLIP-3 .cfg database")
    parser.add_argument("--outdir", default="results/dataset_analysis",
                        help="Directory to save <cfg_stem>.png into (default: "
                             "results/dataset_analysis)")
    parser.add_argument("--energy-bin-width", type=float, default=0.01,
                        help="Energy histogram bin width, eV/atom (default: 0.01)")
    parser.add_argument("--force-bin-width", type=float, default=0.05,
                        help="Force histogram bin width, eV/Å (default: 0.05)")
    parser.add_argument("--stress-bin-width", type=float, default=0.5,
                        help="Stress histogram bin width, GPa (default: 0.5)")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false",
                        help="Plot all three y-axes on a linear scale "
                             "(default: log scale, since group sizes span "
                             "several orders of magnitude)")
    parser.add_argument("--no-mean-line", dest="show_mean", action="store_false",
                        help="Don't draw a dashed vertical line at each mean value")
    parser.set_defaults(log_scale=True)
    args = parser.parse_args()

    if not Path(args.cfg_path).exists():
        parser.error(f"cfg_path does not exist: {args.cfg_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{Path(args.cfg_path).stem}.png"

    has_detail = _has_detail_groups(args.cfg_path)

    if has_detail:
        fig, ((ax_e0, ax_f0, ax_s0), (ax_e1, ax_f1, ax_s1)) = plt.subplots(2, 3, figsize=(21, 12))
    else:
        fig, (ax_e0, ax_f0, ax_s0) = plt.subplots(1, 3, figsize=(21, 6))

    # Row 1 (or only row, if no defects/othertypes subgroups are present):
    # coarse grouping (othertypes / defects combined)
    _, _, mean_e = plot_grouped_energy_histogram(
        args.cfg_path, args.energy_bin_width, log_scale=args.log_scale,
        show_mean=args.show_mean, ax=ax_e0, grouping=GROUPING_COARSE,
    )
    _, _, mean_f = plot_grouped_force_histogram(
        args.cfg_path, args.force_bin_width, log_scale=args.log_scale,
        show_mean=args.show_mean, ax=ax_f0, grouping=GROUPING_COARSE,
    )
    _, _, mean_s = plot_grouped_stress_histogram(
        args.cfg_path, args.stress_bin_width, log_scale=args.log_scale,
        show_mean=args.show_mean, ax=ax_s0, grouping=GROUPING_COARSE,
    )

    if has_detail:
        # Row 2: detail grouping (only othertypes/defects members, each separate)
        _, _, mean_e = plot_grouped_energy_histogram(
            args.cfg_path, args.energy_bin_width, log_scale=args.log_scale,
            show_mean=args.show_mean, ax=ax_e1, grouping=GROUPING_DETAIL,
        )
        _, _, mean_f = plot_grouped_force_histogram(
            args.cfg_path, args.force_bin_width, log_scale=args.log_scale,
            show_mean=args.show_mean, ax=ax_f1, grouping=GROUPING_DETAIL,
        )
        _, _, mean_s = plot_grouped_stress_histogram(
            args.cfg_path, args.stress_bin_width, log_scale=args.log_scale,
            show_mean=args.show_mean, ax=ax_s1, grouping=GROUPING_DETAIL,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"  Dataset analysis -> {out_path}")
    print(f"    energy mean = {mean_e:.4f} eV/atom, force mean = {mean_f:.4f} eV/Å, "
          f"stress mean = {mean_s:.4f} GPa")


if __name__ == "__main__":
    main()
