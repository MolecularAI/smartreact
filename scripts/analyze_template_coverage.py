#!/usr/bin/env python3
"""
Analyzes SmartReact template coverage across SMARTS-RX hierarchy levels
and generates a 2x3 grid of chord diagrams — same node layout across all
six panels, 6 reactions per panel, each reaction a distinct color.

Arcs connecting the same Level 1 pair are fanned out angularly so they
appear side-by-side rather than overlapping.

Outputs:
  - Console: coverage stats at each SMARTS-RX level with unique pair counts
  - smartreact_paper/figures/template_coverage_chord_6panel.pdf
  - smartreact_paper/figures/mitsunobu_l3_combinations.tex
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src/smartreact/data"

# Paul Tol bright palette — 6 maximally distinct colors, colorblind-safe
COLORS_6 = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow-olive
    "#66CCEE",  # cyan
    "#AA3377",  # purple
]

# Six panels of 6 reactions each
PANELS: dict[str, list[str]] = {
    "Pd-catalyzed C-C coupling": [
        "ART", "heck", "negishi", "pinacolatoborylation", "sonogashira", "suzuki",
    ],
    "Other C-C forming": [
        "B-H", "C_C_decarboxylation", "chan_lam",
        "cross_electrophile_coupling", "deoxygenative_coupling", "horner_wadsworth_emmons",
    ],
    "Amide & amine forming": [
        "amide_coupling", "amine_acetylation", "amine_sulfonation",
        "reductive_amination_aldehyde", "reductive_amination_ketone", "urea_formation",
    ],
    "N/S nucleophilic substitution": [
        "mitsunobu", "sn2_nheterocycle", "snar_amine",
        "snar_thiol", "ullmann_X", "williamson",
    ],
    "O-functionalization": [
        "ester_schotten_baumann", "ester_sulfonic_schotten_baumann",
        "esterification", "oxadiazole_condensation", "snar_alcohol", "ullmann_phenol",
    ],
    "Heterocycle synthesis": [
        "imidazole_condensation_acid", "imidazole_condensation_amine",
        "imidazole_Xketone_synthesis", "tetrazole_synthesis",
        "triazole_synthesis_1", "triazole_synthesis_2",
    ],
}

NODE_ORDER = [
    "X", "XF", "XKetone", "Mesylate",
    "AcidX", "Acid", "Thioacid", "Ester",
    "Sulfonyl", "Phosphate",
    "Aldehyde", "Ketone",
    "Alkene/MichaelAcceptor", "Alkyne",
    "Boronate", "Boronic",
    "Metals",
    "Nitrile", "Azide", "IsoCyanate", "ThioisoCyanate",
    "Amine", "Amidine", "HydroxyAmidine", "Hydrazine", "NitrogenHeterocycle",
    "Alcohol", "Phenol",
    "Thiol", "Thiophenol",
]

# Angular offset step (radians) between co-located arcs
ANGLE_OFFSET_STEP = 0.04


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_keys() -> pd.DataFrame:
    rows = []
    with open(DATA / "keys.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append({"level1": parts[0], "level2": parts[1], "level3": parts[2]})
    return pd.DataFrame(rows)


def load_templates() -> pd.DataFrame:
    return pd.read_csv(DATA / "templates.txt", sep="\t", header=0)


def extract_cat_pairs(templates: pd.DataFrame) -> list[tuple[str, str, str]]:
    pairs = []
    for _, row in templates.iterrows():
        cats = str(row["reactant_categories"]).split(".")
        if len(cats) == 2:
            pairs.append((row["reaction_name"], cats[0], cats[1]))
    return pairs


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_coverage_analysis(keys, cat_pairs, l3_to_l1, l3_to_l2, n_templates):
    all_cats = {c for _, c1, c2 in cat_pairs for c in (c1, c2)}
    used_l3 = all_cats
    used_l2 = {l3_to_l2[c] for c in all_cats if c in l3_to_l2}
    used_l1 = {l3_to_l1[c] for c in all_cats if c in l3_to_l1}
    total_l3 = keys["level3"].nunique()
    total_l2 = keys["level2"].nunique()
    total_l1 = keys["level1"].nunique()

    def unique_pairs(map_fn):
        return {frozenset([map_fn.get(c1), map_fn.get(c2)])
                for _, c1, c2 in cat_pairs
                if map_fn.get(c1) and map_fn.get(c2)}

    pairs_l1 = unique_pairs(l3_to_l1)
    pairs_l2 = unique_pairs(l3_to_l2)
    pairs_l3 = {frozenset([c1, c2]) for _, c1, c2 in cat_pairs}

    print("=" * 60)
    print("SMARTS-RX Key Coverage in Templates")
    print("=" * 60)
    print(f"  Reactions: {len({r for r, *_ in cat_pairs})}    Templates: {n_templates}")
    print()
    print(f"  {'Level':<10} {'Used':>6} / {'Total':<6}  {'%':>5}   {'Unique pairs':>14}")
    print(f"  {'-'*55}")
    for label, used, total, pairs in [
        ("Level 1", len(used_l1), total_l1, len(pairs_l1)),
        ("Level 2", len(used_l2), total_l2, len(pairs_l2)),
        ("Level 3", len(used_l3), total_l3, len(pairs_l3)),
    ]:
        print(f"  {label:<10} {used:>6} / {total:<6}  {used/total*100:>4.0f}%   {pairs:>14}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Chord diagram helpers
# ---------------------------------------------------------------------------

def _cubic_arc(ax, a1, a2, R, color, alpha=0.82, lw=2.1):
    """Draw a cubic bezier arc between two angles on a circle of radius R."""
    x1, y1 = np.cos(a1) * R, np.sin(a1) * R
    x2, y2 = np.cos(a2) * R, np.sin(a2) * R
    # Control points pulled 25% toward center — smooth inward bow
    verts = [(x1, y1), (x1 * 0.25, y1 * 0.25), (x2 * 0.25, y2 * 0.25), (x2, y2)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none",
                           edgecolor=color, alpha=alpha, lw=lw))


def _label_fontsize(name: str) -> float:
    n = len(name)
    if n <= 4:   return 19.0
    if n <= 7:   return 17.0
    if n <= 11:  return 15.0
    if n <= 15:  return 14.0
    return 13.0


def _draw_panel(ax, cat_pairs, l3_to_l1, global_node_order,
                l1_template_counts, reactions: list[str], title: str):
    """Draw one chord diagram panel with only active nodes on the circle."""
    reaction_color = {r: COLORS_6[i] for i, r in enumerate(reactions)}
    reaction_set = set(reactions)

    # Deduplicated (reaction, l1_1, l1_2) arcs for this panel
    seen: set[tuple[str, str, str]] = set()
    raw_arcs: list[tuple[str, str, str]] = []
    for reaction, c1, c2 in cat_pairs:
        if reaction not in reaction_set:
            continue
        l1_1 = l3_to_l1.get(c1)
        l1_2 = l3_to_l1.get(c2)
        if l1_1 and l1_2:
            key = (reaction, l1_1, l1_2)
            if key not in seen:
                seen.add(key)
                raw_arcs.append(key)

    active_nodes = {n for _, n1, n2 in raw_arcs for n in (n1, n2)}

    # Panel-specific layout: only active nodes, preserving global order
    panel_nodes = [n for n in global_node_order if n in active_nodes]
    n_nodes = len(panel_nodes)
    # Half-slot offset so no node sits exactly at 12 o'clock
    start_angle = np.pi / 2 - np.pi / n_nodes
    angles = {node: start_angle - 2 * np.pi * i / n_nodes
              for i, node in enumerate(panel_nodes)}

    # Group arcs by node-pair for angular offset computation
    pair_to_arcs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for reaction, l1_1, l1_2 in raw_arcs:
        pair_to_arcs[(l1_1, l1_2)].append(reaction)

    R = 1.0
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="#DDDDDD", lw=0.8))

    # Draw arcs — co-located pairs fanned out by angular offset
    for (l1_1, l1_2), reactions_here in pair_to_arcs.items():
        n = len(reactions_here)
        reactions_here_sorted = sorted(reactions_here, key=lambda r: reactions.index(r))
        for i, reaction in enumerate(reactions_here_sorted):
            offset = (i - (n - 1) / 2) * ANGLE_OFFSET_STEP
            color = reaction_color[reaction]
            if l1_1 == l1_2:
                a = angles[l1_1]
                x, y = np.cos(a) * R, np.sin(a) * R
                r_loop = 0.055 + i * 0.04
                ax.add_patch(plt.Circle(
                    (x * (1.0 + r_loop), y * (1.0 + r_loop)), r_loop,
                    fill=False, color=color, alpha=0.82, lw=2.1))
            else:
                _cubic_arc(ax,
                           angles[l1_1] + offset,
                           angles[l1_2] + offset,
                           R, color)

    # Active nodes only — sized by template count, labeled with adaptive font
    max_count = max(l1_template_counts.values(), default=1)
    for node in panel_nodes:
        angle = angles[node]
        x, y = np.cos(angle) * R, np.sin(angle) * R
        count = l1_template_counts.get(node, 1)
        size = 35 + 160 * (count / max_count) ** 0.5
        ax.scatter(x, y, s=size, color="#222222", zorder=6)

        lx, ly = np.cos(angle) * 1.14, np.sin(angle) * 1.14
        rot = np.degrees(angle)
        ha = "left"
        if np.cos(angle) < 0:
            rot += 180
            ha = "right"
        ax.text(lx, ly, node, ha=ha, va="center",
                fontsize=_label_fontsize(node), fontweight="bold", color="#222222",
                rotation=rot, rotation_mode="anchor")

    # Extend ylim downward to carve out space for the legend below the circle
    # without intruding into the neighbouring row
    handles = [mpatches.Patch(color=reaction_color[r], label=r.replace("_", " "))
               for r in reactions]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1),
              fontsize=15, framealpha=0.97, edgecolor="#CCCCCC",
              ncol=2, handlelength=1.8, handleheight=1.4)

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-2.8, 1.65)   # extra room at bottom keeps legend inside the axes


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def draw_six_panel_chord(cat_pairs, l3_to_l1, l1_template_counts) -> plt.Figure:
    # Global ordering — determines clockwise sequence within each panel
    all_nodes_used = {l3_to_l1[c]
                      for _, c1, c2 in cat_pairs
                      for c in (c1, c2) if c in l3_to_l1}
    global_node_order = [n for n in NODE_ORDER if n in all_nodes_used]
    global_node_order += sorted(all_nodes_used - set(global_node_order))

    fig, axes = plt.subplots(2, 3, figsize=(28, 22))
    fig.subplots_adjust(hspace=0.28, wspace=0.05, top=0.95, bottom=0.06)

    for ax, (panel_title, reactions) in zip(axes.flat, PANELS.items()):
        _draw_panel(ax, cat_pairs, l3_to_l1, global_node_order,
                    l1_template_counts, reactions, panel_title)

    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def save_l3_combinations_table(reaction_name: str, templates: pd.DataFrame,
                                out_path: Path) -> None:
    df = templates[templates["reaction_name"] == reaction_name].copy()
    df[["cat1", "cat2"]] = df["reactant_categories"].str.split(".", expand=True)
    pairs = df[["cat1", "cat2"]].drop_duplicates().sort_values(["cat1", "cat2"]).reset_index(drop=True)

    lines = [
        f"% {reaction_name} -- all {len(pairs)} Level 3 reactant category combinations",
        r"% Generated from smartreact templates.txt",
        r"\begin{longtable}{ll}",
        f"\\caption{{All Level 3 reactant category combinations for the {reaction_name.replace('_', ' ').title()} reaction ({len(pairs)} combinations).}} \\label{{tab:{reaction_name}_l3}} \\\\",
        r"\toprule",
        r"\textbf{Reactant 1 (L3)} & \textbf{Reactant 2 (L3)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Reactant 1 (L3)} & \textbf{Reactant 2 (L3)} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{2}{r}{\textit{continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    prev_cat1 = None
    for _, row in pairs.iterrows():
        if prev_cat1 is not None and row.cat1 != prev_cat1:
            lines.append(r"\midrule")
        lines.append(f"{_tex_escape(row.cat1)} & {_tex_escape(row.cat2)} \\\\")
        prev_cat1 = row.cat1

    lines.append(r"\end{longtable}")
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    keys = load_keys()
    templates = load_templates()

    l3_to_l1 = dict(zip(keys["level3"], keys["level1"]))
    l3_to_l2 = dict(zip(keys["level3"], keys["level2"]))

    cat_pairs = extract_cat_pairs(templates)

    print_coverage_analysis(keys, cat_pairs, l3_to_l1, l3_to_l2, n_templates=len(templates))

    l1_counts: Counter = Counter()
    for _, c1, c2 in cat_pairs:
        for c in (c1, c2):
            if c in l3_to_l1:
                l1_counts[l3_to_l1[c]] += 1

    out_dir = ROOT / "smartreact_paper/figures"
    out_dir.mkdir(exist_ok=True)

    save_l3_combinations_table("mitsunobu", templates,
                                out_dir / "mitsunobu_l3_combinations.tex")

    fig = draw_six_panel_chord(cat_pairs, l3_to_l1, dict(l1_counts))
    for ext in ("pdf", "png"):
        out = out_dir / f"template_coverage_chord_6panel.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
