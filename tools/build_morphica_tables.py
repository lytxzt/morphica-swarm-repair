#!/usr/bin/env python3
"""Single-source table generator for MORPHICA v2 final artifacts.

Usage:
    python tools/build_morphica_tables.py                    # output to stdout
    python tools/build_morphica_tables.py --md docs/tables/  # write .md tables
    python tools/build_morphica_tables.py --csv docs/tables/ # write .csv tables
    python tools/build_morphica_tables.py --all docs/tables/ # both .md and .csv

All data paths are defined in the manifest (JSON) — no hardcoded numbers.
"""

import re, os, sys, json, argparse
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_log(path):
    """Return (cases_list, steps_list) from an Experiment_Statistic output file."""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    m1 = re.search(r'case:\s*\[(.*?)\]', text, re.DOTALL)
    m2 = re.search(r'connect_step:\s*\[(.*?)\]', text, re.DOTALL)
    m3 = re.search(r'avg_connect_step:\s*([\d.]+)', text)
    if m1 and m2:
        cases = [int(x.strip()) for x in m1.group(1).split(',')]
        steps = [int(x.strip()) for x in m2.group(1).split(',')]
        return cases, steps
    if m3:
        return [], [float(m3.group(1))]
    return None, None

def stats(arr, as_int=True):
    a = np.array(arr, dtype=np.float64)
    s = {'n': len(a), 'mean': float(np.mean(a)),
         'P50': float(np.percentile(a, 50)),
         'P90': float(np.percentile(a, 90)),
         'P95': float(np.percentile(a, 95)),
         'max': float(np.max(a))}
    if as_int:
        for k in ['P50', 'P90', 'P95', 'max']:
            s[k] = int(round(s[k]))
    return s

def paired_stats(a_steps, b_steps):
    """Paired comparison: delta, win-rate, bootstrap 95% CI."""
    diffs = [a - b for a, b in zip(a_steps, b_steps)]
    d = np.array(diffs)
    wins = int(np.sum(d < 0))
    ties = int(np.sum(d == 0))
    losses = int(np.sum(d > 0))
    # bootstrap CI
    rng = np.random.RandomState(42)
    means = []
    for _ in range(10000):
        idx = rng.randint(0, len(d), len(d))
        means.append(np.mean(d[idx]))
    ci_lo, ci_hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    return {
        'mean_delta': float(np.mean(d)),
        'std_delta': float(np.std(d)),
        'wins': wins, 'ties': ties, 'losses': losses,
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
    }

def fmt_delta(v, as_pct=False):
    if as_pct:
        return f"{v:+.1f}%"
    return f"{v:+.2f}"

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'morphica_manifest.json')

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    # Embedded fallback manifest
    return _default_manifest()

def _default_manifest():
    LOG = "Logs"
    return {
        "cells": [
            {"N": 100, "d": 25}, {"N": 100, "d": 50}, {"N": 100, "d": 75},
            {"N": 200, "d": 50}, {"N": 200, "d": 100}, {"N": 200, "d": 150},
        ],
        "datasets": {
            "A+S_100case": {
                "description": "A+S 100-case (Plan AP)",
                "paper_label": "MORPHICA-A+S",
                "case_count": 100,
                "paths": {
                    "100_d25": f"{LOG}/plan_ap_as_stress/n100_d25/MORPHICA_v2_d25_100cases.txt",
                    "100_d50": f"{LOG}/plan_ap_as_stress/n100_d50/MORPHICA_v2_d50_100cases.txt",
                    "100_d75": f"{LOG}/plan_ap_as_stress/n100_d75/MORPHICA_v2_d75_100cases.txt",
                    "200_d50": f"{LOG}/plan_ap_as_stress/n200_d50/MORPHICA_v2_d50_100cases.txt",
                    "200_d100": f"{LOG}/plan_ap_as_stress/n200_d100/MORPHICA_v2_d100_100cases.txt",
                    "200_d150": f"{LOG}/plan_ap_as_stress/n200_d150/MORPHICA_v2_d150_100cases.txt",
                },
            },
            "FullX_100case": {
                "description": "Full Plan X 100-case (Plan AI + BM)",
                "paper_label": "MORPHICA-Full",
                "case_count": 100,
                "paths": {
                    "100_d25": f"{LOG}/plan_bm_full_x_n100/n100_d25/MORPHICA_v2_d25_100cases.txt",
                    "100_d50": f"{LOG}/plan_bm_full_x_n100/n100_d50/MORPHICA_v2_d50_100cases.txt",
                    "100_d75": f"{LOG}/plan_bm_full_x_n100/n100_d75/MORPHICA_v2_d75_100cases.txt",
                    "200_d50": f"{LOG}/plan_ai_stress_probe/n200_d50/MORPHICA_v2_d50_100cases.txt",
                    "200_d100": f"{LOG}/plan_ai_stress_probe/n200_d100/MORPHICA_v2_d100_100cases.txt",
                    "200_d150": f"{LOG}/plan_ai_stress_probe/n200_d150/MORPHICA_v2_d150_100cases.txt",
                },
            },
            "FullX_50case": {
                "description": "Full Plan X 50-case (Plan AE baseline)",
                "paper_label": "MORPHICA-Full",
                "case_count": 50,
                "paths": {
                    "100_d25": f"{LOG}/plan_ae_baseline_lock/n100/MORPHICA_v2_d25_50cases.txt",
                    "100_d50": f"{LOG}/plan_ae_baseline_lock/n100/MORPHICA_v2_d50_50cases.txt",
                    "100_d75": f"{LOG}/plan_ae_baseline_lock/n100/MORPHICA_v2_d75_50cases.txt",
                    "200_d50": f"{LOG}/plan_ae_baseline_lock/n200/MORPHICA_v2_d50_50cases.txt",
                    "200_d100": f"{LOG}/plan_ae_baseline_lock/n200/MORPHICA_v2_d100_50cases.txt",
                    "200_d150": f"{LOG}/plan_ae_baseline_lock/n200/MORPHICA_v2_d150_50cases.txt",
                },
            },
            "BEARER05_50case": {
                "description": "BEARER0.5 sweep 50-case (旧基线)",
                "paper_label": "BEARER0.5",
                "case_count": 50,
                "paths": {
                    "100_d25": f"{LOG}/algorithm_sweep/N100/MORPHICA_v2_BEARER0.5/MORPHICA_v2_d25_50cases.txt",
                    "100_d50": f"{LOG}/algorithm_sweep/N100/MORPHICA_v2_BEARER0.5/MORPHICA_v2_d50_50cases.txt",
                    "100_d75": f"{LOG}/algorithm_sweep/N100/MORPHICA_v2_BEARER0.5/MORPHICA_v2_d75_50cases.txt",
                },
                "note": "N=200 BEARER0.5 data sourced from exploration doc (50-case, not auto-parsed)",
            },
            "PreContact_50case": {
                "description": "Pre-contact filopodia 50-case (集中式上界)",
                "paper_label": "MORPHICA+pre-contact",
                "case_count": 50,
                "paths": {
                    "200_d50": f"{LOG}/plan_precontact/n200_d50/MORPHICA_v2_d50_50cases.txt",
                    "200_d100": f"{LOG}/plan_precontact/n200_d100/MORPHICA_v2_d100_50cases.txt",
                    "200_d150": f"{LOG}/plan_precontact/n200_d150/MORPHICA_v2_d150_50cases.txt",
                },
            },
            "BE_wound_scout": {
                "description": "BE damage-memory wound-front scout 100-case",
                "paper_label": "BE wound-scout",
                "case_count": 100,
                "paths": {
                    "100_d25": f"{LOG}/be_stress_100case/n100_d25/MORPHICA_v2_d25_100cases.txt",
                    "100_d50": f"{LOG}/be_stress_100case/n100_d50/MORPHICA_v2_d50_100cases.txt",
                    "100_d75": f"{LOG}/be_stress_100case/n100_d75/MORPHICA_v2_d75_100cases.txt",
                    "200_d50": f"{LOG}/be_stress_100case/n200_d50/MORPHICA_v2_d50_100cases.txt",
                    "200_d100": f"{LOG}/be_stress_100case/n200_d100/MORPHICA_v2_d100_100cases.txt",
                    "200_d150": f"{LOG}/be_stress_100case/n200_d150/MORPHICA_v2_d150_100cases.txt",
                },
            },
            "AnchorOnly_50case": {
                "description": "Ablation: anchor-only (去 scout + neighbor)",
                "paper_label": "anchor-only",
                "case_count": 50,
                "paths": {
                    "200_d50": f"{LOG}/plan_ah_anchor_only/n200_d50/MORPHICA_v2_d50_50cases.txt",
                    "200_d100": f"{LOG}/plan_ah_anchor_only/n200_d100/MORPHICA_v2_d100_50cases.txt",
                    "200_d150": f"{LOG}/plan_ah_anchor_only/n200_d150/MORPHICA_v2_d150_50cases.txt",
                },
            },
            "NoBearer_50case": {
                "description": "Ablation: no-bearer K=0",
                "paper_label": "no-bearer",
                "case_count": 50,
                "paths": {
                    "200_d50": f"{LOG}/plan_nobearer/n200_d50/MORPHICA_v2_d50_50cases.txt",
                    "200_d100": f"{LOG}/plan_nobearer/n200_d100/MORPHICA_v2_d100_50cases.txt",
                    "200_d150": f"{LOG}/plan_nobearer/n200_d150/MORPHICA_v2_d150_50cases.txt",
                },
            },
            "MultiBearer_50case": {
                "description": "Ablation: multi-bearer",
                "paper_label": "multi-bearer",
                "case_count": 50,
                "paths": {
                    "200_d50": f"{LOG}/plan_multibearer/n200_d50/MORPHICA_v2_d50_50cases.txt",
                    "200_d100": f"{LOG}/plan_multibearer/n200_d100/MORPHICA_v2_d100_50cases.txt",
                    "200_d150": f"{LOG}/plan_multibearer/n200_d150/MORPHICA_v2_d150_50cases.txt",
                },
            },
        },
    }

# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def build_table1(datasets):
    """Table 1: 6-cell mean/P50/P90/P95/max (A+S vs FullX vs BEARER0.5)."""
    rows = []
    for ds_key in ["A+S_100case", "FullX_100case", "BEARER05_50case"]:
        ds = datasets.get(ds_key)
        if not ds: continue
        for cell_key, path in sorted(ds["paths"].items()):
            cases, steps = parse_log(path)
            if steps is None: continue
            s = stats(steps)
            parts = cell_key.split("_")
            N, d = parts[0], parts[1]
            rows.append({
                "Algorithm": ds["paper_label"],
                "Cases": ds["case_count"],
                "N": N, "d": d,
                "Mean": f"{s['mean']:.2f}",
                "P50": str(s['P50']), "P90": str(s['P90']),
                "P95": str(s['P95']), "max": str(s['max']),
            })
    return rows, ["Algorithm", "Cases", "N", "d", "Mean", "P50", "P90", "P95", "max"]

def build_table2(datasets):
    """Table 2: A+S vs Full X paired delta (matched seeds, 100-case)."""
    ds_a = datasets.get("A+S_100case")
    ds_f = datasets.get("FullX_100case")
    if not ds_a or not ds_f: return [], []
    rows = []
    common_cells = set(ds_a["paths"].keys()) & set(ds_f["paths"].keys())
    for cell_key in sorted(common_cells):
        a_cases, a_steps = parse_log(ds_a["paths"][cell_key])
        f_cases, f_steps = parse_log(ds_f["paths"][cell_key])
        if a_steps is None or f_steps is None: continue
        a_d = {c: s for c, s in zip(a_cases, a_steps)}
        f_d = {c: s for c, s in zip(f_cases, f_steps)}
        common = sorted(set(a_d) & set(f_d))
        if not common: continue
        a_m = [a_d[c] for c in common]
        f_m = [f_d[c] for c in common]
        pa = paired_stats(a_m, f_m)
        parts = cell_key.split("_")
        mean_a = np.mean(a_m); mean_f = np.mean(f_m)
        delta_pct = (mean_a - mean_f) / mean_f * 100
        rows.append({
            "N": parts[0], "d": parts[1],
            "n_paired": str(len(common)),
            "A+S Mean": f"{mean_a:.2f}", "Full X Mean": f"{mean_f:.2f}",
            "Δ mean": f"{pa['mean_delta']:+.2f}",
            "Δ %": f"{delta_pct:+.1f}%",
            "Win": str(pa['wins']), "Tie": str(pa['ties']), "Loss": str(pa['losses']),
            "95% CI": f"[{pa['ci_lo']:+.2f}, {pa['ci_hi']:+.2f}]",
        })
    return rows, ["N", "d", "n_paired", "A+S Mean", "Full X Mean", "Δ mean", "Δ %", "Win", "Tie", "Loss", "95% CI"]

def build_table3(datasets):
    """Table 3: pre-contact upper bound vs A+S vs BEARER0.5 (N=200 50-case)."""
    ds_pc = datasets.get("PreContact_50case")
    ds_as = datasets.get("A+S_100case")
    ds_b5 = datasets.get("BEARER05_50case")
    rows = []
    for cell_key in ["200_d50", "200_d100", "200_d150"]:
        row = {}
        parts = cell_key.split("_")
        row["N"] = parts[0]; row["d"] = parts[1]
        for label, ds in [("Pre-contact", ds_pc), ("A+S", ds_as), ("BEARER0.5", ds_b5)]:
            if ds and cell_key in ds["paths"]:
                _, steps = parse_log(ds["paths"][cell_key])
                if steps:
                    row[label] = f"{np.mean(steps):.2f} ({len(steps)}c)"
                else:
                    row[label] = "—"
            else:
                row[label] = "—"
        rows.append(row)
    return rows, ["N", "d", "Pre-contact", "A+S", "BEARER0.5"]

def build_table4(datasets):
    """Table 4: Key ablations (anchor-only, A+S, no-bearer, multi-bearer) N=200 50-case."""
    ablations = ["FullX_50case", "A+S_100case", "AnchorOnly_50case", "NoBearer_50case", "MultiBearer_50case"]
    rows = []
    for cell_key in ["200_d50", "200_d100", "200_d150"]:
        row = {}
        parts = cell_key.split("_")
        row["N"] = parts[0]; row["d"] = parts[1]
        for ab_key in ablations:
            ds = datasets.get(ab_key)
            label = ds["paper_label"] if ds else ab_key
            if ds and cell_key in ds["paths"]:
                _, steps = parse_log(ds["paths"][cell_key])
                if steps:
                    row[label] = f"{np.mean(steps):.2f}"
                else:
                    row[label] = "—"
            else:
                row[label] = "—"
        rows.append(row)
    headers = ["N", "d"] + [datasets.get(k, {}).get("paper_label", k) for k in ablations]
    return rows, headers

def build_table5(datasets):
    """Table 5: Negative results summary (from exploration doc, manual)."""
    return [
        {"Direction": "Blind spiral (AX)", "N200 d100": "+7.0%", "Verdict": "退化"},
        {"Direction": "Stall pulse (AY)", "N200 d150": "+2.5%", "Verdict": "退化"},
        {"Direction": "Shell rendezvous (BA)", "N100 d75": "+7.0%", "Verdict": "退化"},
        {"Direction": "Damage direction (BE)", "N200 d100": "±10%", "Verdict": "中性"},
        {"Direction": "Confidence-gated (BI)", "N200 d100": "+3.1%", "Verdict": "中性"},
        {"Direction": "Count urgency (BJ)", "N200 d100": "+0.8%", "Verdict": "中性"},
        {"Direction": "CR contact seal (BC)", "N200 d100": "−2.3%", "Verdict": "d150退化 +1.8%"},
        {"Direction": "Edge-age seal (BP)", "N200 d100": "−2.2%", "Verdict": "P95不降"},
    ], ["Direction", "N200 d100", "Verdict"]

def build_table6():
    """Table 6: Runtime / complexity (from BN §50.6)."""
    return [
        {"Algorithm": "A+S", "Complexity": "O(N log N)", "Per 50 cases (N=200)": "~3-5 min", "O(N²)": "No"},
        {"Algorithm": "Full Plan X", "Complexity": "O(N·deg) ≈ O(N)", "Per 50 cases (N=200)": "~3-5 min", "O(N²)": "No"},
        {"Algorithm": "BEARER0.5", "Complexity": "O(N·deg) ≈ O(N)", "Per 50 cases (N=200)": "~2-3 min", "O(N²)": "No"},
        {"Algorithm": "Pre-contact", "Complexity": "O(N²)", "Per 50 cases (N=200)": "slower", "O(N²)": "Yes — not default"},
        {"Algorithm": "Damage-memory", "Complexity": "O(N·deg) + O(N) mem", "Per 50 cases (N=200)": "~2-3 min", "O(N²)": "No"},
    ], ["Algorithm", "Complexity", "Per 50 cases (N=200)", "O(N²)"]

def build_table7():
    """Table 7: Tail taxonomy (from AM §37, BO §51)."""
    return [
        {"Type": "I: Multi-fragment merge", "Typical d": "d25/d50", "Root cause": "Fragments migrate independently, merge order is random", "CR-only fix": "None — blind to external fragments"},
        {"Type": "II: Contact oscillation", "Typical d": "d75/d100", "Root cause": "New contact interface tears under A+S separation", "CR-only fix": "Edge-age seal (BP) — partial, −2.2%"},
        {"Type": "III: Slow approach", "Typical d": "d150", "Root cause": "Anchor-parallel motion, closing velocity ≈ 0", "CR-only fix": "None — pre-contact sensing needed"},
        {"Special": "Ring-center", "Typical d": "any", "Root cause": "Rotational symmetry, gap_gate ≈ 0 everywhere", "CR-only fix": "Fundamental limit"},
    ], ["Type", "Typical d", "Root cause", "CR-only fix"]

def build_table8():
    """Table 8: Final commands and log paths."""
    return [
        {"What": "A+S 100-case", "Command": "Experiment_Statistic.py --algorithm MORPHICA_v2 --destroyed-values <d> --max-cases 100 --num-agents <N> --output-dir ...", "Log": "Logs/plan_ap_as_stress/"},
        {"What": "Full X N=200 100c", "Command": "Experiment_Statistic.py --algorithm MORPHICA_v2 --destroyed-values 50 100 150 --max-cases 100 --num-agents 200 --output-dir ...", "Log": "Logs/plan_ai_stress_probe/"},
        {"What": "Full X N=100 100c", "Command": "Experiment_Statistic.py --algorithm MORPHICA_v2 --destroyed-values 25 50 75 --max-cases 100 --num-agents 100 --output-dir ...", "Log": "Logs/plan_bm_full_x_n100/"},
        {"What": "BEARER0.5 sweep", "Command": "Experiment_Statistic.py --algorithm MORPHICA_v2 --destroyed-values <d> --max-cases 50 --num-agents <N> ...", "Log": "Logs/algorithm_sweep/"},
        {"What": "Pre-contact", "Command": "(modified MORPHICA_v2 with global nearest-subnet scan)", "Log": "Logs/plan_precontact/"},
        {"What": "BE 100-case", "Command": "Experiment_Statistic.py --algorithm MORPHICA_v2 ... (damage-memory branch)", "Log": "Logs/be_stress_100case/"},
        {"What": "BP edge-age seal", "Command": "(modified MORPHICA_v2 with edge-age seal F_neighbor)", "Log": "Logs/plan_bp_edge_seal/"},
    ], ["What", "Command", "Log"]

# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def to_markdown(rows, headers):
    if not rows: return "*No data.*\n"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for r in rows:
        vals = [str(r.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

def to_csv(rows, headers):
    import csv, io
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=headers, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TABLE_BUILDERS = [
    ("Table_1_main_performance", "Table 1: 6-cell performance", build_table1),
    ("Table_2_paired_delta", "Table 2: A+S vs Full X paired delta", build_table2),
    ("Table_3_precontact_upper", "Table 3: Pre-contact upper bound", build_table3),
    ("Table_4_ablations", "Table 4: Key ablations", build_table4),
    ("Table_5_negative_results", "Table 5: Negative results", build_table5),
    ("Table_6_runtime", "Table 6: Runtime / complexity", build_table6),
    ("Table_7_tail_taxonomy", "Table 7: Tail taxonomy", build_table7),
    ("Table_8_commands", "Table 8: Final commands & logs", build_table8),
]

def main():
    parser = argparse.ArgumentParser(description="MORPHICA v2 table generator")
    parser.add_argument("--md", type=str, help="Write Markdown tables to directory")
    parser.add_argument("--csv", type=str, help="Write CSV tables to directory")
    parser.add_argument("--all", type=str, help="Write both .md and .csv to directory")
    args = parser.parse_args()

    manifest = load_manifest()
    datasets = manifest.get("datasets", {})

    md_dir = args.md or (args.all if args.all else None)
    csv_dir = args.csv or (args.all if args.all else None)

    if not md_dir and not csv_dir:
        md_dir = None  # print to stdout

    for filename, title, builder in TABLE_BUILDERS:
        if builder in (build_table6, build_table7, build_table8):
            rows, headers = builder()
        else:
            rows, headers = builder(datasets)

        md_text = f"## {title}\n\n" + to_markdown(rows, headers)

        if md_dir:
            os.makedirs(md_dir, exist_ok=True)
            with open(os.path.join(md_dir, f"{filename}.md"), "w") as f:
                f.write(md_text)
            print(f"  Wrote {filename}.md")
        else:
            print(md_text)

        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            with open(os.path.join(csv_dir, f"{filename}.csv"), "w") as f:
                f.write(to_csv(rows, headers))
            print(f"  Wrote {filename}.csv")

    # Also write combined doc
    if md_dir:
        combined = []
        combined.append("# MORPHICA v2 Final Tables\n")
        combined.append(f"> Auto-generated from {len(datasets)} datasets\n")
        for filename, title, builder in TABLE_BUILDERS:
            combined.append(f"## {title}\n")
            if builder in (build_table6, build_table7, build_table8):
                rows, headers = builder()
            else:
                rows, headers = builder(datasets)
            combined.append(to_markdown(rows, headers))
        combined_path = os.path.join(md_dir, "MORPHICA_final_tables.md")
        with open(combined_path, "w") as f:
            f.write("\n".join(combined))
        print(f"  Wrote combined MORPHICA_final_tables.md")

    print(f"\nDone. {len(TABLE_BUILDERS)} tables generated.")

if __name__ == "__main__":
    main()
