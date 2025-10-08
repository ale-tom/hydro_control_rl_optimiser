from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, wilcoxon

from src.hydro_control_rl_optimiser.cooling_control import PumpingRegime


def format_global_search_report(loaded: Dict[str, Any]) -> str:
    """Report results from the global search pass.
    """

    def _fmt_kwh(x: float) -> str:
        return f"{x:,.2f}"

    def _fmt_sw(x: float) -> str:
        return f"{x:.1f}"

    def _mean_sd(a: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(a, dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        if arr.size == 1:
            return float(arr[0]), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    gs = loaded.get("global_summary", {}) or {}
    outer = loaded.get("outer_eval", {}) or {}
    rl_test_seeds = int(outer.get("rl_test_seeds", 1))

    # Globally selected params
    best_heur = tuple(
        gs.get("best_heuristic_params", loaded.get("best_heuristic_params", (None, None)))
    )
    best_rl = tuple(gs.get("best_rl_params", loaded.get("best_rl_params", (None, None))))

    # Expects keys like gs["validation"]["heuristic"] = {"viol": 0, "unmet": 0, "sw": 16.1, "E": 122675.80}
    val_heur = (gs.get("validation", {}) or {}).get("heuristic", {})
    val_rl = (gs.get("validation", {}) or {}).get("rl", {})

    # If not present, fall back to mean inner cost for the chosen params (no viol/unmet detail)
    heur_val_costs = loaded.get("heur_val_costs", {}) or {}
    rl_val_costs = loaded.get("rl_val_costs", {}) or {}

    # Helper to grab the cost list for chosen params
    def _find_costs(cost_dict: Dict[tuple, List[float]], key_tuple: tuple) -> float | None:
        # Direct key first
        if key_tuple in cost_dict:
            vals = cost_dict[key_tuple]
            return float(np.mean(vals)) if len(vals) else None
        # Try numeric-coerced (in case of JSON float vs int quirks)
        for k, v in cost_dict.items():
            if len(k) == len(key_tuple) and all(float(a) == float(b) for a, b in zip(k, key_tuple)):
                return float(np.mean(v)) if len(v) else None
        return None

    heur_val_fallback = (
        _find_costs(heur_val_costs, best_heur) if all(x is not None for x in best_heur) else None
    )
    rl_val_fallback = (
        _find_costs(rl_val_costs, best_rl) if all(x is not None for x in best_rl) else None
    )

    # Per-fold outer test metrics
    # Prefer detailed arrays from global_summary (which include rates); otherwise fall back to the basic ones saved.
    heur_block = gs.get("heuristic", {}) or {}
    rl_block = gs.get("rl", {}) or {}

    heur_E_folds = heur_block.get("energy_per_fold", loaded.get("heur_test_metrics", []))
    heur_S_folds = heur_block.get("switches_per_fold", None)

    # If we fell back to heur_test_metrics, unpack columns
    if (
        isinstance(heur_E_folds, list)
        and len(heur_E_folds)
        and isinstance(heur_E_folds[0], (list, tuple))
    ):
        # Loaded from heur_test_metrics [(E, S), ...]
        _energies = [float(e) for e, _ in heur_E_folds]
        _switches = [int(s) for _, s in heur_E_folds]
        heur_E_folds, heur_S_folds = _energies, _switches

    if heur_S_folds is None:
        heur_S_folds = heur_block.get("switches_per_fold", [])

    heur_V_flags = heur_block.get("violation_per_fold", [])
    heur_U_flags = heur_block.get("unmet_per_fold", [])  # optional

    rl_E_folds = rl_block.get("energy_mean_per_fold", loaded.get("rl_test_metrics", []))
    rl_S_folds = rl_block.get("switch_mean_per_fold", None)

    if (
        isinstance(rl_E_folds, list)
        and len(rl_E_folds)
        and isinstance(rl_E_folds[0], (list, tuple))
    ):
        # Loaded from rl_test_metrics [(E, S), ...] – no multi-seed averaging info
        _energies = [float(e) for e, _ in rl_E_folds]
        _switches = [float(s) for _, s in rl_E_folds]
        rl_E_folds, rl_S_folds = _energies, _switches

    if rl_S_folds is None:
        rl_S_folds = rl_block.get("switch_mean_per_fold", [])

    rl_V_rates = rl_block.get("violation_rate_per_fold", [])
    rl_U_rates = rl_block.get("unmet_rate_per_fold", [])  # optional

    # Aggregate stats
    heur_E_mean, heur_E_sd = _mean_sd(heur_E_folds)
    heur_S_mean, heur_S_sd = _mean_sd(heur_S_folds)
    heur_V_rate = (
        float(np.mean(np.asarray(heur_V_flags, dtype=float))) if len(heur_V_flags) else 0.0
    )
    heur_U_rate = (
        float(np.mean(np.asarray(heur_U_flags, dtype=float))) if len(heur_U_flags) else 0.0
    )

    rl_E_mean, rl_E_sd = _mean_sd(rl_E_folds)
    rl_S_mean, rl_S_sd = _mean_sd(rl_S_folds)
    rl_V_rate_mean = float(np.mean(np.asarray(rl_V_rates, dtype=float))) if len(rl_V_rates) else 0.0
    rl_U_rate_mean = float(np.mean(np.asarray(rl_U_rates, dtype=float))) if len(rl_U_rates) else 0.0

    # Build lines
    lines: List[str] = []

    # Header lines with validation summaries (or fallbacks)
    if val_heur:
        lines.append(
            f"Global best heuristic (by validation): pg={best_heur[0]}, sp={best_heur[1]} "
            f"(val summary: viol={val_heur.get('viol', 'NA')}, unmet={val_heur.get('unmet', 'NA')}, "
            f"sw={val_heur.get('sw', 'NA')}, E={_fmt_kwh(float(val_heur.get('E', 0.0)))})"
        )
    else:
        tail = (
            f"mean val cost={_fmt_kwh(heur_val_fallback)}"
            if heur_val_fallback is not None
            else "val summary: NA"
        )
        lines.append(
            f"Global best heuristic (by validation): pg={best_heur[0]}, sp={best_heur[1]} ({tail})"
        )

    if val_rl:
        lines.append(
            f"Global best RL (by validation): sp_rl={best_rl[0]}, unmet={best_rl[1]} "
            f"(val summary: viol={val_rl.get('viol', 'NA')}, unmet={val_rl.get('unmet', 'NA')}, "
            f"sw={val_rl.get('sw', 'NA')}, E={_fmt_kwh(float(val_rl.get('E', 0.0)))})"
        )
    else:
        tail = (
            f"mean val cost={_fmt_kwh(rl_val_fallback)}"
            if rl_val_fallback is not None
            else "val summary: NA"
        )
        lines.append(
            f"Global best RL (by validation): sp_rl={best_rl[0]}, unmet={best_rl[1]} ({tail})"
        )

    # Per-fold lines
    n_folds = min(len(heur_E_folds), len(heur_S_folds), len(rl_E_folds), len(rl_S_folds))
    for i in range(n_folds):
        hE = _fmt_kwh(float(heur_E_folds[i]))
        hS = int(heur_S_folds[i])

        hV = None
        if i < len(heur_V_flags):
            hV = "Y" if bool(heur_V_flags[i]) else "N"
        else:
            hV = "N"

        # If we tracked unmet per fold for heuristic (often 0), show; else default N
        hU = None
        if i < len(heur_U_flags):
            hU = "Y" if bool(heur_U_flags[i]) else "N"
        else:
            hU = "N"

        rE = _fmt_kwh(float(rl_E_folds[i]))
        rS = _fmt_sw(float(rl_S_folds[i]))
        rV = f"{float(rl_V_rates[i]):.2f}" if i < len(rl_V_rates) else "NA"
        rU = f"{float(rl_U_rates[i]):.2f}" if i < len(rl_U_rates) else "NA"

        lines.append(
            f"Fold {i+1}: Heur E={hE} kWh, S={hS}, Viol={hV}, Unmet={hU}; "
            f"RL E={rE} kWh, S={rS}, ViolRate={rV}, UnmetRate={rU} (over {rl_test_seeds} seeds)"
        )

    # Footer aggregates
    lines.append("")
    lines.append("=== Global params on outer test folds (masked, with unmet) ===")
    lines.append(f"Heuristic energy: {_fmt_kwh(heur_E_mean)} ± {_fmt_kwh(heur_E_sd)} kWh")
    lines.append(f"Heuristic switches: {_fmt_sw(heur_S_mean)} ± {_fmt_sw(heur_S_sd)}")
    lines.append(f"Heuristic violation rate: {heur_V_rate:.2f}")
    lines.append(f"Heuristic unmet rate: {heur_U_rate:.2f}")
    lines.append(f"RL energy: {_fmt_kwh(rl_E_mean)} ± {_fmt_kwh(rl_E_sd)} kWh")
    lines.append(f"RL switches: {_fmt_sw(rl_S_mean)} ± {_fmt_sw(rl_S_sd)}")
    lines.append(f"RL violation rate: {rl_V_rate_mean:.2f}")
    lines.append(f"RL unmet rate: {rl_U_rate_mean:.2f}")

    return "\n".join(lines)


def to_tuple_actions(actions: Iterable[Iterable[int]]) -> List[Tuple[int, ...]]:
    """Return actions as a list of tuples, one tuple per minute: (pumps_station_0, ..., pumps_station_{k-1})."""
    return [tuple(map(int, a)) for a in actions]


def unique_sorted_regimes(actions_list: List[List[Tuple[int, ...]]]) -> List[List[int]]:
    """Collect the sorted unique pump counts (regimes) that appear for each station across multiple action sets."""
    max_len = max(len(a[0]) for a in actions_list if len(a) > 0)
    regimes_per_station: List[set] = [set() for _ in range(max_len)]
    for actions in actions_list:
        for a in actions:
            for i, v in enumerate(a):
                regimes_per_station[i].add(v)
    return [sorted(list(s)) for s in regimes_per_station]


def compute_specific_energy_kwh_per_m3(regimes: List["PumpingRegime"]) -> List[Dict[int, float]]:
    """Return specific energy tables per station: dict[pump_count] -> kWh/m³."""
    tables: List[Dict[int, float]] = []
    for reg in regimes:
        table: Dict[int, float] = {}
        for pumps, flow in reg.flow_rates.items():
            power = float(reg.power_draws.get(pumps, 0.0))
            spec = power / flow if flow > 0 else np.inf
            table[pumps] = float(spec)
        tables.append(table)
    return tables


def flows_for_action(regimes: List["PumpingRegime"], action: Tuple[int, ...]) -> List[float]:
    """Return list of flows (m^3/h) per station for a given action."""
    return [reg.flow(p) for reg, p in zip(regimes, action)]


def power_for_action(regimes: List["PumpingRegime"], action: Tuple[int, ...]) -> List[float]:
    """Return list of power draws (kW) per station for a given action."""
    return [reg.power(p) for reg, p in zip(regimes, action)]


def delivered_flow_for_action(regimes: List["PumpingRegime"], action: Tuple[int, ...]) -> float:
    """Return delivered flow to demand (m^3/h): flow of the last station in the action vector."""
    return regimes[-1].flow(action[-1])


def regime_usage(
    actions: List[Tuple[int, ...]], station_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute time fraction spent in each regime per station."""
    if not actions:
        return pd.DataFrame(columns=["station", "regime", "minutes", "fraction"])
    n = len(actions)
    n_stations = len(actions[0])
    names = station_names or [f"S{i+1}" for i in range(n_stations)]

    rows = []
    for i in range(n_stations):
        counts: Dict[int, int] = {}
        for a in actions:
            counts[a[i]] = counts.get(a[i], 0) + 1
        for regime, c in sorted(counts.items()):
            rows.append({"station": names[i], "regime": regime, "minutes": c, "fraction": c / n})
    return pd.DataFrame(rows)


def transition_matrices(actions: List[Tuple[int, ...]]) -> List[pd.DataFrame]:
    """Compute transition count matrices per station: from regime r to r'."""
    if not actions:
        return []
    n_stations = len(actions[0])
    mats: List[pd.DataFrame] = []
    for i in range(n_stations):
        series = [a[i] for a in actions]
        uniq = sorted(set(series))
        idx = {r: k for k, r in enumerate(uniq)}
        mat = np.zeros((len(uniq), len(uniq)), dtype=int)
        for x, y in zip(series[:-1], series[1:]):
            mat[idx[x], idx[y]] += 1
        mats.append(pd.DataFrame(mat, index=uniq, columns=uniq))
    return mats


def dwell_times(actions: List[Tuple[int, ...]]) -> Dict[int, List[int]]:
    """Return dwell times (minutes spent before changing) aggregated across all stations. Dict[station_idx] -> list of durations."""
    if not actions:
        return {}
    n_stations = len(actions[0])
    per_station: Dict[int, List[int]] = {i: [] for i in range(n_stations)}
    for i in range(n_stations):
        prev = actions[0][i]
        run = 1
        for a in actions[1:]:
            if a[i] == prev:
                run += 1
            else:
                per_station[i].append(run)
                prev = a[i]
                run = 1
        per_station[i].append(run)
    return per_station


def delivered_minus_demand_series(
    regimes: List["PumpingRegime"],
    actions: List[Tuple[int, ...]],
    demand_series_m3_per_h: List[float],
) -> np.ndarray:
    """Return per-step (delivered - demand) in m^3/h."""
    delivered = [delivered_flow_for_action(regimes, a) for a in actions]
    dmd = np.asarray(demand_series_m3_per_h[: len(delivered)], dtype=float)
    return np.asarray(delivered, dtype=float) - dmd


def set_pretty():
    """Set a clean visual theme."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_regime_usage_bars(
    usage_heur: pd.DataFrame, usage_rl: pd.DataFrame, title: str = "Regime usage (time fraction)"
) -> None:
    """Side-by-side bars of time fraction in each regime, per station."""
    df_h = usage_heur.copy()
    df_h["model"] = "Heuristic"
    df_r = usage_rl.copy()
    df_r["model"] = "RL"
    df = pd.concat([df_h, df_r], ignore_index=True)

    stations = df["station"].unique().tolist()
    ncols = min(3, len(stations))
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ax, st in zip(axes.flatten(), stations):
        sub = df[df["station"] == st].copy()
        sub.sort_values(["regime", "model"], inplace=True)
        sns.barplot(data=sub, x="regime", y="fraction", hue="model", ax=ax)
        ax.set_title(f"{st}")
        ax.set_xlabel("Pump count")
        ax.set_ylabel("Time fraction")
        ax.legend(loc="upper right")
    # hide empty axes
    for j in range(len(stations), nrows * ncols):
        axes.flatten()[j].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_transition_heatmaps(
    mats_heur: List[pd.DataFrame],
    mats_rl: List[pd.DataFrame],
    station_names: Optional[List[str]] = None,
) -> None:
    """Compare transition patterns (from regime to regime) per station."""
    n = len(mats_heur)
    names = station_names or [f"S{i+1}" for i in range(n)]
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n))
    for i in range(n):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        sns.heatmap(mats_heur[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1)
        ax1.set_title(f"{names[i]}: Heuristic transitions")
        ax1.set_xlabel("Next regime")
        ax1.set_ylabel("Prev regime")
        sns.heatmap(mats_rl[i], annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax2)
        ax2.set_title(f"{names[i]}: RL transitions")
        ax2.set_xlabel("Next regime")
        ax2.set_ylabel("Prev regime")
    fig.tight_layout()
    plt.show()


def plot_dwell_violin(
    dwell_heur: Dict[int, List[int]],
    dwell_rl: Dict[int, List[int]],
    station_names: Optional[List[str]] = None,
) -> None:
    """Violin plots of dwell times (minutes) per station, Heuristic vs RL."""
    n = len(dwell_heur)
    names = station_names or [f"S{i+1}" for i in range(n)]
    rows = []
    for i in range(n):
        for v in dwell_heur.get(i, []):
            rows.append({"station": names[i], "dwell_min": v, "model": "Heuristic"})
        for v in dwell_rl.get(i, []):
            rows.append({"station": names[i], "dwell_min": v, "model": "RL"})
    df = pd.DataFrame(rows)
    g = sns.catplot(
        data=df,
        x="station",
        y="dwell_min",
        hue="model",
        kind="violin",
        split=True,
        cut=0,
        inner="quart",
        height=5,
        aspect=1.6,
    )
    g.set_axis_labels("Station", "Dwell time (min)")
    g.fig.suptitle("Dwell time distributions (Heuristic vs RL)", y=1.02)
    plt.show()


def plot_timelines_raster(
    actions_heur: List[Tuple[int, ...]],
    actions_rl: List[Tuple[int, ...]],
    station_names: Optional[List[str]] = None,
) -> None:
    """Raster of pump counts over time per station, for Heuristic and RL."""
    n_stations = len(actions_heur[0])
    names = station_names or [f"S{i+1}" for i in range(n_stations)]

    def _to_array(actions: List[Tuple[int, ...]]) -> np.ndarray:
        arr = np.array(actions, dtype=int)
        return arr.T  # rows=stations, cols=time

    arr_h = _to_array(actions_heur)
    arr_r = _to_array(actions_rl)
    T = arr_h.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    im0 = axes[0].imshow(arr_h, aspect="auto", interpolation="nearest")
    axes[0].set_title("Heuristic: Pump counts (rows=stations, cols=time)")
    axes[0].set_yticks(range(n_stations))
    axes[0].set_yticklabels(names)
    fig.colorbar(im0, ax=axes[0], orientation="vertical", fraction=0.015)

    im1 = axes[1].imshow(arr_r, aspect="auto", interpolation="nearest")
    axes[1].set_title("RL: pump counts (rows=stations, cols=time)")
    axes[1].set_yticks(range(n_stations))
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel("Time (minutes)")
    fig.colorbar(im1, ax=axes[1], orientation="vertical", fraction=0.015)

    fig.suptitle("Pump regime timeline (48 hours @ 1-min resolution)", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_overdelivery_violin(
    dmd: List[float], delivered_diff_heur: np.ndarray, delivered_diff_rl: np.ndarray
) -> None:
    """Violin plot of (delivered - demand) across minutes, Heuristic vs RL."""
    df = pd.DataFrame(
        {
            "diff": np.concatenate([delivered_diff_heur, delivered_diff_rl]),
            "model": ["Heuristic"] * len(delivered_diff_heur) + ["RL"] * len(delivered_diff_rl),
        }
    )
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="model", y="diff", cut=0, inner="quart")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("Delivered - Demand (m^3/h)")
    plt.title("Distribution of delivered minus demand")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.show()


def build_and_plot_all(
    regimes: List["PumpingRegime"],
    actions_heur: List[Iterable[int]],
    actions_rl: List[Iterable[int]],
    demand_series_m3_per_h: List[float],
    station_names: Optional[List[str]] = None,
    timestep_hours: float = 1.0 / 60.0,
) -> None:
    """End-to-end construction of diagnostics and plots comparing Heuristic vs RL."""
    set_pretty()

    # Normalise inputs
    heur = to_tuple_actions(actions_heur)
    rl = to_tuple_actions(actions_rl)
    n_stations = len(heur[0])
    names = station_names or [f"S{i+1}" for i in range(n_stations)]

    # ---- Usage + transitions ----
    usage_h = regime_usage(heur, station_names=names)
    usage_r = regime_usage(rl, station_names=names)
    mats_h = transition_matrices(heur)
    mats_r = transition_matrices(rl)
    plot_regime_usage_bars(usage_h, usage_r, title="Regime usage (time fraction) by station")
    plot_transition_heatmaps(mats_h, mats_r, station_names=names)

    # ---- Dwell times ----
    dwell_h = dwell_times(heur)
    dwell_r = dwell_times(rl)
    plot_dwell_violin(dwell_h, dwell_r, station_names=names)

    # ---- Timelines ----
    plot_timelines_raster(heur, rl, station_names=names)

    # ---- Delivered vs demand ----
    diff_h = delivered_minus_demand_series(regimes, heur, demand_series_m3_per_h)
    diff_r = delivered_minus_demand_series(regimes, rl, demand_series_m3_per_h)
    plot_overdelivery_violin(demand_series_m3_per_h, diff_h, diff_r)


@dataclass
class PairedStats:
    """Container for paired-comparison results."""

    diffs: np.ndarray
    median: float
    ci95: Tuple[float, float]
    wilcoxon_p: float
    sign_wins: int
    sign_losses: int
    sign_p_two_sided: float
    cles: float


def bootstrap_ci_median(
    diffs: np.ndarray, n_boot: int = 20000, seed: int = 17
) -> Tuple[float, float]:
    """Return a 95% bootstrap CI for the median of paired differences."""
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(diffs[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(a: Sequence[float], b: Sequence[float]) -> PairedStats:
    """
    Compute paired statistics for arrays a and b (paired by index).
    Returns diffs = a - b (so negative means a < b).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    med = float(np.median(d))
    ci = bootstrap_ci_median(d)
    # Wilcoxon signed-rank (two-sided), Pratt method includes zeros
    res = wilcoxon(d, zero_method="pratt", alternative="greater", correction=False, mode="exact")
    # Sign test
    wins = int(np.sum(d < 0))  # a < b (RL < Heur)
    losses = int(np.sum(d > 0))
    n = wins + losses
    p_sign = binomtest(wins, n, p=0.5, alternative="greater").pvalue if n > 0 else 1.0
    # Common-language effect size (probability a < b), with ties split
    cles = float((np.sum(d < 0) + 0.5 * np.sum(d == 0)) / d.size)
    return PairedStats(
        diffs=d,
        median=med,
        ci95=ci,
        wilcoxon_p=float(res.pvalue),
        sign_wins=wins,
        sign_losses=losses,
        sign_p_two_sided=float(p_sign),
        cles=cles,
    )


def print_summary(stats: Dict[str, PairedStats]) -> None:
    """Pretty-print the statistical summary for energy and switches."""
    e = stats["energy"]
    s = stats["switches"]
    print("=== Energy (RL - Heuristic) ===")
    print(f"Deltas per fold (kWh): {np.array2string(e.diffs, precision=0, separator=', ')}")
    print(
        f"Median ΔE: {e.median:.0f} kWh   95% CI [{e.ci95[0]:.0f}, {e.ci95[1]:.0f}]   Wilcoxon p={e.wilcoxon_p:.3f}"
    )
    print(
        f"Sign test: wins={e.sign_wins}, losses={e.sign_losses}, one-sided p={e.sign_p_two_sided:.3f}"
    )
    print(f"Common-language effect size (P[RL < Heur]): {e.cles:.2f}")
    print("\n=== Switches (RL - Heuristic) ===")
    print(f"Deltas per fold (count): {np.array2string(s.diffs, precision=0, separator=', ')}")
    print(
        f"Median ΔS: {s.median:.0f}   95% CI [{s.ci95[0]:.0f}, {s.ci95[1]:.0f}]   Wilcoxon p={s.wilcoxon_p:.3f}"
    )
    print(
        f"Sign test: wins={s.sign_wins}, losses={s.sign_losses}, one-sided p={s.sign_p_two_sided:.3f}"
    )
    print(f"Common-language effect size (P[RL < Heur]): {s.cles:.2f}")


def compute_all_stats(df: pd.DataFrame) -> Dict[str, PairedStats]:
    """Compute paired stats for energy and switches using RL - Heuristic per fold."""
    # Align per fold
    df_piv_e = df.pivot(index="fold", columns="model", values="energy_kwh").sort_index()
    df_piv_s = df.pivot(index="fold", columns="model", values="switches").sort_index()
    rl_e = df_piv_e["RL"].to_numpy()
    he_e = df_piv_e["Heuristic"].to_numpy()
    rl_s = df_piv_s["RL"].to_numpy()
    he_s = df_piv_s["Heuristic"].to_numpy()
    energy_stats = paired_stats(rl_e, he_e)
    switch_stats = paired_stats(rl_s, he_s)
    return {"energy": energy_stats, "switches": switch_stats}
