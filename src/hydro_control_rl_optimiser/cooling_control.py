"""
This module contains classes and functions for simulating and controlling a
pumping pipeline system using a proportional heuristic controller.

The core of this module is the `CoolingLoopSimulator` class, which maintains
state for each reservoir and provides methods for stepping the system forward in
time given a set of pump activations and demand values.

"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PumpingRegime:
    """Stores flow and power characteristics for a single pumping station.

    The regime maps an integer number of active pumps to a corresponding
    volumetric flow rate (in m^3/h) and electrical power draw (in kW). The
    mapping is provided via dictionaries loaded from a provided JSON file.
    """

    flow_rates: dict[int, float]
    power_draws: dict[int, float]

    @staticmethod
    def from_dict(data: dict[str, dict[str, float]]) -> "PumpingRegime":
        """Create a `PumpingRegime` from nested dictionaries.

        The input `data` dictionary must contain two keys: "Flow Rate (m3/h)"
        and "Power Draw (kW)", each mapping string integers to floats.
        These values are converted into integer keyed dictionaries for more
        convenient use. If a key is missing in the source data, a default of
        zero is used.
        """
        flow_str_dict = data.get("Flow Rate (m3/h)", {})
        power_str_dict = data.get("Power Draw (kW)", {})
        flow_rates = {int(k): float(v) for k, v in flow_str_dict.items()}
        power_draws = {int(k): float(v) for k, v in power_str_dict.items()}
        return PumpingRegime(flow_rates=flow_rates, power_draws=power_draws)

    def flow(self, pumps: int) -> float:
        """Retrieve the flow rate for a given pump count.

        If the requested number of pumps is not explicitly present in the
        mapping, the nearest available lower pump count is used. This
        conservative fallback prevents overestimating the flow from an
        undefined regime.
        """
        if pumps in self.flow_rates:
            return self.flow_rates[pumps]
        # find the largest key less than pumps
        valid_keys = [k for k in self.flow_rates if k <= pumps]
        return self.flow_rates[max(valid_keys)] if valid_keys else 0.0

    def power(self, pumps: int) -> float:
        """Retrieve the power draw for a given pump count.

        Fallback behaviour mirrors that of :meth:`flow` by using the
        closest lower defined pump count when an exact match is unavailable.
        """
        if pumps in self.power_draws:
            return self.power_draws[pumps]
        valid_keys = [k for k in self.power_draws if k <= pumps]
        return self.power_draws[max(valid_keys)] if valid_keys else 0.0


@dataclass
class CoolingLoopSimulator:
    """Simulate the dynamics of a multi-station pumping pipeline.

    This simulator maintains internal states representing the current volume
    of water stored in each reservoir and provides a step method to advance the state
    forward by one minute given a vector of pump activations and a demand
    value. Instances of this class can be reused to simulate multiple
    episodes by resetting the `tank_volumes` attribute.
    """

    regimes: list[PumpingRegime]
    tank_capacity: float
    timestep_hours: float = 1.0 / 60.0  # one minute timestep in hours
    lower_limit: float = 0.3  # minimum allowed capacity
    upper_limit: float = 0.9  # maximum allowed capacity
    target_fraction: float = 0.6  # nominal desired volume
    tank_volumes: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tank_volumes:
            # initialise volumes at the target fraction if not provided
            self.tank_volumes = [self.target_fraction * self.tank_capacity] * (
                len(self.regimes) + 1
            )

    def reset(self, initial_fractions: Optional[list[float]] = None) -> None:
        """Reset reservoir volumes to given fractions of capacity or to the target.

        This method allows simulation of multiple independent episodes with
        different initial conditions. When ``initial_fractions`` is None,
        all reservoirs are reset to the ``target_fraction`` of capacity.
        """
        n_tanks = len(self.regimes) + 1
        if initial_fractions is None:
            self.tank_volumes = [self.target_fraction * self.tank_capacity] * n_tanks
        else:
            if len(initial_fractions) != n_tanks:
                raise ValueError(
                    f"Expected {n_tanks} initial fractions but got {len(initial_fractions)}"
                )
            self.tank_volumes = [f * self.tank_capacity for f in initial_fractions]

    def step(self, pump_counts: list[int], demand_flow: float) -> tuple[list[float], float]:
        """Advance the system state by one timestep.

        Pump counts specify the number of active pumps at each station
        (a list of integers whose length equals the number of stations). The
        method computes inflows and outflows for each reservoir, updates the
        stored volumes accordingly, and clips them to physical limits. The
        return value includes the updated reservoir volumes and the total energy
        consumption over this timestep.
        """
        n_stations = len(self.regimes)
        if len(pump_counts) != n_stations:
            raise ValueError("pump_counts length must match number of regimes")
        # compute flows and power at each station
        flows = [reg.flow(p) for reg, p in zip(self.regimes, pump_counts)]
        powers = [reg.power(p) for reg, p in zip(self.regimes, pump_counts)]
        # convert flows from m^3/h to m^3 per timestep
        flows_per_step = [f * self.timestep_hours for f in flows]
        demand_per_step = demand_flow * self.timestep_hours
        # update volumes: reservoir 0 receives from unlimited source
        new_volumes = self.tank_volumes.copy()
        outflow0 = flows_per_step[0]
        new_volumes[0] = min(self.tank_capacity, max(0.0, new_volumes[0] - outflow0))
        # intermediate reservoirs
        for i in range(1, n_stations):
            inflow = flows_per_step[i - 1]
            outflow = flows_per_step[i]
            new_volumes[i] = min(self.tank_capacity, max(0.0, new_volumes[i] + inflow - outflow))
        # final reservoir receives inflow from last station and releases demand
        last_inflow = flows_per_step[-1]
        new_volumes[n_stations] = min(
            self.tank_capacity, max(0.0, new_volumes[n_stations] + last_inflow - demand_per_step)
        )
        energy = sum(powers) * self.timestep_hours
        self.tank_volumes = new_volumes
        return new_volumes, energy

    def get_levels(self) -> list[float]:
        """Return current reservoir levels as fractions of total capacity."""
        return [v / self.tank_capacity for v in self.tank_volumes]


def heuristic_controller(
    demand_series: Iterable[float],
    regimes: list[PumpingRegime],
    tank_capacity: float,
    lower_limit: float = 0.3,
    upper_limit: float = 0.9,
    target_fraction: float = 0.6,
    proportional_gain: float = 0.5,
    switching_penalty: float = 10.0,
) -> tuple[list[list[int]], list[float], list[list[float]]]:
    """Generate a pump schedule using a simple proportional control heuristic."""
    n_stations = len(regimes)
    sim = CoolingLoopSimulator(
        regimes=regimes,
        tank_capacity=tank_capacity,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        target_fraction=target_fraction,
    )
    pump_history: list[list[int]] = []
    energy_history: list[float] = []
    level_history: list[list[float]] = []
    previous_pumps: list[int] = [0] * n_stations
    available_pumps: list[list[int]] = [sorted(reg.flow_rates.keys()) for reg in regimes]
    for demand in demand_series:
        levels = sim.get_levels()
        # compute target flows bottom-up
        target_flows: list[float] = [0.0] * n_stations
        downstream_target = demand
        for i in reversed(range(n_stations)):
            downstream_level = levels[i + 1]
            adjustment = proportional_gain * (
                (target_fraction - downstream_level) * tank_capacity / sim.timestep_hours
            )
            target_flow = downstream_target + adjustment
            target_flow = max(0.0, target_flow)
            target_flows[i] = target_flow
            downstream_target = target_flow
        # choose pumps by minimising flow error plus switching penalty
        chosen_pumps: list[int] = []
        for i, reg in enumerate(regimes):
            target = target_flows[i]
            best_cost = float("inf")
            best_pump = 0
            for p in available_pumps[i]:
                flow = reg.flow(p)
                switch_cost = switching_penalty if p != previous_pumps[i] else 0.0
                cost = (flow - target) ** 2 + switch_cost
                if cost < best_cost:
                    best_cost = cost
                    best_pump = p
            chosen_pumps.append(best_pump)
        _, energy = sim.step(chosen_pumps, demand)
        pump_history.append(chosen_pumps)
        energy_history.append(energy)
        level_history.append(sim.get_levels())
        previous_pumps = chosen_pumps
    return pump_history, energy_history, level_history


def load_pumping_regimes(path: str) -> tuple[list[PumpingRegime], dict[int, float]]:
    """Load pumping regimes and demand levels from a JSON file.

    The JSON file must follow the structure used in the supplied
    `pumping_regimes.json`: a top-level mapping with keys corresponding
    to station names and a special "Demand" key providing discrete
    demand flow rates. The function returns a list of `PumpingRegime`
    instances ordered by ascending station index and a dictionary mapping
    discrete demand levels to flow rates.
    """
    with open(path, "r") as f:
        data = json.load(f)
    regimes = []
    demand_dict = {}
    for key, value in data.items():
        if key == "Demand":
            demand_dict = {int(k): float(v) for k, v in value.get("Flow Rate (m3/h)", {}).items()}
        else:
            regimes.append(PumpingRegime.from_dict(value))
    regimes.sort(key=lambda r: r.flow_rates[1])
    return regimes, demand_dict


def assign_demand_levels(
    demand_series: Iterable[float], discrete_levels: dict[int, float]
) -> list[int]:
    """Assign each demand value to the closest discrete demand level index."""
    levels = np.array([v for _, v in sorted(discrete_levels.items())], dtype=float)
    indices = []
    for d in demand_series:
        idx = int(np.abs(levels - d).argmin())
        indices.append(idx)
    return indices


def compute_switch_count(pump_history: list[list[int]]) -> int:
    """Count the total number of pump switching events across all stations."""
    if not pump_history:
        return 0
    n_stations = len(pump_history[0])
    switches = 0
    prev = pump_history[0]
    for curr in pump_history[1:]:
        for i in range(n_stations):
            if curr[i] != prev[i]:
                switches += 1
        prev = curr
    return switches


def count_switches(pump_history: List[List[int]]) -> int:
    """Count total station switch events between consecutive timesteps."""
    if not pump_history:
        return 0
    switches = 0
    prev = pump_history[0]
    for curr in pump_history[1:]:
        switches += sum(int(c != p) for c, p in zip(curr, prev))
        prev = curr
    return switches


def build_action_map_from_regimes(regimes: List[PumpingRegime]) -> List[Tuple[int, ...]]:
    """Return the Cartesian product of available pump counts per station."""
    per_station = [sorted(reg.flow_rates.keys()) for reg in regimes]
    return list(product(*per_station))


def simulate_one_step_levels(
    sim: CoolingLoopSimulator, action: Tuple[int, ...], demand: float
) -> np.ndarray:
    """Return next-step levels (R-2..R-5) after applying `action` at `demand`, without mutating `sim`."""
    sim_tmp = copy.deepcopy(sim)
    sim_tmp.step(list(action), demand)
    return np.asarray(sim_tmp.get_levels()[1:], dtype=float)


def build_safe_action_set(
    sim: CoolingLoopSimulator,
    demand: float,
    action_map: List[Tuple[int, ...]],
    lower_limit: float,
    upper_limit: float,
) -> Tuple[List[int], Optional[int]]:
    """Return (indices_of_safe_actions, index_of_least_unsafe_action)."""
    safe_indices: List[int] = []
    best_idx: Optional[int] = None
    best_violation: float = float("inf")

    for idx, a in enumerate(action_map):
        next_levels = simulate_one_step_levels(sim, a, demand)
        low_viol = np.maximum(0.0, lower_limit - next_levels)
        high_viol = np.maximum(0.0, next_levels - upper_limit)
        viol_mag = float(np.max(np.maximum(low_viol, high_viol)))
        if viol_mag == 0.0:
            safe_indices.append(idx)
        if viol_mag < best_violation:
            best_violation = viol_mag
            best_idx = idx

    return safe_indices, best_idx


def make_violation_info(
    levels_history: Sequence[Sequence[float]],
    lower_limit: float,
    upper_limit: float,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Summarise bound violations for a levels time-series and return a compact info dict.

    The function scans the time-by-reservoir matrix `levels_history` and flags any timestep where
    at least one reservoir is outside [lower_limit, upper_limit] (expanded by ±tol). It also
    reports which timesteps violated and how many violations each reservoir had.
    """
    arr = np.asarray(levels_history, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    lower = lower_limit - tol
    upper = upper_limit + tol

    viol_mask = (arr < lower) | (arr > upper)  # shape: (T, K)
    violation_steps = np.where(viol_mask.any(axis=1))[0].tolist()
    per_tank_counts = viol_mask.sum(axis=0).astype(int).tolist()

    return {
        "had_violation": bool(len(violation_steps) > 0),
        "violation_steps": violation_steps,  # indices in 0..T-1
        "violation_counts_per_tank": per_tank_counts,  # length K
        "bounds": {"lower": float(lower_limit), "upper": float(upper_limit), "tol": float(tol)},
        "total_violations": int(viol_mask.sum()),
    }


def heuristic_controller_safe(
    demand_series: Iterable[float],
    regimes: List[PumpingRegime],
    tank_capacity: float,
    lower_limit: float = 0.3,
    upper_limit: float = 0.9,
    target_fraction: float = 0.6,
    proportional_gain: float = 0.5,
    switching_penalty: float = 10.0,
    safety_margin: float = 0.0,
    use_weighted_scoring: bool = False,
) -> Tuple[List[List[int]], List[float], List[List[float]], dict]:
    """Heuristic with hard one-step safety mask.

    Among safe actions, minimise switches this step; break ties by matching the proportional targets.
    Set `use_weighted_scoring=True` to use a scalar score with `switching_penalty` instead of lexicographic.
    """
    n_stations = len(regimes)
    sim = CoolingLoopSimulator(
        regimes=regimes,
        tank_capacity=tank_capacity,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        target_fraction=target_fraction,
    )
    action_map = build_action_map_from_regimes(regimes)

    pump_history: List[List[int]] = []
    energy_history: List[float] = []
    level_history: List[List[float]] = []
    previous_pumps: List[int] = [0] * n_stations

    max_flows = [max(reg.flow_rates.values()) if reg.flow_rates else 1.0 for reg in regimes]
    flow_err_scale = float(sum(mf * mf for mf in max_flows)) or 1.0

    def flows_for_action(a: Tuple[int, ...]) -> List[float]:
        return [reg.flow(p) for reg, p in zip(regimes, a)]

    for demand in demand_series:
        levels = sim.get_levels()  # includes reservoir at index 0
        # bottom-up proportional targets
        target_flows: List[float] = [0.0] * n_stations
        downstream_target = demand
        for i in reversed(range(n_stations)):
            downstream_level = levels[i + 1]  # use R-(i+2), skip reservoir
            adjustment = proportional_gain * (
                (target_fraction - downstream_level) * tank_capacity / sim.timestep_hours
            )
            target_flow = max(0.0, downstream_target + adjustment)
            target_flows[i] = target_flow
            downstream_target = target_flow

        safe_set, fallback_idx = build_safe_action_set(
            sim=sim,
            demand=demand,
            action_map=action_map,
            lower_limit=lower_limit - safety_margin,
            upper_limit=upper_limit + safety_margin,
        )
        candidate_indices = safe_set if len(safe_set) > 0 else [fallback_idx]  # type: ignore

        best_idx: Optional[int] = None
        best_tuple: Optional[Tuple[int, float]] = None
        best_scalar: Optional[float] = None

        for idx in candidate_indices:
            a = action_map[idx]
            sw_step = sum(int(ai != pi) for ai, pi in zip(a, previous_pumps))
            flows = flows_for_action(a)
            flow_err = sum((f - t) ** 2 for f, t in zip(flows, target_flows))
            if use_weighted_scoring:
                score = switching_penalty * sw_step + (flow_err / flow_err_scale)
                if best_scalar is None or score < best_scalar:
                    best_scalar = score
                    best_idx = idx
            else:
                score_tuple = (sw_step, flow_err)
                if best_tuple is None or score_tuple < best_tuple:
                    best_tuple = score_tuple
                    best_idx = idx

        chosen = action_map[best_idx]  # type: ignore
        _, energy = sim.step(list(chosen), demand)

        pump_history.append(list(chosen))
        energy_history.append(energy)
        level_history.append(sim.get_levels()[1:])  # <-- ONLY R-2..R-5

        previous_pumps = list(chosen)

    info = make_violation_info(
        level_history, lower_limit=lower_limit, upper_limit=upper_limit, tol=0.0
    )
    return pump_history, energy_history, level_history, info
