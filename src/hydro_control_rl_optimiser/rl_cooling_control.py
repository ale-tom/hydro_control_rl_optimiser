"""
This module adopts the reinforcement learning framework to control all pumping
stations simultaneously, incorporating penalties for unmet demand and level
violations and adds a safety layer that overrides unsafe actions (e.g., leading to violations).

Due to the exponential growth of the state-action space,
a coarse discretisation (three level bins) and a limited number of
training episodes are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.hydro_control_rl_optimiser.cooling_control import (CoolingLoopSimulator, PumpingRegime)


@dataclass
class FullStateDiscretiser:
    """Discretise multiple reservoir levels and demand into a single state index.

    Each reservoir level is discretised into a specified number of bins defined by
    cut points. The demand is mapped to an index of discrete demand levels.
    The resulting state index is constructed by combining the bin indices
    using mixed-radix encoding.
    """

    level_bins: list[float]
    demand_map: dict[int, float]
    n_tanks: int

    def levels_to_bins(self, levels: list[float]) -> list[int]:
        bins = []
        for level in levels:
            b = 0
            for edge in self.level_bins:
                if level < edge:
                    break
                b += 1
            bins.append(b)
        return bins

    def demand_to_index(self, demand: float) -> int:
        levels = [v for _, v in sorted(self.demand_map.items())]
        return int(np.abs(np.array(levels) - demand).argmin())

    def state_to_index(self, levels: list[float], demand: float) -> int:
        bins = self.levels_to_bins(levels)
        demand_idx = self.demand_to_index(demand)
        radix = len(self.level_bins) + 1
        state = demand_idx
        for b in bins:
            state = state * radix + b
        return state

    @property
    def n_states(self) -> int:
        radix = len(self.level_bins) + 1
        return len(self.demand_map) * (radix**self.n_tanks)


@dataclass
class FullQLearningAgent:
    """Tabular Q-learning agent controlling all stations simultaneously.

    Actions are represented as pump count vectors for all stations. The
    Q-table maps discretised state indices to action values.
    """

    n_states: int
    pump_options: list[int]
    alpha: float
    gamma: float
    epsilon: float
    q_table: Optional[np.ndarray] = None
    action_map: Optional[list[tuple[int, ...]]] = None

    def __post_init__(self) -> None:
        # generate all possible actions as Cartesian product of pump_options
        from itertools import product

        self.action_map = list(product(self.pump_options, repeat=4))
        if self.q_table is None:
            self.q_table = np.zeros((self.n_states, len(self.action_map)))

    def choose_action(self, state: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_map))  # type: ignore
        return int(np.argmax(self.q_table[state]))  # type: ignore

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next = np.max(self.q_table[next_state])  # type: ignore
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]  # type: ignore
        self.q_table[state, action] += self.alpha * td_error  # type: ignore


def safe_apply_action(
    sim: CoolingLoopSimulator,
    action: tuple[int, ...],
    demand: float,
    lower_limit: float = 0.3,
    upper_limit: float = 0.9,
    max_iterations: int = 3,
) -> tuple[int, ...]:
    """Ensure that applying ``action`` will not violate reservoir level bounds.

    The function simulates one step with the proposed pump counts; if any reservoir
    level would fall below ``lower_limit`` or exceed ``upper_limit``, the pump
    counts for the offending station are adjusted up or down respectively.
    The adjustment is repeated up to ``max_iterations`` times. The original
    simulator state is restored before returning the safe action.
    """
    original_volumes = sim.tank_volumes.copy()
    pump_counts = list(action)
    for _ in range(max_iterations):
        temp_sim = CoolingLoopSimulator(regimes=sim.regimes, tank_capacity=sim.tank_capacity)
        temp_sim.tank_volumes = sim.tank_volumes.copy()
        temp_sim.step(pump_counts, demand)
        levels = temp_sim.get_levels()[1:]  # skip reservoir
        safe = True
        for i, lvl in enumerate(levels):
            if lvl < lower_limit and pump_counts[i] < max(sim.regimes[i].flow_rates.keys()):
                pump_counts[i] += 1
                safe = False
            elif lvl > upper_limit and pump_counts[i] > 0:
                pump_counts[i] -= 1
                safe = False
        if safe:
            break
    sim.tank_volumes = original_volumes
    return tuple(pump_counts)


def full_run_episode(
    agent: FullQLearningAgent,
    sim: CoolingLoopSimulator,
    demand_series: list[float],
    discretiser: FullStateDiscretiser,
    target_fraction: float = 0.6,
    switching_penalty: float = 5.0,
    energy_penalty: float = 0.001,
    unmet_penalty: float = 1.0,
    violation_penalty: float = 50.0,
    training: bool = True,
    lower_limit: float = 0.3,
    upper_limit: float = 0.9,
) -> float:
    """Run one episode; the safety layer uses the provided bounds during training and evaluation.

    Passing consistent lower_limit and upper_limit here aligns the agent’s experience with the
    evaluation mask, reducing post-training violations caused by train/eval mismatches.
    """
    total_reward = 0.0
    previous_action = (0, 0, 0, 0)
    for demand in demand_series:
        levels = sim.get_levels()[1:]
        state = discretiser.state_to_index(levels, demand)
        action_idx = agent.choose_action(state, training=training)
        action = agent.action_map[action_idx]  # type: ignore

        # Use the SAME limits as evaluation
        safe_action = safe_apply_action(
            sim, action, demand, lower_limit=lower_limit, upper_limit=upper_limit
        )

        _, energy = sim.step(list(safe_action), demand)
        delivered = sim.regimes[-1].flow(safe_action[-1])
        reward = 0.0
        if delivered < demand:
            shortfall_m3 = (demand - delivered) * sim.timestep_hours
            reward -= unmet_penalty * shortfall_m3
        new_levels = sim.get_levels()[1:]
        for lvl in new_levels:
            if lvl < lower_limit or lvl > upper_limit:
                reward -= violation_penalty
            reward -= abs(lvl - target_fraction)
        if safe_action != previous_action:
            reward -= switching_penalty
        reward -= energy_penalty * energy

        total_reward += reward
        next_state = discretiser.state_to_index(new_levels, demand)
        if training:
            agent.update(state, action_idx, reward, next_state)
        previous_action = safe_action
    return total_reward


def full_train_agent(
    agent: FullQLearningAgent,
    regimes: list[PumpingRegime],
    demand_series: list[float],
    discretiser: FullStateDiscretiser,
    episodes: int = 50,
    target_fraction: float = 0.6,
    switching_penalty: float = 5.0,
    energy_penalty: float = 0.001,
    unmet_penalty: float = 1.0,
    violation_penalty: float = 50.0,
    lower_limit: float = 0.3,
    upper_limit: float = 0.9,
) -> None:
    """Train the agent using the same safety limits that will be used at evaluation time."""
    for _ in range(episodes):
        sim = CoolingLoopSimulator(regimes=regimes, tank_capacity=1500.0)
        full_run_episode(
            agent=agent,
            sim=sim,
            demand_series=demand_series,
            discretiser=discretiser,
            target_fraction=target_fraction,
            switching_penalty=switching_penalty,
            energy_penalty=energy_penalty,
            unmet_penalty=unmet_penalty,
            violation_penalty=violation_penalty,
            training=True,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )


def full_evaluate_agent(
    agent: FullQLearningAgent,
    regimes: list[PumpingRegime],
    demand_series: list[float],
    discretiser: FullStateDiscretiser,
    target_fraction: float = 0.6,
    switching_penalty: float = 5.0,
    energy_penalty: float = 0.001,
    unmet_penalty: float = 1.0,
    violation_penalty: float = 50.0,
) -> tuple[list[tuple[int, ...]], float, list[list[float]]]:
    """Evaluate the trained agent on a fixed demand series."""
    sim = CoolingLoopSimulator(regimes=regimes, tank_capacity=1500.0)
    actions: list[tuple[int, ...]] = []
    energies: list[float] = []
    levels_history: list[list[float]] = []
    
    for demand in demand_series:
        levels = sim.get_levels()[1:]
        state = discretiser.state_to_index(levels, demand)
        action_idx = agent.choose_action(state, training=False)
        action = agent.action_map[action_idx]  # type: ignore
        safe_action = safe_apply_action(sim, action, demand)
        _, energy = sim.step(list(safe_action), demand)
        actions.append(safe_action)
        energies.append(energy)
        levels_history.append(sim.get_levels()[1:])
        
    return actions, float(sum(energies)), levels_history
