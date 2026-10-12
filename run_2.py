"""Batch runner for the industry-bias ABM.

`run_sweep` takes one configuration dictionary, expands it into one simulation per
(strategy, bias level, repetition), runs those across processes and returns a tidy
DataFrame. The notebook in the repository root is the intended front end; the block
at the bottom of this file does the same thing from the command line.
"""

import os
import random
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from model_binomial_v2 import AltIndustry, BiasedPublicAgent, IndustryAgent, run


# --- what a bias type means for the model's parameters ---------------------
# Public Researchers are always unbiased `BiasedPublicAgent`s on the neutral
# ("softmax") inquiry rule, so only the industry side of the community varies.
#
#   agent    the industry agent class
#   inquiry  the industry inquiry rule. "softmax_industry" adds the +1 utility
#            bonus for the agent's own favourite hypothesis; "softmax" is the
#            same neutral rule the public agents use
#   channel  the model parameter the bias level is written into
BIAS_TYPES = {
    "design_bias": {
        "agent": IndustryAgent, "inquiry": "softmax_industry", "channel": "site_bias",
        "meaning": "probability of comparing a hypothesis against the inferior control",
    },
    "publication_bias": {
        "agent": IndustryAgent, "inquiry": "softmax_industry", "channel": "pub_bias",
        "meaning": "probability of withholding a negative result about the favourite hypothesis",
    },
    "industry_selection": {
        "agent": AltIndustry, "inquiry": "softmax", "channel": "site_bias",
        "meaning": "probability of biased results about the second-best hypothesis",
    },
}

# Evaluator strategies, keyed by the names used in the paper.
STRATEGIES = {
    "No adjustment": "normal",
    "Ignoring industry": "ignore",
    "Down-weighting": "down_weighing_neutral",
    "Relative adjustment": "adaptive_adjustment_significance",
}

DEFAULT_CONFIG = {
    # --- the world ---------------------------------------------------------
    "hypotheses": [0.51, 0.5, 0.49],  # success rates, best first
    "control": 0.5,                   # success rate of the control condition
    "biased_control": 0.48,           # inferior comparator industry can switch to
    # --- the community -----------------------------------------------------
    "n_public": 5,
    "n_industry": 5,
    "study_size_public": 200,         # participants per arm, per study
    "study_size_industry": 200,       # industry selection runs studies 1.5x this size
    "pulls": 200,                     # participants per arm, per agent, per round;
                                      # pulls / study_size = studies per agent per round
    "utility_weight": 10,             # w in u(H) = w * Suc(H)
    # --- what is varied ----------------------------------------------------
    "bias_type": "design_bias",
    "bias_levels": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "strategies": list(STRATEGIES),
    # --- Evaluator settings ------------------------------------------------
    "ind_weight": 0.5,                # weight on industry studies under down-weighting
    "min_studies": 5,                 # studies per funding source before relative
                                      # adjustment will test for a bias factor
    # --- run length --------------------------------------------------------
    "rounds": 100,
    "repetitions": 200,               # the paper uses 5000
    "data_collection": 1,             # record every nth round
    "seed": None,                     # int for reproducible runs, None for fresh entropy
}


def resolve_strategy(strategy, default_weight):
    """Read one `strategies` entry.

    An entry is either a strategy name, or a (name, industry weight) pair. The pair
    form exists so that several down-weighting weights can be swept in one file, as
    in Figure 2 of the paper: ("Down-weighting", 0.1).
    """
    name, weight = strategy if isinstance(strategy, tuple) else (strategy, default_weight)
    style = STRATEGIES.get(name, name)

    if style not in set(STRATEGIES.values()):
        raise ValueError(f"Unknown strategy {name!r}. Use one of {sorted(STRATEGIES)} "
                         f"or its internal name {sorted(set(STRATEGIES.values()))}.")
    return style, weight


def resolve_config(config):
    """Merge a user configuration onto the defaults and check it over."""
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {sorted(unknown)}. "
                         f"Valid keys are {sorted(DEFAULT_CONFIG)}.")

    cfg = {**DEFAULT_CONFIG, **config}

    if cfg["bias_type"] not in BIAS_TYPES:
        raise ValueError(f"bias_type must be one of {sorted(BIAS_TYPES)}, "
                         f"got {cfg['bias_type']!r}.")

    for strategy in cfg["strategies"]:
        resolve_strategy(strategy, cfg["ind_weight"])

    if not all(0 <= level <= 1 for level in cfg["bias_levels"]):
        raise ValueError("bias_levels are probabilities and must lie in [0, 1].")

    # The log odds ratio divides by (1 - mean), so certain success or certain
    # failure would break the statistic rather than merely being uninformative.
    rates = list(cfg["hypotheses"]) + [cfg["control"], cfg["biased_control"]]
    if not all(0 < rate < 1 for rate in rates):
        raise ValueError("Hypothesis and control success rates must lie strictly "
                         "between 0 and 1, so that the odds ratio stays defined.")

    if cfg["biased_control"] >= cfg["control"]:
        raise ValueError("biased_control is the inferior comparator and must be "
                         "below control, otherwise it inflates nothing.")

    # AltIndustry treats hypotheses[1] as the second-best option, which only holds
    # if the list is ordered.
    if cfg["bias_type"] == "industry_selection":
        if list(cfg["hypotheses"]) != sorted(cfg["hypotheses"], reverse=True):
            raise ValueError("industry_selection biases results about hypotheses[1], "
                             "taken to be the second-best option, so hypotheses must "
                             "be listed in descending order of success rate.")

    return cfg


def describe_config(cfg):
    """A short human-readable summary of a resolved configuration."""
    bias = BIAS_TYPES[cfg["bias_type"]]
    strategies = [s[0] if isinstance(s, tuple) else s for s in cfg["strategies"]]

    return "\n".join([
        f"  bias type    {cfg['bias_type']} - {bias['meaning']}",
        f"  levels       {cfg['bias_levels']}",
        f"  strategies   {', '.join(strategies)}",
        f"  hypotheses   {len(cfg['hypotheses'])}, best {max(cfg['hypotheses'])}, "
        f"control {cfg['control']} / {cfg['biased_control']}",
        f"  agents       {cfg['n_public']} public, {cfg['n_industry']} industry",
        f"  study size   {cfg['study_size_public']} public / {cfg['study_size_industry']} "
        f"industry, {cfg['pulls']} participants per agent per round",
        f"  Evaluator    down-weighting w = {cfg['ind_weight']}, "
        f"relative adjustment from {cfg['min_studies']} studies per source",
        f"  run length   {cfg['rounds']} rounds x {cfg['repetitions']} repetitions",
    ])


def build_tasks(cfg):
    """Expand a resolved configuration into one (keyword arguments, seed) task per run."""
    bias = BIAS_TYPES[cfg["bias_type"]]
    tasks = []

    for strategy in cfg["strategies"]:
        style, ind_weight = resolve_strategy(strategy, cfg["ind_weight"])

        for level in cfg["bias_levels"]:
            # Public agents are never biased, so the first element of each pair is 0.
            pub_bias = (0, level if bias["channel"] == "pub_bias" else 0)
            site_bias = (0, level if bias["channel"] == "site_bias" else 0)

            for _ in range(cfg["repetitions"]):
                run_id = len(tasks)
                kwargs = {
                    "pub_agents": (BiasedPublicAgent, cfg["n_public"]),
                    "ind_agents": (bias["agent"], cfg["n_industry"]),
                    "bandits": cfg["hypotheses"],
                    "run_id": run_id,
                    "placebo": cfg["control"],
                    "placebo_site": cfg["biased_control"],
                    "inquiry_pub": "softmax",
                    "inquiry_ind": bias["inquiry"],
                    "style": style,
                    "study_size_public": cfg["study_size_public"],
                    "study_size_industry": cfg["study_size_industry"],
                    "pub_bias": pub_bias,
                    "site_select_bias": site_bias,
                    "criterion": 0,
                    "pulls": cfg["pulls"],
                    "weight": cfg["utility_weight"],
                    "rounds": cfg["rounds"],
                    "data_collection": cfg["data_collection"],
                    "ind_weight": ind_weight,
                    "min_studies": cfg["min_studies"],
                }
                seed = None if cfg["seed"] is None else cfg["seed"] + run_id
                tasks.append((kwargs, seed))

    return tasks


def worker(task):
    """Run one simulation and return its rows as a DataFrame."""
    kwargs, seed = task

    # Without an explicit seed, take fresh entropy from the OS. Workers otherwise
    # inherit one RNG state on fork and produce identical "independent" runs.
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    np.random.seed(seed)
    random.seed(seed)  # the model draws from both generators

    return pd.DataFrame(run(**kwargs))


def run_sweep(config, n_workers=None, out_path=None, quiet=False):
    """Run every (strategy, bias level, repetition) in `config` and return the results.

    Set `out_path` to also write the results to CSV. Pass `n_workers=1` to run in
    the current process, which is slower but easier to interrupt and debug.
    """
    cfg = resolve_config(config)
    tasks = build_tasks(cfg)
    n_workers = n_workers or max(1, cpu_count() - 1)

    if not quiet:
        print(describe_config(cfg))
        print(f"\n{len(tasks)} simulations on {n_workers} worker(s)")

    if n_workers == 1:
        frames = [worker(task) for task in tqdm(tasks, disable=quiet)]
    else:
        # Results are concatenated per run, so the order runs finish in does not matter.
        with Pool(n_workers) as pool:
            frames = list(tqdm(pool.imap_unordered(worker, tasks, chunksize=8),
                               total=len(tasks), disable=quiet))

    results = pd.concat(frames, ignore_index=True)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        if not quiet:
            print(f"Written to {out_path} ({len(results)} rows)")

    return results


# The paper's first scenario (Table 1): 3 hypotheses, five agents of each kind, small studies.
SCENARIO_1 = {
    "hypotheses": [0.51, 0.5, 0.49],
    "control": 0.5,
    "biased_control": 0.48,
    "n_public": 5,
    "n_industry": 5,
    "study_size_public": 200,
    "study_size_industry": 200,
    "pulls": 200,
    "rounds": 100,
    "repetitions": 5000,
}

# The paper's second scenario (Table 2): 55 hypotheses, one agent of each kind, large studies.
SCENARIO_2 = {
    "hypotheses": [0.515] + [round(0.508 - 0.002 * step, 3) for step in range(54)],
    "control": 0.5,
    "biased_control": 0.48,
    "n_public": 1,
    "n_industry": 1,
    "study_size_public": 20000,
    "study_size_industry": 20000,
    "pulls": 20000,
    "min_studies": 2,                 # one agent of each kind: see footnote 2 of the paper
    "rounds": 100,
    "repetitions": 5000,
}


if __name__ == "__main__":
    freeze_support()
    run_sweep({**SCENARIO_2, "bias_type": "publication_bias"},
              n_workers=20, out_path="data/scenario2_publication_bias.csv")
