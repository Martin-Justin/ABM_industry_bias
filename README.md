# Industry Sponsorship as Meta-Evidence: A Simulation Study

Code and simulation data for the manuscript *Industry Sponsorship as Meta-Evidence: A Simulation
Study*. The model is an agent-based model of scientific inquiry in which industry-funded
Researchers introduce bias into their results and a central Evaluator aggregates those results
under different strategies. The model, the parameters, and the results are described in the paper;
this file only explains how to reproduce the results and how to run the model yourself.

## Contents

| Path | |
| --- | --- |
| `ABM_Analysis.ipynb` | Reproduces the paper's figures and tables from the simulation data |
| `Industry_Bias_ABM.ipynb` | Runs new simulations, then analyses their output |
| `code/model_binomial_v2.py` | The model: agents, biasing mechanisms, Evaluator strategies |
| `code/run_2.py` | Batch runner, driven by `Industry_Bias_ABM.ipynb` |
| `data/` | Simulation output (not in the repository; see below) |

There are two notebooks. If you want to check the paper's results, use
`ABM_Analysis.ipynb`: it only reads data and cannot overwrite anything. If you want to run the model
with your own parameters, use `Industry_Bias_ABM.ipynb`.

## Setup

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Reproducing the paper's results

### Running the analysis

Open `ABM_Analysis.ipynb` and run all cells. For each scenario it

- saves the convergence chart as `base_final.png`, `hard_final.png`, or `weights_final.png` next to
  the notebook, and
- prints the trajectory breakdown tables, which split the convergence window into the different
  ways a run can converge or fail to.

Missing files are skipped, so you can download a single scenario, or a single bias type, and
analyse only that. Each file is read in full, which takes from a minute (`diffweight`) to several
minutes (`hard`).

## Running your own simulations

Part 1 of `Industry_Bias_ABM.ipynb` runs simulations. Everything is set in a single `CONFIG`
dictionary — the hypotheses and their success rates, the number of agents of each kind, study
sizes, the run length, and how many times each parameter combination is repeated. Any key you leave
out keeps its default, and the notebook prints the full list of keys and defaults before you edit
it.

A run sweeps one **bias type** across a range of **bias levels**, under each **Evaluator strategy**.
Setting `bias_type` is all that is needed to switch between the three biasing mechanisms; the
notebook works out what that means for the model's parameters. The bias level is always a
probability:

| `bias_type` | The bias level is the probability that... |
| --- | --- |
| `design_bias` | an industry study compares its hypothesis against the inferior control |
| `publication_bias` | an industry agent withholds a negative result about its favourite hypothesis |
| `industry_selection` | an industry study of the second-best hypothesis is biased |

The four Evaluator strategies are `"No adjustment"` and `"Ignoring industry"` (the two benchmarks),
`"Down-weighting"` and `"Relative adjustment"`. To compare several down-weighting weights in one
file, write the entry as a pair, `("Down-weighting", 0.1)`.

A minimal configuration therefore looks like this:

```python
CONFIG = {
    "hypotheses": [0.51, 0.5, 0.49],
    "n_public": 5,
    "n_industry": 5,
    "bias_levels": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "repetitions": 200,
}
RUN_NAME = "my_run"
BIAS_TYPES_TO_RUN = ["design_bias", "publication_bias", "industry_selection"]
```

Running the next cell writes one CSV per bias type into `data/`, named
`{RUN_NAME}_{bias_type}.csv`. A file of the same name is overwritten, so give every run its own
`RUN_NAME`. Simulations are spread across processes; set `N_WORKERS` to the number of cores to use,
or to `1` to run in the notebook itself. Setting `"seed"` to an integer makes the whole sweep
reproducible.

The parameters used in the paper are given in Tables 1 and 2, and are also stored in
`code/run_2.py` as `SCENARIO_1` (base) and `SCENARIO_2` (hard). That file can be run directly from
the command line (`python code/run_2.py`) if you would rather not work in a notebook.

### Analysing your own results

Part 2 of `Industry_Bias_ABM.ipynb` reads the CSVs back and reports **correct convergence**: the
share of runs in which the public agents test the best hypothesis, and only it, in every round of a
window at the end of the simulation. It produces the convergence chart, the same rates as a table,
and the trajectory breakdown. Run only the Part 2 cells if you do not want to start a new sweep.

`RESULT_FILES` maps each panel title to a file and by default points at the files Part 1 wrote for
the current `RUN_NAME`. Missing entries are skipped, so the analysis works whether you ran one bias
type or all three. Part 2 can also read the paper's data files; point `RESULT_FILES` at them
directly:

```python
RESULT_FILES = {
    "Design bias": DATA_DIR / "diffweight_sitebias.csv",
    "Publication bias": DATA_DIR / "diffweight_pubbias.csv",
    "Industry selection": DATA_DIR / "diffweight_indselect.csv",
}
```
