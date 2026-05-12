# Industry Bias Agent-Based Model (ABM)

This project contains an Agent-Based Model (ABM) simulating the effects of industry bias in scientific research, specifically focusing on fields like biomedicine where industry funding is prevalent and introduces specific methodological biases.

## Model Description

The model simulates both publicly funded and industry-funded agents researching various "bandits" (representing treatments or interventions). The simulation explores how different types of bias impact the overall scientific discovery process:
- **Selection Bias:** Industry agents sample an alternative, less effective control to make a treatment appear more effective.
- **Publication Bias:** Negative results may be selectively hidden by industry agents.
- **Industry Selection Bias:** Industry agents conduct larger and more frequent studies on their preferred topics, occasionally producing biased results (inspired by Holman & Bruner, 2017).

The model tests various regulatory/evaluator responses to this bias:
1. **Normal / Ignore:** Benchmark conditions (taking industry results at face value or completely ignoring them).
2. **Down-weighing (`down_weighing_neutral`):** Giving industry results a lower weight when aggregating evidence.
3. **Adaptive Adjustment (`adaptive_adjustment` / `Solomon`):** Dynamically adjusting the perceived effect of industry results based on their historical divergence from publicly funded results.

## Requirements

The project uses a few core scientific Python libraries. To set up the environment, you can install the dependencies listed in `requirements.txt`:

```bash
# Optional: create a virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Model

The main simulation script is `run_2.py`. It uses `multiprocessing` to run thousands of iterations across different model parameters and stores the output data.

1. **Configure Parameters:** Open `run_2.py` to adjust settings like `n_workers` (number of CPU cores to use) or the number of rounds/iterations. By default, it runs an extensive grid of simulations and outputs to a CSV file.
2. **Execute:**
   ```bash
   python run_2.py
   ```
3. **Data Analysis:** After running the simulations, you can use the Jupyter Notebook `ABM_Analysis.ipynb` to analyze the generated CSV files and visualize the convergence, approval rates, and evaluation scores. Run `jupyter notebook` in your terminal to launch the notebook environment.
