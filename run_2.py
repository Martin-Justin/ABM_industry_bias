import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
import os

# Import the agents and run function from your existing model file
from model_binomial_v2 import AltIndustry, BiasedPublicAgent, IndustryAgent, run

def worker(args):
    pub_agents, ind_agents, bandits, i, placebo, placebo_site, inquiry_pub, inquiry_ind, style, public_study, industry_study, pub_bias, site_bias, criterion, pulls, weight, rounds, data_collection = args
    
    # Fetch cryptographically secure random bytes from the OS to seed the RNGs.
    # This prevents the "identical twin" problem on Linux/Mac clusters without
    # locking the simulation into perfect reproducibility.
    os_entropy = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(os_entropy)
    
    # We also import and seed the standard random library, since your BayesAgent uses it
    import random
    random.seed(os_entropy)
    
    results = run(pub_agents, ind_agents, bandits, i, placebo, placebo_site, inquiry_pub, inquiry_ind, style, public_study, industry_study, pub_bias, site_bias, criterion, pulls, weight, rounds, data_collection)
    return pd.DataFrame(results)

bandits1 = [0.51, 0.5, 0.49]
bandits2 = [0.7, 0.52, 0.51, 0.5, 0.49, 0.48]

if __name__ == "__main__":
    freeze_support()
    n_workers = 22  # Adjust if needed
    args_list = []
    placebo_site = 0.48
    placebo_site2 = 0.3
    inquiry_pub, inquiry_ind = "softmax", "softmax"
    agents = (1, 3)
    
    # 1. Build the complete list of tasks
    for style in ["down_weighing_neutral", "normal", "ignore", "adaptive_adjustment"]:
        for site_ind in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for i in range(5000): # Change to 5000 for final paper run
                args = ((BiasedPublicAgent, agents[0]), (AltIndustry, agents[1]),
                        bandits2, i, 0.5, placebo_site2, inquiry_pub, inquiry_ind, 
                        style, 10000, 10000, (0, 0), (0, site_ind), 0.01, 10000, 10, 10, 1)
                args_list.append(args)
    
    # 2. Execute all tasks in a single Pool (Much faster)
    print(f"Starting {len(args_list)} simulations across {n_workers} cores...")
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(worker, args_list), total=len(args_list)))
        
    # 3. Concatenate and save
    results_drug_approval_test = pd.concat(results, ignore_index=True)
    
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    results_drug_approval_test.to_csv("G:/My Drive/simulation results/industry funding bias/final/hard_indselect.csv", index=False)
    # results_drug_approval_test.to_csv("test3.csv", index=False)
    print("Done!")