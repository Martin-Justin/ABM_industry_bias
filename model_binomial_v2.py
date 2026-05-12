# Approval multi-bandit model
# Updated version from Mar 2026

from random import randint, random, choices, choice
from collections import defaultdict
# import bnlearn as bn
import math
# import pickle
import joblib
import numpy as np


class BayesAgent:
    def __init__(self, study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i):
        self.id = i
        self.studies = study_size
        self.inquiry = inquiry_norm
        self.last_tested = None

    def choose_bandit(self, bandits, round, agents, weight):
        if self.inquiry == "softmax":
            def utility(bandit):
                return weight * bandit["evaluation"]

            if not self.last_tested:
                return bandits[randint(0, len(bandits) - 1)]
            else:
                # 1. Calculate Utilities
                # We create a list of just the scores for vectorization
                raw_scores = np.array([utility(b) for b in bandits])

                ### for linear:
                temp = max(0, (1 - 0.02 * round))
                if temp > 0:
                    shifted_scores = raw_scores - np.max(raw_scores)
                    exp_weights = np.exp(shifted_scores / temp)
                    probabilities = exp_weights / np.sum(exp_weights)
                    chosen_index = np.random.choice(len(bandits), p=probabilities)
                else:
                    chosen_index = np.argmax(raw_scores)

                ### for exponential:
                # decay_rate = 0.96
                # temp = max(0.001, 1.0 * (decay_rate ** round))
                # shifted_scores = raw_scores - np.max(raw_scores)
                # exp_weights = np.exp(shifted_scores / temp)
                # probabilities = exp_weights / np.sum(exp_weights)
                # chosen_index = np.random.choice(len(bandits), p=probabilities)

                return bandits[chosen_index]

        elif self.inquiry == "softmax_industry":
            def utility(bandit):
                return weight * bandit["evaluation"] + 1 if self.favorite == bandit else weight * bandit["evaluation"]

            if not self.last_tested:
                self.favorite = bandits[randint(0, len(bandits) - 1)]
                self.favorite["industry_fav"] = True
                return self.favorite
            else:
                # 1. Calculate Utilities
                # We create a list of just the scores for vectorization
                raw_scores = np.array([utility(b) for b in bandits])

                ### for linear:
                # temp = max(0, (1 - 0.02 * round))
                temp = 0.2
                if temp > 0:
                    shifted_scores = raw_scores - np.max(raw_scores)
                    exp_weights = np.exp(shifted_scores / temp)
                    probabilities = exp_weights / np.sum(exp_weights)
                    chosen_index = np.random.choice(len(bandits), p=probabilities)
                else:
                    chosen_index = np.argmax(raw_scores)

                ### for exponential:
                # decay_rate = 0.96
                # temp = max(0.001, 1.0 * (decay_rate ** round))
                # shifted_scores = raw_scores - np.max(raw_scores)
                # exp_weights = np.exp(shifted_scores / temp)
                # probabilities = exp_weights / np.sum(exp_weights)
                # chosen_index = np.random.choice(len(bandits), p=probabilities)

                return bandits[chosen_index]


        elif self.inquiry == "myopic":
            def utility(bandit):
                return weight * bandit["evaluation"]

            if not self.last_tested:
                return bandits[randint(0, len(bandits) - 1)]
            else:
                utilities = [(bandit, utility(bandit)) for bandit in bandits]
                max_utility = max(u for _, u in utilities)
                best_candidates = [bandit for bandit, u in utilities if u == max_utility]
                return choice(best_candidates)

        elif self.inquiry == "myopic_industry":
            def utility(bandit):
                return weight * bandit["evaluation"] + 1 if self.favorite == bandit else weight * bandit["evaluation"]

            if not self.last_tested:
                self.favorite = bandits[randint(0, len(bandits) - 1)]
                self.favorite["industry_fav"] = True
                return self.favorite
            else:
                utilities = [(bandit, utility(bandit)) for bandit in bandits]
                max_utility = max(u for _, u in utilities)
                best_candidates = [bandit for bandit, u in utilities if u == max_utility]
                return choice(best_candidates)

        else:
            raise ValueError("Invalid inquiry_norm value provided")

    def sample_sizes(self, pulls):
        study_sizes = list()
        samples = pulls
        while samples > 0:
            part = min(self.studies, samples - self.studies) if (samples - self.studies) >= self.studies else samples
            study_sizes.append(part)
            samples -= part
        return study_sizes

    def test(self, bandits, placebo, site_placebo, pulls, round, agents, success, weight):
        study_sizes = self.sample_sizes(pulls)
        bandit = self.choose_bandit(bandits, round, agents, weight)
        self.last_tested = bandit

        for study_size in study_sizes:
             bandit_sample = np.random.binomial(study_size, bandit["p"])
             bandit_mean = bandit_sample / study_size
             placebo_sample = np.random.binomial(study_size, placebo)
             placebo_mean = placebo_sample / study_size

             bandit_odds = bandit_mean / (1 - bandit_mean)
             placebo_odds = placebo_mean / (1 - placebo_mean)
             log_OR = math.log(bandit_odds / placebo_odds)

             bandit["pub_studies"].append(log_OR)
             success[0] += bandit_sample
             success[1] += study_size - bandit_sample



class IndustryAgent(BayesAgent):
    def __init__(self, study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i):
        super().__init__(study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i)
        self.pub_bias = pub_bias
        self.site_bias = site_select_bias
        self.favorite = None

    def test(self, bandits, placebo, site_placebo, pulls, round, agents, success, weight):
        study_sizes = self.sample_sizes(pulls)
        bandit = self.choose_bandit(bandits, round, agents, weight)
        self.last_tested = bandit

        for study_size in study_sizes:
            control = site_placebo if random() < self.site_bias else placebo
            effect = bandit["p"]

            bandit_sample = np.random.binomial(study_size, effect)
            bandit_mean = bandit_sample / study_size
            placebo_sample = np.random.binomial(study_size, control)
            placebo_mean = placebo_sample / study_size

            bandit_odds = bandit_mean / (1 - bandit_mean)
            placebo_odds = placebo_mean / (1 - placebo_mean)
            log_OR = math.log(bandit_odds / placebo_odds)

            if self.last_tested == self.favorite:
                if (log_OR <= 0 and random() > self.pub_bias) or log_OR > 0:
                    bandit["nr_industry_studies"] += 1
                    bandit["industry_studies"].append(log_OR)
            else:
                bandit["nr_industry_studies"] += 1
                bandit["industry_studies"].append(log_OR)

class BiasedPublicAgent(BayesAgent):
    ### Public agents with bias!
    def __init__(self, study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i):
        super().__init__(study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i)
        self.site_bias = site_select_bias
        self.pub_bias = pub_bias

    def test(self, bandits, placebo, site_placebo, pulls, round, agents, success, weight):
        study_sizes = self.sample_sizes(pulls)
        bandit = self.choose_bandit(bandits, round, agents, weight)
        self.last_tested = bandit

        for study_size in study_sizes:
            control = site_placebo if random() < self.site_bias else placebo
            effect = bandit["p"]

            bandit_sample = np.random.binomial(study_size, effect)
            bandit_mean = bandit_sample / study_size
            placebo_sample = np.random.binomial(study_size, control)
            placebo_mean = placebo_sample / study_size

            bandit_odds = bandit_mean / (1 - bandit_mean)
            placebo_odds = placebo_mean / (1 - placebo_mean)
            log_OR = math.log(bandit_odds / placebo_odds)

            success[0] += bandit_sample
            success[1] += study_size - bandit_sample

            if (log_OR <= 0 and random() > self.pub_bias) or log_OR > 0:
                bandit["pub_studies"].append(log_OR)

class AltIndustry(BayesAgent):
    def __init__(self, study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i):
        super().__init__(study_size, inquiry_norm, pub_bias, site_select_bias, bandits, i)
        self.pub_bias = pub_bias
        self.site_bias = site_select_bias
        self.favorite = bandits[1]

    def test(self, bandits, placebo, site_placebo, pulls, round, agents, success, weight):
        study_sizes = self.sample_sizes(pulls)
        bandit = self.choose_bandit(bandits, round, agents, weight)
        if bandit == self.favorite:
            control = site_placebo if random() < self.site_bias else placebo
        else:
            control = placebo
        self.last_tested = bandit

        for study_size in study_sizes:
            study_size = int(study_size * 1.5)
            effect = bandit["p"]

            bandit_sample = np.random.binomial(study_size, effect)
            bandit_mean = bandit_sample / study_size
            placebo_sample = np.random.binomial(study_size, control)
            placebo_mean = placebo_sample / study_size

            bandit_odds = bandit_mean / (1 - bandit_mean)
            placebo_odds = placebo_mean / (1 - placebo_mean)
            log_OR = math.log(bandit_odds / placebo_odds)

            bandit["nr_industry_studies"] += 1
            bandit["industry_studies"].append(log_OR)

            # if (log_OR <= 0 and random() > self.pub_bias) or log_OR > 0:
            #     bandit["nr_industry_studies"] += 1
            #     bandit["industry_studies"].append(log_OR)


def evaluation(drugs, style, criterion, round):
#     for drug in drugs:
#         drug["studies"].extend(drug["pub_studies"])
#         drug["pub_studies_collected"].extend(drug["pub_studies"])
#         drug["nr_studies"] += len(drug["pub_studies"])
#         drug["pub_studies"] = list()
#
#         drug["ind_studies_collected"].extend(drug["industry_studies"])
#         if drug["industry_studies"] and style != "ignore":
#             for study in drug["industry_studies"]:
#                 new_value = discounting(study, drug, style, round)
#                 drug["studies"].append(new_value)
#             drug["industry_studies"] = list()
#         else:
#             drug["industry_studies"] = list()
#
#         if drug["studies"]:
#             drug["evaluation"] = sum(drug["studies"]) / len(drug["studies"])
#
#         if round % 20 == 0:
#             criterion_pass = drug["evaluation"] > criterion
#             drug["approved"] = True if not drug["approved"] and criterion_pass else drug["approved"]
#             drug["approved"] = False if drug["approved"] and not criterion_pass else drug["approved"]

    for drug in drugs:
        # 1. Update the historical collections with the new studies from this round
        drug["pub_studies_collected"].extend(drug["pub_studies"])
        # drug["ind_studies_collected"].extend(drug["industry_studies"])

        # 2. Maintain the raw lists for standard aggregation
        drug["studies"].extend(drug["pub_studies"])
        drug["nr_studies"] += len(drug["pub_studies"])
        drug["pub_studies"] = list()
        if drug["industry_studies"] and style != "ignore":
            for study in drug["industry_studies"]:
                # Only use the discounting function for adaptive/Solomon adjustments
                if style not in ["down_weighing_neutral"]:
                    new_value = discounting(study, drug, style, round)
                    drug["studies"].append(new_value)
                else:
                    # For down-weighting, keep the unadjusted value in the general list
                    # (Though we won't use this list to calculate the evaluation below)
                    drug["studies"].append(study)

            drug["ind_studies_collected"].extend(drug["industry_studies"])
            drug["industry_studies"] = list()
        else:
            drug["industry_studies"] = list()

        # 3. Calculate the evaluation score
        if style in ["down_weighing_neutral"]:
            # --- FIXED WEIGHTED AVERAGE ---
            w = 0.5  # The weight assigned to industry studies

            total_pub_weight = len(drug["pub_studies_collected"])
            total_ind_weight = len(drug["ind_studies_collected"]) * w
            total_weight = total_pub_weight + total_ind_weight

            if total_weight > 0:
                sum_pub = sum(drug["pub_studies_collected"])
                sum_ind = sum(drug["ind_studies_collected"])
                drug["evaluation"] = (sum_pub + (w * sum_ind)) / total_weight
            else:
                drug["evaluation"] = 0

        else:
            # Standard unweighted average (used for normal, adaptive, etc.)
            if drug["studies"]:
                drug["evaluation"] = sum(drug["studies"]) / len(drug["studies"])

def discounting(study, drug, style, round):
#     if style == "down_weighing":
#         if study > 0:
#             return study * 0.5
#         else:
#             return study
#
#     if style == "down_weighing_neutral":
#         return study * 0.5

    if style == "adaptive_adjustment":
        if round > 0:
            industry_eval = sum(drug["ind_studies_collected"]) / len(drug["ind_studies_collected"]) if drug["ind_studies_collected"] else 0
            public_eval = sum(drug["pub_studies_collected"]) / len(drug["pub_studies_collected"]) if drug["pub_studies_collected"] else 0

            if industry_eval and public_eval:
                factor = industry_eval - public_eval
            else:
                factor = 0
            return study - factor

        else:
            return study

    elif style == "Solomon":
        if len(drug["ind_studies_collected"]) > 5 and len(drug["pub_studies_collected"]) > 5:
            pos_ind = sum(i > 0 for i in drug["ind_studies_collected"])
            neg_ind = len(drug["ind_studies_collected"]) - pos_ind
            pos_pub = sum(i > 0 for i in drug["pub_studies_collected"])
            neg_pub = len(drug["pub_studies_collected"]) - pos_pub

            odds_ratio = ((pos_ind + 0.01) * (neg_pub + 0.01)) / ((neg_ind + 0.01) * (pos_pub + 0.01))  # Applying Haldane-Anscombe correction to prevent division by zero

            # study_odds = math.exp(study)
            # return math.log(study_odds * odds_ratio**(-1))
            return study - math.log(odds_ratio)  # The two options are mathematically equal

        else:
            return study


    elif style == "normal":
        return study

    else:
        raise ValueError("Invalid style value provided")

def compute_convergence_approval(drugs, placebo):
    nr_drugs = len(drugs)
    correctly_approved = 0
    for drug in drugs:
        if drug["p"] > placebo and drug["approved"]:
            correctly_approved += 1
        elif drug["p"] <= placebo and not drug["approved"]:
            correctly_approved += 1
    return correctly_approved / nr_drugs

def compute_convergence_testing(agents, max_p, bandits):
    correct_convergence = 0
    for agent in agents:
        if agent.last_tested["p"] == max_p:
            correct_convergence += 1
    return correct_convergence / len(agents)

def active_hypotheses(agents):
    testing = set()
    for agent in agents:
        if agent.id[:3] == "pub":
            testing.add(agent.last_tested["p"])
    return testing

def false_pos_neg(drugs, criterion):
    false_positives = 0
    false_negatives = 0
    for drug in drugs:
        if drug["p"] > criterion and not drug["approved"]:
            false_negatives += 1
        elif drug["p"] < criterion and drug["approved"]:
            false_positives += 1
    return false_positives, false_negatives


def which_approved(drugs):
    dictionary = {}
    for drug in drugs:
        dictionary[drug["p"]] = drug["approved"]
    return dictionary

def success_v_testing_eval(drugs):
    l = list()
    for drug in drugs:
        success = drug["evaluation"]
        testing = drug["nr_studies"]
        p = drug["p"]
        l.append((p, success, testing))
    return l

def industry_fav(drugs):
    for drug in drugs:
        if drug["industry_fav"]:
            return drug["p"]

def run(pub_agents, ind_agents, bandits, run_id, placebo, placebo_site, inquiry_pub="myopic", inquiry_ind="myopic", style="normal",
        study_size_public=500, study_size_industry=100, pub_bias=(0,0), site_select_bias=(0,0),
        criterion=0, pulls=1000, weight=0, rounds=50, data_collection=1):
    with joblib.parallel_backend('sequential'):
        timer = 0
        convergence = 0
        drugs = list()
        agents = list()
        rows = list()
        success = [0, 0]

        for p in bandits:
            drug = {"p": p,
                    "industry_studies": list(),
                    "nr_industry_studies": 0,
                    "pub_studies": list(),
                    "studies": list(),
                    "pub_studies_collected": list(),
                    "ind_studies_collected": list(),
                    "nr_studies": 0,
                    "evaluation": 0,
                    "approved": False,
                    "industry_fav": False
                    }
            drugs.append(drug)
        max_p = max([drug["p"] for drug in drugs])

        id = 0
        for pub in range(pub_agents[1]):
            name = f"pub_{id}"
            # agents.append(BayesAgent(study_size_public, inquiry_pub, id))
            agents.append(pub_agents[0](study_size_public, inquiry_pub, pub_bias[0], site_select_bias[0], drugs, name))
            id += 1

        for ind in range(ind_agents[1]):
            name = f"ind_{id}"
            agents.append(ind_agents[0](study_size_industry, inquiry_ind, pub_bias[1], site_select_bias[1], drugs, name))
            id += 1

        for i in range(1, rounds+1):
            for agent in agents:
                agent.test(drugs, placebo, placebo_site, pulls, i, agents, success, weight)
            # Add timer if necessary
            evaluation(drugs, style, criterion, i)


            row = {"run_id": run_id,
                   "round": i,
                   "agents": (pub_agents[1], ind_agents[1]),
                   "bandits": bandits,
                   "placebo_site": placebo_site,
                   "public study size": study_size_public,
                   "industry study size": study_size_industry,
                   "pulls": pulls,
                   "pub_bias": pub_bias,
                   "site_bias": site_select_bias,
                   "style": style,
                   "inquiry_pub": inquiry_pub,
                   "inquiry_ind": inquiry_ind,
                   "weight": weight,
                   "criterion": criterion,
                   "industry_fav": industry_fav(drugs),
                   "which_approved": which_approved(drugs),
                   "convergence_approval": compute_convergence_approval(drugs, placebo),
                   "success_testing_comparison": success_v_testing_eval(drugs),
                   "active_hypotheses": active_hypotheses(agents),
                   "success": success[0],
                   "failure": success[1],}

            if i == 1 or (i % data_collection == 0):
                rows.append(row)

        return rows


# run((BiasedPublicAgent, 2), (IndustryAgent, 2),  [0.45, 0.51],
#      1, 0.5, 0.45, "softmax", "softmax_industry", "Solomon",
#     1000, 1000, [0, 0], [0, 0],  1, 1000, 5, 1000, 1)