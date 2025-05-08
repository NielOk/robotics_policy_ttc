import json
import os

POLICY_EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(POLICY_EVALS_DIR, "results")

def load_clean_results(results_path):
    """
    Load results from JSON
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    cleaned_results = {}
    for epoch_id in results.keys():
        epoch_results = results[epoch_id]
        all_scores = epoch_results["all_scores"]
        cleaned_all_scores = [score for score in all_scores if score != 0.0]

        cleaned_mean = sum(cleaned_all_scores) / len(cleaned_all_scores) if cleaned_all_scores else 0.0
        cleaned_std = (sum((x - cleaned_mean) ** 2 for x in cleaned_all_scores) / len(cleaned_all_scores)) ** 0.5 if cleaned_all_scores else 0.0

        cleaned_results[epoch_id] = {
            "mean": cleaned_mean,
            "std": cleaned_std,
            "all_scores": cleaned_all_scores
        }

    return cleaned_results

def main():
    results_path = os.path.join(RESULTS_DIR, "state_policy_eval_results.json")
    cleaned_results = load_clean_results(results_path)

    print(cleaned_results)

if __name__ == '__main__':
    main()