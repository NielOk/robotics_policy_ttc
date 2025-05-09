import json
import os
import matplotlib.pyplot as plt

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
        cleaned_std = (
            (sum((x - cleaned_mean) ** 2 for x in cleaned_all_scores) / len(cleaned_all_scores)) ** 0.5
            if cleaned_all_scores else 0.0
        )

        cleaned_results[epoch_id] = {
            "mean": cleaned_mean,
            "std": cleaned_std,
            "all_scores": cleaned_all_scores
        }

    return cleaned_results

def plot_results(cleaned_results):
    epochs = sorted(cleaned_results.keys(), key=lambda x: int(x))
    means = [cleaned_results[ep]["mean"] for ep in epochs]
    stds = [cleaned_results[ep]["std"] for ep in epochs]
    x = list(range(len(epochs)))

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, label='Mean ± Std Dev', color='steelblue')
    plt.xticks(x, epochs, rotation=45)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Results per Epoch')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    results_path = os.path.join(RESULTS_DIR, "state_policy_eval_results.json")
    cleaned_results = load_clean_results(results_path)
    plot_results(cleaned_results)

if __name__ == '__main__':
    main()