"""
Compute evaluation metrics on saved prediction JSON files.

Metrics:
  - ROUGE-L (recall-oriented, good for open-ended QA)
  - Token F1 (word overlap)

Usage:
  python eval/evaluate_results.py \
      --pred results/without_autogaze.json \
      --pred results/with_autogaze.json
"""

import argparse
import json
import re
from collections import Counter


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_file(path: str) -> dict:
    with open(path) as f:
        results = json.load(f)

    f1_scores     = []
    rouge_scores  = []
    empty_preds   = 0

    for r in results:
        pred  = normalize(r["prediction"])
        gold  = normalize(r["answer"])

        if not pred:
            empty_preds += 1
            f1_scores.append(0.0)
            rouge_scores.append(0.0)
            continue

        f1_scores.append(token_f1(pred, gold))
        rouge_scores.append(rouge_l(pred, gold))

    n = len(results)
    return {
        "n_samples":    n,
        "empty_preds":  empty_preds,
        "token_f1":     sum(f1_scores) / n if n else 0,
        "rouge_l":      sum(rouge_scores) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True, help="One or more result JSON files")
    args = parser.parse_args()

    print(f"{'File':<45} {'N':>6} {'Empty':>6} {'Token-F1':>10} {'ROUGE-L':>10}")
    print("-" * 82)
    for path in args.pred:
        metrics = evaluate_file(path)
        name = path.split("/")[-1].replace(".json", "")
        print(
            f"{name:<45} "
            f"{metrics['n_samples']:>6} "
            f"{metrics['empty_preds']:>6} "
            f"{metrics['token_f1']:>10.4f} "
            f"{metrics['rouge_l']:>10.4f}"
        )


if __name__ == "__main__":
    main()
