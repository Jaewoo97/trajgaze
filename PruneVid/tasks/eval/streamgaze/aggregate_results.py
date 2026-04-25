"""Aggregate StreamGaze EGTEA eval results into a markdown table.

Usage:
    python -m tasks.eval.streamgaze.aggregate_results \
        --run 16f=test_results/streamgaze_egtea/vanilla_16f \
        --run 32f=test_results/streamgaze_egtea/vanilla_32f \
        --run 64f=test_results/streamgaze_egtea/vanilla_64f
"""
import argparse
import json
import os


TASK_ORDER = [
    "past_gaze_sequence_matching",
    "past_non_fixated_object_identification",
    "past_object_transition_prediction",
    "past_scene_recall",
    "present_object_attribute_recognition",
    "present_object_identification_easy",
    "present_object_identification_hard",
    "present_future_action_prediction_egtea",
]


def load_run(path):
    fp = os.path.join(path, 'all_results.json')
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    with open(fp) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', default=[],
                    help='label=path, repeatable')
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, path = spec.split('=', 1)
        runs.append((label, load_run(path)))

    header = "| Task | N | " + " | ".join(f"{lbl} acc (%)" for lbl, _ in runs) + " |"
    sep = "|" + "---|" * (2 + len(runs))
    print(header)
    print(sep)

    overall_correct = {lbl: 0 for lbl, _ in runs}
    overall_total = {lbl: 0 for lbl, _ in runs}

    for task in TASK_ORDER:
        n_any = None
        cells = []
        for lbl, run in runs:
            acc_dict = run.get('acc_dict', {})
            correct, total = acc_dict.get(task, [0, 0])
            n_any = n_any or total
            acc = correct / total * 100 if total else float('nan')
            cells.append(f"{acc:.2f}" if total else "—")
            overall_correct[lbl] += correct
            overall_total[lbl] += total
        print(f"| {task} | {n_any or 0} | " + " | ".join(cells) + " |")

    print(sep)
    overall_cells = [
        f"{overall_correct[lbl] / max(overall_total[lbl], 1) * 100:.2f}"
        for lbl, _ in runs
    ]
    any_total = next(iter(overall_total.values()), 0)
    print(f"| **Overall (micro)** | {any_total} | " + " | ".join(overall_cells) + " |")


if __name__ == "__main__":
    main()
