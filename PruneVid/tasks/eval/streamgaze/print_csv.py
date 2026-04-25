"""Print 8-task accuracy in user-specified comma-separated order.

Usage:
    python -m tasks.eval.streamgaze.print_csv <save_path>

Reads <save_path>/all_results.json and emits one CSV line plus overall_micro.
"""

import json
import os
import sys


# User-specified column order. The eval dataset stores task_type without the .json
# suffix; for the future-action task it uses '_egtea' variant.
COLUMNS = [
    'past_gaze_sequence_matching',
    'past_non_fixated_object_identification',
    'past_object_transition_prediction',
    'past_scene_recall',
    'present_object_attribute_recognition',
    'present_object_identification_easy',
    'present_object_identification_hard',
    'present_future_action_prediction_egtea',
]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python -m tasks.eval.streamgaze.print_csv <save_path>",
              file=sys.stderr)
        sys.exit(1)
    save_path = sys.argv[1]
    fp = os.path.join(save_path, 'all_results.json')
    with open(fp, 'r') as f:
        data = json.load(f)
    acc_dict = data['acc_dict']

    cells, total_correct, total_n = [], 0, 0
    for col in COLUMNS:
        if col in acc_dict:
            correct, n = acc_dict[col]
            acc = correct / max(n, 1) * 100
            cells.append(f"{acc:.2f}")
            total_correct += correct
            total_n += n
        else:
            cells.append('NA')

    print(','.join(cells))
    if total_n > 0:
        print(f"overall_micro: {total_correct / total_n * 100:.2f}")


if __name__ == '__main__':
    main()
