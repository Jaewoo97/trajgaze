"""Convert StreamGaze_v2 qa JSONs (8 non-proactive tasks) into ITVidTrainDataset
format. EGTEA records are filtered out (held out for evaluation).

Output JSON list:
    [
      {"video": "<source>/original/<basename>.mp4",
       "QA":   [{"q": "<eval-aligned prompt>", "a": "(X) <content>"}],
       "start": <float seconds>, "end": <float seconds>},
      ...
    ]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter


TASK_FILES = [
    'past_gaze_sequence_matching.json',
    'past_non_fixated_object_identification.json',
    'past_object_transition_prediction.json',
    'past_scene_recall.json',
    'present_object_attribute_recognition.json',
    'present_object_identification_easy.json',
    'present_object_identification_hard.json',
    'present_future_action_prediction.json',  # mother file (EGTEA filtered out below)
]

FUTURE_TASK_BASENAME = 'present_future_action_prediction'

_RESPONSE_TIME_RE = re.compile(
    r'\[\s*(\d+:\d+(?::\d+)?)\s*-\s*(\d+:\d+(?::\d+)?)\s*\]'
)


def _parse_mmss(ts):
    parts = ts.strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unparseable time string: {ts!r}")


def _parse_response_time(rt):
    m = _RESPONSE_TIME_RE.match(rt.strip())
    if not m:
        raise ValueError(f"Unparseable response_time: {rt!r}")
    return _parse_mmss(m.group(1)), _parse_mmss(m.group(2))


def _route_video(video_path, videos_root):
    """Return relative path under videos_root, locating the file by trying each
    non-EGTEA subdir. EGTEA records are filtered upstream."""
    base = os.path.basename(video_path)
    if base.startswith('OP'):
        return None  # EGTEA, held out for eval
    for sub in ('holoassist', 'egoexolearn'):
        rel = os.path.join(sub, 'original', base)
        if os.path.exists(os.path.join(videos_root, rel)):
            return rel
    return None


def _qa_template(question_dict):
    """Mirror StreamGazeEgteaDataset.qa_template — returns (q_text, answer_letter, answer_content)."""
    q = f"Question: {question_dict['question']}\n"
    q += "Options:\n"
    answer_letter = str(question_dict['answer']).strip().upper()
    answer_idx = ord(answer_letter) - ord('A')
    answer_content = None
    for idx, opt in enumerate(question_dict['options']):
        m = re.match(r'^\s*([A-D])[\.\):]\s*(.+)$', opt)
        content = m.group(2).strip() if m else str(opt).strip()
        q += f"({chr(ord('A') + idx)}) {content}\n"
        if idx == answer_idx:
            answer_content = content
    q = q.rstrip()
    if answer_content is None:
        answer_content = ''
    return q, answer_letter, answer_content


def build(qa_dir, videos_root, out_path):
    records = []
    counters = {'sources': Counter(), 'tasks': Counter(), 'skipped_egtea': 0,
                'skipped_missing_video': 0, 'skipped_bad_rt': 0,
                'skipped_degenerate_window': 0}

    for fname in TASK_FILES:
        task_type = fname[:-5]
        path = os.path.join(qa_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        for sample in data:
            video_path = sample.get('video_path')
            if video_path is None:
                continue
            base = os.path.basename(video_path)
            if base.startswith('OP'):
                counters['skipped_egtea'] += 1
                continue

            rel = _route_video(video_path, videos_root)
            if rel is None:
                counters['skipped_missing_video'] += 1
                continue

            try:
                rt_start, rt_end = _parse_response_time(sample['response_time'])
            except Exception:
                counters['skipped_bad_rt'] += 1
                continue

            if task_type == FUTURE_TASK_BASENAME:
                start, end = 0.0, rt_start
            else:
                start, end = rt_start, rt_end
            if end <= start:
                counters['skipped_degenerate_window'] += 1
                start, end = max(0.0, start - 0.5), start + 0.5

            for q in sample.get('questions', []):
                try:
                    q_text, answer_letter, answer_content = _qa_template(q)
                except Exception:
                    continue

                # Eval-aligned prompt: include the trailing "Only give the best option."
                # so training prefix matches `infer_mvbench` post_query_prompt.
                q_field = f"{q_text}\nOnly give the best option."
                a_field = f"({answer_letter}) {answer_content}"

                records.append({
                    'video': rel,
                    'QA': [{'q': q_field, 'a': a_field}],
                    'start': float(start),
                    'end': float(end),
                })

                if rel.startswith('holoassist'):
                    counters['sources']['holoassist'] += 1
                elif rel.startswith('egoexolearn'):
                    counters['sources']['egoexolearn'] += 1
                counters['tasks'][task_type] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {out_path}")
    print(f"  by source: {dict(counters['sources'])}")
    print(f"  by task:   {dict(counters['tasks'])}")
    print(f"  skipped: egtea={counters['skipped_egtea']} "
          f"missing_video={counters['skipped_missing_video']} "
          f"bad_rt={counters['skipped_bad_rt']} "
          f"degenerate={counters['skipped_degenerate_window']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qa_dir', default='/home/yujin/dataset/StreamGaze_v2/qa')
    p.add_argument('--videos_root', default='/home/yujin/dataset/StreamGaze_v2/videos')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    build(args.qa_dir, args.videos_root, args.out)


if __name__ == '__main__':
    main()
