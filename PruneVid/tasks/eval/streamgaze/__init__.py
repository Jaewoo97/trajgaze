import os
import re
import json

from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)


def check_ans(pred, gt):
    flag = False
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content and gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    if not any([c in pred_option for c in 'abcdefgABCDEFG']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


def save_results(result_list, save_path):
    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    overall_correct, overall_total = 0, 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / max(v[1], 1) * 100
        overall_correct += v[0]
        overall_total += v[1]
    final_res['Avg'] = overall_correct / max(overall_total, 1) * 100

    all_results = {"acc_dict": acc_dict, "result_list": result_list}
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')


def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None:
        return all_results['result_list']
    return None


def _parse_mmss(ts):
    # "MM:SS" -> seconds (int/float)
    parts = ts.strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unparseable time string: {ts!r}")


_RESPONSE_TIME_RE = re.compile(
    r'\[\s*(\d+:\d+(?::\d+)?)\s*-\s*(\d+:\d+(?::\d+)?)\s*\]'
)


def _parse_response_time(rt):
    m = _RESPONSE_TIME_RE.match(rt.strip())
    if not m:
        raise ValueError(f"Unparseable response_time: {rt!r}")
    return _parse_mmss(m.group(1)), _parse_mmss(m.group(2))


class StreamGazeEgteaDataset(EvalDataset):
    """StreamGaze_v2 EGTEA subset — 8 non-proactive tasks, 4-choice MC."""

    VIDEO_ROOT = '/home/yujin/dataset/StreamGaze_v2/videos/egtea/original'
    QA_ROOT = '/home/yujin/dataset/StreamGaze_v2/qa'

    TASK_FILES = [
        'past_gaze_sequence_matching.json',
        'past_non_fixated_object_identification.json',
        'past_object_transition_prediction.json',
        'past_scene_recall.json',
        'present_future_action_prediction_egtea.json',
        'present_object_attribute_recognition.json',
        'present_object_identification_easy.json',
        'present_object_identification_hard.json',
    ]

    FUTURE_TASK = 'present_future_action_prediction_egtea'

    def __init__(self, num_segments, test_ratio=None, max_samples=None):
        super().__init__(num_segments=num_segments, test_ratio=test_ratio)

        egtea_videos = set(os.listdir(self.VIDEO_ROOT))
        self.data_list = []

        for fname in self.TASK_FILES:
            task_type = fname[:-5]
            path = os.path.join(self.QA_ROOT, fname)
            with open(path, 'r') as f:
                data = json.load(f)

            for sample in data:
                video_file = sample.get('video_path')
                if video_file not in egtea_videos:
                    continue
                try:
                    rt_start, rt_end = _parse_response_time(sample['response_time'])
                except Exception as e:
                    print(f"[skip] {fname} {video_file}: {e}")
                    continue

                if task_type == self.FUTURE_TASK:
                    bound = (0.0, rt_start)
                else:
                    bound = (rt_start, rt_end)
                if bound[1] <= bound[0]:
                    # degenerate window; widen by 1 second to avoid division-by-zero in get_index
                    bound = (max(0.0, bound[0] - 0.5), bound[0] + 0.5)

                for q_idx, q in enumerate(sample['questions']):
                    self.data_list.append({
                        'task_type': task_type,
                        'prefix': self.VIDEO_ROOT,
                        'data_type': 'video',
                        'bound': bound,
                        'data': {
                            'question': q['question'],
                            'options': q['options'],
                            'answer': q['answer'],
                        },
                        'video_file': video_file,
                        'q_idx': q_idx,
                    })

        if max_samples is not None and max_samples > 0:
            # Keep at least one sample per task when truncating (smoke test).
            by_task = {}
            for d in self.data_list:
                by_task.setdefault(d['task_type'], []).append(d)
            picked = []
            for t in self.TASK_FILES:
                tt = t[:-5]
                if tt in by_task:
                    picked.append(by_task[tt][0])
            # Fill remainder in original order.
            seen = {id(x) for x in picked}
            for d in self.data_list:
                if len(picked) >= max_samples:
                    break
                if id(d) not in seen:
                    picked.append(d)
            self.data_list = picked[:max_samples]

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        question, answer = self.qa_template(entry['data'])
        task_type = entry['task_type']
        decord_method = self.decord_method[entry['data_type']]
        bound = entry['bound']
        video_path = os.path.join(entry['prefix'], entry['video_file'])

        images_group = decord_method(video_path, bound)

        return {
            'video_path': video_path,
            'video_pils': images_group,
            'question': question,
            'answer': answer,
            'task_type': task_type,
            'bound': bound,
            'q_idx': entry['q_idx'],
        }

    def qa_template(self, data):
        q = f"Question: {data['question']}\n"
        q += "Options:\n"
        answer_letter = str(data['answer']).strip().upper()
        answer_idx = ord(answer_letter) - ord('A')
        answer_content = None
        for idx, opt in enumerate(data['options']):
            # opt is "A. xxx" / "B. xxx" ... strip the letter prefix if present.
            m = re.match(r'^\s*([A-D])[\.\):]\s*(.+)$', opt)
            content = m.group(2).strip() if m else str(opt).strip()
            q += f"({chr(ord('A') + idx)}) {content}\n"
            if idx == answer_idx:
                answer_content = content
        q = q.rstrip()
        if answer_content is None:
            answer_content = ''
        answer = f"({answer_letter}) {answer_content}"
        return q, answer
