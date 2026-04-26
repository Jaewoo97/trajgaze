import functools
import itertools
import logging
import os
from argparse import ArgumentParser
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
import torch
import transformers
from tqdm import tqdm

from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from tasks.eval.streamgaze import (
    StreamGazeEgteaDataset,
    check_ans,
    save_results,
    load_results,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank. Default 128 matches the HF release adapter.")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated module names that LoRA wraps.")
    parser.add_argument("--weight_dir", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default='eval_mvbench')
    parser.add_argument("--pooling_shape", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--softmax", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--temporal_segment_ratio", type=float, default=1.0)
    parser.add_argument("--cluster_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Truncate dataset for smoke testing. Default: full 526 QA.")
    parser.add_argument("--multiprocess", type=int, default=1,
                        help="1 to shard across all visible GPUs (default), 0 for single-process debug.")
    parser.add_argument("--answer_prompt", type=str, default="Best option:(",
                        help="Forced ASSISTANT prefix for generation. "
                             "Pass 'none' to disable (e.g. for LoRA models trained "
                             "to emit '(X) ...' from a bare ASSISTANT: turn).")
    parser.add_argument("--return_prompt", type=str, default="(",
                        help="String prepended to model output before parsing. "
                             "Pass 'none' to disable.")
    return parser.parse_args()


def load_model_and_dataset(rank, world_size, args, pooling_shape):
    lora_target_modules = tuple(
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    )
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        pooling_shape=pooling_shape,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        softmax=args.softmax,
        head=args.head,
        tau=args.tau,
        cluster_ratio=args.cluster_ratio,
        temporal_segment_ratio=args.temporal_segment_ratio,
    )
    logger.info('done loading pllava')

    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = StreamGazeEgteaDataset(
        num_segments=args.num_frames, max_samples=args.max_samples,
    )
    dataset.set_rank_and_world_size(rank, world_size)
    if rank == 0:
        logger.info(f'dataset size (this shard): {len(dataset)}')
    return model, processor, dataset


def infer_mvbench(args, model, processor, data_sample, conv_mode,
                  pre_query_prompt=None, post_query_prompt=None,
                  answer_prompt=None, return_prompt=None, print_res=False):
    video_list = data_sample["video_pils"]
    conv = conv_templates[conv_mode].copy()
    conv.user_query(data_sample['question'], pre_query_prompt, post_query_prompt, is_mm=True)
    if answer_prompt is not None:
        conv.assistant_response(answer_prompt)

    llm_message, conv = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        img_list=video_list,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        print_res=print_res,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    if answer_prompt is not None:
        llm_message = ''.join(llm_message.split(answer_prompt)[1:])
    if return_prompt is not None:
        llm_message = return_prompt + llm_message

    return llm_message


def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    pooling_shape = (16, 12, 12)
    if args.pooling_shape is not None:
        pooling_shape = tuple(int(x) for x in args.pooling_shape.split("-"))

    logger.info(f'loading model + dataset on gpu {rank}')
    model, processor, dataset = load_model_and_dataset(rank, world_size, args, pooling_shape)

    conv_mode = args.conv_mode
    pre_query_prompt = None
    post_query_prompt = "\nOnly give the best option."

    if rank == 0:
        tbar = tqdm(total=len(dataset))
        # Sanity-log the first sample's window (especially for future-action task).
        if len(dataset) > 0:
            first = dataset.data_list[0]
            logger.info(
                f"[window-probe] task={first['task_type']} video={first['video_file']} "
                f"bound={first['bound']}"
            )

    correct, total = 0, 0
    result_list = []
    acc_dict = {}
    done_count = 0

    for example in dataset:
        task_type = example['task_type']
        acc_dict.setdefault(task_type, [0, 0])
        acc_dict[task_type][1] += 1
        total += 1

        answer_prompt = None if args.answer_prompt.lower() == 'none' else args.answer_prompt
        return_prompt = None if args.return_prompt.lower() == 'none' else args.return_prompt

        try:
            pred = infer_mvbench(
                args, model, processor, example,
                conv_mode=conv_mode,
                pre_query_prompt=pre_query_prompt,
                post_query_prompt=post_query_prompt,
                answer_prompt=answer_prompt,
                return_prompt=return_prompt,
                print_res=False,
            )
        except Exception as e:
            logger.exception(f"inference failed: {example['video_path']}")
            pred = f"(ERROR) {e}"

        gt = example['answer']
        result_list.append({
            'pred': pred,
            'gt': gt,
            'task_type': task_type,
            'video_path': example['video_path'],
            'question': example['question'],
            'bound': example.get('bound'),
            'q_idx': example.get('q_idx', 0),
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

        if rank == 0:
            tbar.update(len(result_list) - done_count)
            tbar.set_description_str(
                f"{task_type}: "
                f"{acc_dict[task_type][0]}/{acc_dict[task_type][1]} "
                f"({acc_dict[task_type][0] / max(acc_dict[task_type][1],1) * 100:.1f}%) "
                f"| overall {correct}/{total} ({correct / max(total,1) * 100:.1f}%)"
            )
            done_count = len(result_list)
    return result_list


def main():
    args = parse_args()
    save_path = args.save_path

    cached = load_results(save_path)
    if cached is not None:
        logger.info(f'reusing cached results at {save_path}')
        save_results(cached, save_path)
        return

    use_multiprocess = bool(args.multiprocess) and torch.cuda.device_count() > 1
    if use_multiprocess:
        mp.set_start_method('spawn', force=True)
        world_size = torch.cuda.device_count()
        logger.info(f'benchmarking on {world_size} GPUs → {save_path}')
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))
        result_list = [r for r in itertools.chain(*result_lists)]
    else:
        result_list = run(0, args=args, world_size=1)

    save_results(result_list, save_path)


if __name__ == "__main__":
    main()
