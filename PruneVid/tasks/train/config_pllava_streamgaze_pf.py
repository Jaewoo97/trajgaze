from tasks.train.instruction_data import *

# ========================= data ==========================
train_corpus = "streamgaze_noegtea"
train_file = "${available_corpus[${train_corpus}]}"
test_file = dict()
test_types = []
num_workers = 8
save_steps = 500
ckpt_steps = 500
stop_key = None
deepspeed = False

# ========================= input ==========================
num_frames = 64                       # PHASE A; switch to 16 for PHASE B
num_frames_test = 1
batch_size = 2                        # 64f bf16 with backward + LoRA, conservative for H200 single-GPU
gradient_accumulation_steps = 8       # effective batch 16
max_txt_l = 1024                      # 4-option MC + system + question text
max_train_steps = None
pre_text = False
inputs = dict(
    image_res=336,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model = dict(
    repo_id="MODELS/pllava-7b",
    # PLLaVA-7B safetensors has keys under `language_model.base_model.model.*` (saved
    # after the original LoRA wrap). from_pretrained() at this prefix returns NO
    # LLM weights (random init → zero logits). pretrained_path triggers the manual
    # post-LoRA-wrap load in setup_model that handles the prefix correctly.
    pretrained_path="MODELS/pllava-7b",
    load_from_origin=False,
    origin_vision="",
    origin_llm="",
    vision_encoder=dict(name="vit_l14"),
    torch_dtype='bfloat16',
    freeze_projector=True,            # ← projector frozen (vs sister: False)
    projector_unfreeze_modules=['all'],
    freeze_lm=True,                   # base LLM frozen — LoRA only
    lm_unfreeze_modules=['all'],
    freeze_vision_tower=True,         # CLIP frozen
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    num_frames="${num_frames}",
    pooling_method='avg',
    use_pooling=True,
    frame_shape=(24, 24),             # 336 / 14 = 24
    pooling_shape=(16, 12, 12),       # match zero-shot eval setting
    # PruneVid pruning fully OFF for training:
    #  - tau/cluster_ratio/temporal_segment_ratio = 1 disables merge_frames_dynamic.
    #  - selected_layer=99 disables LLM-VTP; eval-time padding-free input safely uses
    #    layer 10, but training right-padding makes the pad-token-based image-region
    #    detector in elastic_cache misidentify pad slots as image, so we skip VTP.
    selected_layer=99,
    alpha=0.1,
    head=8,
    softmax=1.0,
    tau=1.0,
    cluster_ratio=1.0,
    temporal_segment_ratio=1.0,
)

preprocess = dict(
    system=("Carefully watch the video and pay attention to the cause and "
            "sequence of events, the detail and movement of objects, and "
            "the action and pose of persons. Based on your observations, "
            "select the best option that accurately addresses the question.\n "),
    mm_alone=False,                   # image token lives in same USER turn as question
    random_shuffle=False,
    add_second_msg=False,             # disable "Video contains X frames..." auto-msg
    roles=['USER: ', 'ASSISTANT:'],   # match conv_eval_mvbench role spacing
    end_signal=(' ', '</s>'),
    begin_signal='',
    dataset_image_placeholder='<Image></Image>',
    dataset_video_placeholder='<Video></Video>',
    image_token_index=32000,
    max_txt_l="${max_txt_l}",
    ignore_index=-100,
    center_pad=False,
    longest_edge=762,
    shortest_edge=336,
    clip_transform=False,
    num_frames="${num_frames}",
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,                          # standard LoRA lr (base LM frozen)
    opt_betas=[0.9, 0.999],
    weight_decay=0.0,
    max_grad_norm=1.0,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(
    is_videochat2_custom=False,
    sched="cosine",
    epochs=6,                         # ← 6 epoch (vs sister: 2)
    warmup_ratio=0.05,
    min_lr_multi=0.1,
)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

fp16 = True
gradient_checkpointing = True

wandb = dict(enable=False, entity="user", project="streamgaze_lora")
dist_url = "env://"
device = "cuda"
mode = "it"

output_dir = "test_results/streamgaze_lora/pllava7b_r16_pf_64f"  # ← new dir
resume = False
debug = False
log_freq = 10
metric_window_size = 10
seed = 42
report_to = 'tensorboard'
save_latest = True
auto_resume = True
pretrained_path = ""
