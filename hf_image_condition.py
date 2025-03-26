import os
import torch
import random
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import transforms
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils_vis import render_example
from data_loader_jsonl import JSONLDataset
from data_loader_images import ImageFolderDataset
from data_loader_h5 import H5Dataset, PairedH5Dataset
from data_augmentations import augment_image_rgb, RandomizeBackgrounds, complexify_text, DepthAugmentation


def augment_suffix(suffix):
    parts = suffix.split(' ; ')
    random.shuffle(parts)
    return ' ; '.join(parts)

def get_collate_fn(processor, num_devices, num_images_in_conetxt, image_order, TORCH_DTYPE, DEVICE):
    if num_devices == 1:
        def collate_fn(batch):
            images, labels = zip(*batch)        # images will be lists of lists since one batch input has multiple images
            prefixes, suffixes = [], []
            for i in range(len(labels)):
                tmp_prefix = ""
                if image_order == "interleaved":
                    for j in range(num_images_in_context):
                        tmp_prefix += "<image>" + labels[i][j]["suffix"]
                    tmp_prefix += "<image>"
                else:
                    tmp_prefix = "<image>"*(num_images_in_context + 1)
                    for j in range(num_images_in_context):
                        tmp_prefix += labels[i][j]["suffix"]
                prefixes.append(tmp_prefix)
                suffixes.append(labels[i][-1]["suffix"])
            
            images_flat = [image for images_list in images for image in images_list]

            inputs = processor(
                text=prefixes,
                images=images_flat,
                return_tensors="pt",
                suffix=suffixes,
                padding="longest"
            ).to(TORCH_DTYPE).to(DEVICE)

            return inputs
    else:
        def collate_fn(batch):
            images, labels = zip(*batch)        # images will be lists of lists since one batch input has multiple images
            prefixes, suffixes = [], []
            for i in range(len(labels)):
                tmp_prefix = ""
                if image_order == "interleaved":
                    for j in range(num_images_in_context):
                        tmp_prefix += "<image>" + labels[i][j]["suffix"]
                    tmp_prefix += "<image>"
                else:
                    tmp_prefix = "<image>"*(num_images_in_context + 1)
                    for j in range(num_images_in_context):
                        tmp_prefix += labels[i][j]["suffix"]
                prefixes.append(tmp_prefix)
                suffixes.append(labels[i][-1]["suffix"])
            
            images_flat = [image for images_list in images for image in images_list]

            inputs = processor(
                text=prefixes,
                images=images_flat,
                return_tensors="pt",
                suffix=suffixes,
                padding="longest"
            ).to(TORCH_DTYPE)

            return inputs
    return collate_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images_in_context", type=int, default=1)
    parser.add_argument("--image_order", type=str, choices=["interleaved", "images_first"], default="interleaved")
    parser.add_argument("--extra_run_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_dev", type=int, default=2)

    return parser.parse_args()


# DATA COPY-PASTING AND CHECK
if not os.path.exists('/tmp/indoorCVPR') or not os.path.exists('/tmp/clevr-act-7-depth'):
    cmd1 = (
    "rsync -a --progress /data/lmbraid19/argusm/datasets/indoorCVPR.tar /tmp/ && "
    "mkdir -p /tmp/indoorCVPR && "
    "tar -xf /tmp/indoorCVPR.tar -C /tmp/indoorCVPR"
    )
    subprocess.run(cmd1, shell=True, check=True)

    # Command 2: Copy the second dataset directory
    cmd2 = "rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-act-7-depth /tmp/"
    subprocess.run(cmd2, shell=True, check=True)

    # Command 3: Check file type for /tmp/indoorCVPR
    cmd3 = "file /tmp/indoorCVPR"
    result1 = subprocess.run(cmd3, shell=True, check=True, capture_output=True, text=True)
    print(result1.stdout)

    # Command 4: Check file type for /tmp/clevr-act-7-depth
    cmd4 = "file /tmp/clevr-act-7-depth"
    result2 = subprocess.run(cmd4, shell=True, check=True, capture_output=True, text=True)
    print(result2.stdout)
    #!rsync -a --progress /data/lmbraid19/argusm/datasets/indoorCVPR.tar /tmp/ && mkdir -p /tmp/indoorCVPR && tar -xf /tmp/indoorCVPR.tar -C /tmp/indoorCVPR
    #!rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-act-7-depth /tmp/
    #!file /tmp/indoorCVPR
    #!file /tmp/clevr-act-7-depth
else:
    print('Data already exists')

# SETTING UP THE PATHS
model_location = Path("/data/lmbraid21/bratulic/max_pali/models")
dataset_location = Path("/tmp/clevr-act-7-depth")

args = get_args()

# SETTING UP THE DATASET
num_images_in_context = args.num_images_in_context
image_order = args.image_order
run_name = f"_imagesInContext_{num_images_in_context}_promptOrder_{image_order}"
load_presampled_pairs_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"train_dataset_{run_name}.pkl"
run_name += args.extra_run_name

bg_image_dataset = ImageFolderDataset("/tmp/indoorCVPR/Images", transform=transforms.RandomResizedCrop((448,448)))
randomize_background = RandomizeBackgrounds(p=0.2, background_images=bg_image_dataset)
augment_depth = DepthAugmentation(depth_range=(25, 100), max_delta_depth=35)

raw_dataset = H5Dataset(dataset_location, augment_rgbds=randomize_background, augment_rgb=augment_image_rgb, augment_text=complexify_text,
                          augment_depth=augment_depth, return_depth=False)
train_dataset = PairedH5Dataset(raw_dataset, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=load_presampled_pairs_path)

print("dataset_location:", dataset_location,"samples:", len(raw_dataset), "paired_samples:", len(train_dataset))


# SETTING UP THE MODEL
print("cuda visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])
devices_good = sorted((int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
DEVICE = torch.device('cuda')
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"], "DEVICE:", DEVICE, "devices_good:", devices_good, "num_devices:", len(devices_good))

TORCH_DTYPE = torch.bfloat16
SEQLEN = 12
MODEL_ID = "google/paligemma2-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, device_map="auto", attn_implementation='eager')

collate_fn = get_collate_fn(processor, len(devices_good), num_images_in_context, image_order, TORCH_DTYPE, DEVICE)

# SETTING UP THE TRAINER
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False
    
for name, param in model.named_parameters():
    if param.requires_grad == True:
        if "self_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

TRAIN_EXAMPLES = len(train_dataset)
BATCH_SIZE = args.batch_size
BATCH_SIZE_DEV = args.batch_size_dev # on l40 was 8
GRAD_ACCUM = int(round(BATCH_SIZE / BATCH_SIZE_DEV))
TRAIN_STEPS = (TRAIN_EXAMPLES // BATCH_SIZE)
SEQLEN = 12
SAVE_STEPS = int(TRAIN_STEPS / 15)
SAVE_LIMIT = 5

save_path = model_location / (str(Path(dataset_location).stem) + run_name)
print("save_path", save_path)
print("TRAIN_STEPS",TRAIN_STEPS)
print("GRAD_ACCUM", GRAD_ACCUM)

args_jax = Seq2SeqTrainingArguments(
    max_steps=TRAIN_STEPS,
    remove_unused_columns=False,
    per_device_train_batch_size=BATCH_SIZE_DEV,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=3e-5,  # 1e-5, 2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=.05,
    generation_max_length=SEQLEN,
    logging_steps=10,
    optim="adafactor",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    output_dir=save_path,
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False,
    dataloader_num_workers=4,
    #dataloader_prefetch_factor=2,
    #eval_strategy="steps",
    #eval_steps=4,
    #per_device_eval_batch_size=BATCH_SIZE_DEV,
    #eval_accumulation_steps=GRAD_ACCUM
)
#gradient_checkpointing=True,
#weight_decay=3e-7,

trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    #eval_dataset=train_dataset_small,
    data_collator=collate_fn,
    args=args_jax,
    #compute_metrics=compute_metrics
)

# TRAINING THE MODEL

trainer.train()