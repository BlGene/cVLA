import os
import re
import torch
import random
import argparse
import subprocess
import numpy as np

from pathlib import Path
from datetime import datetime
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig, AutoProcessor, AutoModelForCausalLM

from data_loader_images import ImageFolderDataset
from data_loader_h5 import H5Dataset
from data_loader_paired import PairedDataset
from utils_traj_tokens import TrajectoryEncoder_xyzrotvec2
from utils_eval import Evaluator
from data_augmentations import augment_image_rgb, RandomizeBackgrounds, complexify_text


def extract_tokens(text):
    """
    Extracts loc and seg tokens as a list, e.g., ["<locxxx>", "<locyyy>", "<segxxx>", ...]
    """
    return re.findall(r"<loc\d+>|<seg\d+>", text)


def augment_suffix(suffix):
    parts = suffix.split(' ; ')
    random.shuffle(parts)
    return ' ; '.join(parts)


def get_robot_state(prefix):
    pattern = r"((?:<loc\d+>|<seg\d+>)+)"
    match = re.search(pattern, prefix)
    if match:
        extracted = match.group(1)
        return extracted
    return ""


def get_collate_fn(processor, num_images_in_context, image_order, TORCH_DTYPE, args, DEVICE):
    def collate_fn(batch):
        images, labels = zip(*batch)        # images will be lists of lists since one batch input has multiple images
        if args.conditioning == "trajectory":
            prefixes, suffixes = [], []
            for i in range(len(labels)):
                tmp_prefix = ""
                if image_order == "interleaved":
                    for j in range(num_images_in_context):
                        tmp_prefix += "<image>" + get_robot_state(labels[i][j]["prefix"]) + " " + labels[i][j]["suffix"]
                    tmp_prefix += "<image>"
                    tmp_prefix += get_robot_state(labels[i][-1]["prefix"]) + " "
                else:
                    tmp_prefix = "<image>"*(num_images_in_context + 1)
                    for j in range(num_images_in_context):
                        tmp_prefix += labels[i][j]["suffix"]

                prefixes.append(tmp_prefix)
                suffixes.append(labels[i][-1]["suffix"])
            
            images_flat = [image for images_list in images for image in images_list]
        else:
            prefixes = ["<image>" + label["prefix"] for label in labels]
            suffixes = [label["suffix"] for label in labels]
            images_flat = images


        inputs = processor(
            text=prefixes,
            images=images_flat,
            return_tensors="pt",
            suffix=suffixes,
            padding="longest",

        ).to(TORCH_DTYPE)

        return inputs
    return collate_fn


def get_compute_metrics_fn(processor, max_tokens, eval_dummy_camera, action_encoder):
    eval_dummy_camera.extrinsic_matrix = torch.tensor([[[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]])
    evaluator = Evaluator(action_encoder, eval_dummy_camera)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred    
        metric = []
        whole_text = []
        for i in range(len(predictions)):
            prefix_len = sum(labels[i] == -100) # first x tokens are -100, end is generated
            decoded_preds = processor.decode(predictions[i, labels[i].shape[0]:], skip_special_tokens=True)
            decoded_labels = processor.decode(labels[i, prefix_len:], skip_special_tokens=True)
            num_tokens_in_pred = len(extract_tokens(decoded_preds))
            num_tokens_in_label = len(extract_tokens(decoded_labels)) 
            if num_tokens_in_pred == num_tokens_in_label:
                metric.append(1)
            else:
                metric.append(0)
            if len(decoded_preds) == len(decoded_labels):
                whole_text.append(1)
            else:
                whole_text.append(0)

            evaluator.evaluate(decoded_preds, decoded_labels)

        final_metrics = evaluator.report_stats()
        final_metrics["valid_samples_ratio"] = np.sum(metric) / len(metric)
        final_metrics["whole_text_ratio"] = np.sum(whole_text) / len(whole_text)
        evaluator.reset()

        return final_metrics
    return compute_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/paligemma2-3b-pt-224")
    parser.add_argument("--conditioning", type=str, choices=["image", "text", "trajectory"], default="trajectory")
    parser.add_argument("--num_images_in_context", type=int, default=1)
    parser.add_argument("--image_order", type=str, choices=["interleaved", "images_first"], default="interleaved")
    parser.add_argument("--extra_run_name", type=str, default="debug")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_dev", type=int, default=3)
    parser.add_argument("--p_background", type=float, default=0.2)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_augs", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=13, help="Max tokens for generation (basically sequence length)")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--p_copy", type=float, default=0.0, help="Percentage of pairs with direct copy of images in context")
    parser.add_argument("--apply_copy_augs", action="store_true", help="Apply augmentations to the copy of the images in context")
    parser.add_argument("--p_sort_by_l2_distance", type=float, default=0.0, help="Sort the images in context by L2 distance to the query image for some percentage")
    parser.add_argument("--save_steps", type=int, default=350)
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="/work/dlclarge2/bratulic-cvla/models")
    parser.add_argument("--no_eval", action="store_true", help="Do not evaluate the model")
    parser.add_argument("--encoder", type=str, default="xyzrotvec2", help="Encoder to use for the model")


    return parser.parse_args()


def get_model(model_type="google/paligemma2-3b-pt-224", TORCH_DTYPE=torch.bfloat16, DEVICE=torch.device('cuda'), checkpoint=None):
    if "paligemma" not in model_type:
        processor = AutoProcessor.from_pretrained(model_type)
        if checkpoint is not None:
            model_type = checkpoint
        model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=TORCH_DTYPE, device_map="auto", attn_implementation='eager')
    else:
        processor = PaliGemmaProcessor.from_pretrained(model_type, use_fast=True)
        if checkpoint is not None:
            model_type = checkpoint
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_type, torch_dtype=TORCH_DTYPE, device_map="auto", attn_implementation='eager')
    return processor, model


def save_hyperparams(save_path_final, save_path, args):
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    if not save_path_final.exists():
        save_path_final.mkdir(parents=True)

    # save command line arguments in nice format
    with open(save_path_final / "args.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    # save the script - just copy paste it to the save_path with os
    os.system(f"cp /home/bratulic/git_repos/robo/cVLA/hf_image_condition.py {save_path_final}/hf_image_condition.py")


def get_trainer(args, model, processor, train_dataset, eval_dataset, collate_fn, save_path, eval_dummy_camera, action_encoder):
    # FT ONLY THE SELF-ATTENTION LAYERS
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

    TRAIN_EXAMPLES = len(train_dataset) * args.num_repeats
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_DEV = args.batch_size_dev # on l40 was 8
    GRAD_ACCUM = int(round(BATCH_SIZE / BATCH_SIZE_DEV))
    TRAIN_STEPS = (TRAIN_EXAMPLES // BATCH_SIZE)
    SEQLEN = args.max_tokens
    SAVE_STEPS = args.save_steps
    SAVE_LIMIT = args.save_limit
    
    generation_config = GenerationConfig(
        max_new_tokens=SEQLEN,
        min_new_tokens=SEQLEN,
        num_beams=1,
        use_cache=False,
    )
    if args.no_eval:
        eval_strategy = "no"
    else:
        eval_strategy = "steps"

    args_jax = Seq2SeqTrainingArguments(
        max_steps=TRAIN_STEPS,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE_DEV,
        per_device_eval_batch_size=BATCH_SIZE_DEV,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,  # 1e-5, 2e-5,
        lr_scheduler_type="cosine",
        # generation_max_length=SEQLEN,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        optim="adafactor",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMIT,
        output_dir=save_path,
        save_safetensors=True,  # Optimized saving format
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # Add evaluation-related settings
        eval_strategy=eval_strategy,  # or "epoch"
        eval_steps= SAVE_STEPS,        # how often to evaluate
        predict_with_generate=True,   # important for generation tasks
        generation_config=generation_config,
        do_eval=not args.no_eval,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        args=args_jax,
        compute_metrics=get_compute_metrics_fn(processor, SEQLEN, eval_dummy_camera, action_encoder),
    )

    trainer.model.config.use_cache = False

    return trainer


def get_datasets(args, dataset_location):
    # SETTING UP THE DATASET
    num_images_in_context = args.num_images_in_context
    image_order = args.image_order
    run_name = f"_imagesInContext_{num_images_in_context}_promptOrder_{image_order}"
    load_presampled_pairs_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"train_dataset_{run_name}_new.pkl"
    run_name += f"maxTokens{args.max_tokens}_lr{args.lr}" + args.extra_run_name

    bg_image_dataset = ImageFolderDataset("/tmp/indoorCVPR/Images", transform=transforms.RandomResizedCrop((448,448)))
    randomize_background = RandomizeBackgrounds(p=args.p_background, background_images=bg_image_dataset)
    forced_image_augs = RandomizeBackgrounds(p=1.0, background_images=bg_image_dataset)
    if args.no_augs:
        raw_dataset = H5Dataset(dataset_location, augment_rgbds=None, augment_rgb=None, augment_text=None, augment_depth=None, return_depth=False, 
                                augment_rgb_forced=forced_image_augs, action_encoder=args.encoder)
    else:
        raw_dataset = H5Dataset(dataset_location, augment_rgbds=randomize_background, augment_rgb=augment_image_rgb, augment_text=None,
                                augment_depth=None, return_depth=False, augment_rgb_forced=forced_image_augs, action_encoder=args.encoder)
        
    if args.conditioning == "trajectory":
        train_dataset = PairedDataset(raw_dataset, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=load_presampled_pairs_path,
                                    mode="train", p_copy=args.p_copy, apply_copy_augs=args.apply_copy_augs, p_sort_by_l2_distance=args.p_sort_by_l2_distance)

        eval_dataset = PairedDataset(raw_dataset, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=load_presampled_pairs_path,
                                    mode="test", p_copy=args.p_copy, apply_copy_augs=args.apply_copy_augs, p_sort_by_l2_distance=args.p_sort_by_l2_distance)
        eval_dummy_camera = eval_dataset[0][1][0]["camera"]
    elif args.conditioning == "text":
        train_dataset = raw_dataset
        eval_dataset = H5Dataset(dataset_location, augment_rgbds=None, augment_rgb=None, augment_text=None, augment_depth=None, return_depth=False, limit_samples=200)
        eval_dummy_camera = eval_dataset[0][1]["camera"]

    print("dataset_location:", dataset_location,"samples:", len(raw_dataset), "paired_samples:", len(train_dataset))

    return train_dataset, eval_dataset, run_name, eval_dummy_camera, raw_dataset.action_encoder


if __name__ == "__main__":
    
    # SETTING UP THE PATHS AND ARGS
    dataset_location = Path("/tmp/clevr-act-7-depth")
    current_time =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args = get_args()

    if not os.path.exists('/tmp/indoorCVPR') or not os.path.exists('/tmp/clevr-act-7-depth'):
        cmd1 = (
        "rsync -a --progress /work/dlclarge2/bratulic-cvla/indoorCVPR_09.tar /tmp/ && "
        "mkdir -p /tmp/indoorCVPR && "
        "tar -xf /tmp/indoorCVPR_09.tar -C /tmp/indoorCVPR"
        )
        subprocess.run(cmd1, shell=True, check=True)

        # Command 2: Copy the second dataset directory
        cmd2 = "rsync -a --progress /work/dlclarge2/bratulic-cvla/clevr-act-7-depth /tmp/"
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

    # SETTING UP THE DATASETS
    train_dataset, eval_dataset, run_name, eval_dummy_camera, action_encoder = get_datasets(args, dataset_location)
    
    # SETTING UP THE SAVE PATHS
    save_path_final = Path(args.save_path) / (str(Path(dataset_location).stem) + run_name + "_" + current_time)
    save_path = Path("/tmp/bratulic_cvla") / (str(Path(dataset_location).stem) + run_name + "_" + current_time)
    save_hyperparams(save_path_final, save_path, args)

    # SETTING UP THE MODEL
    DEVICE, TORCH_DTYPE = torch.device('cuda'), torch.bfloat16
    processor, model = get_model(model_type=args.model_id, TORCH_DTYPE=TORCH_DTYPE, DEVICE=DEVICE, checkpoint=args.checkpoint)
    collate_fn = get_collate_fn(processor, args.num_images_in_context, args.image_order, TORCH_DTYPE, args, DEVICE)

    # SETTING UP THE TRAINER
    trainer = get_trainer(args, model, processor, train_dataset, eval_dataset, collate_fn, save_path, eval_dummy_camera, action_encoder)
    
    # TRAINING THE MODEL
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted")
    
    # TRANSFER THE MODEL TO FINAL LOCATION
    os.system(f"mv {save_path}/* {save_path_final}/")