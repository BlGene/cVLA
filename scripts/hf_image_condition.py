import os
import re
import json
import torch
import random
import getpass
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig, AutoProcessor, AutoModelForCausalLM

from cvla.utils_eval import Evaluator
from cvla.data_loader_h5 import H5Dataset
from cvla.data_loader_jsonl import JSONLDataset
from cvla.data_loader_images import ImageFolderDataset
from cvla.data_loader_paired import PairedDataset
from cvla.data_augmentations import CleanText
from cvla.data_augmentations import augment_image_rgb, RandomizeBackgrounds, complexify_text


class MultiEvalSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **kwargs):
        original_compute_metrics = self.compute_metrics  # Save original

        # Evaluate on synthetic (default) dataset
        self.compute_metrics = self.compute_metrics_sim
        metrics_synth = super().evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix="eval_data_sim",
            **kwargs,
        )

        metrics_real = {}
        if hasattr(self, "eval_dataset_real") and self.eval_dataset_real is not None:
            self.compute_metrics = self.compute_metrics_real
            metrics_real = super().evaluate(
                eval_dataset=self.eval_dataset_real,
                metric_key_prefix="eval_data_real",
                **kwargs,
            )

        # restore
        self.compute_metrics = original_compute_metrics

        # merge and return
        merged = {**metrics_synth, **metrics_real}
        return merged


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
        
        if args.conditioning == "trajectory":
            images, labels, depths = zip(*batch)        # images will be lists of lists since one batch input has multiple images
            #if args.depth:
            #    raise NotImplementedError("Depth not implemented for trajectory conditioning")
            prefixes, suffixes = [], []
            for i in range(len(labels)):
                tmp_prefix = ""
                if image_order == "interleaved":
                    for j in range(num_images_in_context):
                        if args.depth_on_both:
                            tmp_prefix += "<image>"
                        tmp_prefix += "<image>" + get_robot_state(labels[i][j]["prefix"]) + " " + labels[i][j]["suffix"]
                    if args.depth or args.depth_on_both:
                        tmp_prefix += "<image>"
                    tmp_prefix += "<image>"
                    tmp_prefix += get_robot_state(labels[i][-1]["prefix"]) + " "
                else:
                    if args.depth_on_both:
                        tmp_prefix = "<image><image>" * (num_images_in_context + 1)
                    else:
                        tmp_prefix = "<image>"*(num_images_in_context + 1)
                        if args.depth:
                            tmp_prefix += "<image>"
                    for j in range(num_images_in_context):
                        tmp_prefix += labels[i][j]["suffix"]

                prefixes.append(tmp_prefix)
                suffixes.append(labels[i][-1]["suffix"])
            # if depth is added, we should have from first tuple all but last, then last depth and then last image in a tuple
            # ASSUMES INTERLEAVED ORDER!
            if args.depth or args.depth_on_both:
                images_flat = []
                for images_tuple, depth_tuple in zip(images, depths):
                    if args.depth_on_both:
                        for img, depth in zip(images_tuple, depth_tuple):
                            images_flat.append(depth)
                            images_flat.append(img)
                    else:
                        images_flat.extend(images_tuple[:-1])
                        images_flat.append(depth_tuple[-1])
                        images_flat.append(images_tuple[-1])
            else:
                images_flat = [image for images_list in images for image in images_list]
        else:
            images, labels = zip(*batch)        # images will be lists of lists since one batch input has multiple images
            if not args.depth:    
                prefixes = ["<image>" + label["prefix"] for label in labels]
                suffixes = [label["suffix"] for label in labels]
                images_flat = images
            else:
                assert np.all([len(x) == 2 for x in images])
                prefixes = ["<image><image>" + label["prefix"] for label in labels]
                suffixes = [label["suffix"] for label in labels]
                images_flat = [img for img_list_x in images for img in img_list_x]


        inputs = processor(
            text=prefixes,
            images=images_flat,
            return_tensors="pt",
            suffix=suffixes,
            padding="longest",

        ).to(TORCH_DTYPE)

        return inputs
    return collate_fn


def get_compute_metrics_fn(processor, max_tokens, camera_fixed, action_encoder, action_encoder_labels=None):
    # Warning: this code assumes that the camera and robot pose are fixed, need to pass the eval datasets here and get real values
    # TODO(max): pass eval dataset
    # robot pose is only needed for action in robot coords, where the robot is not located at origin (i.e. simulation)
    camera_fixed.extrinsic_matrix = torch.tensor([[[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]])
    from mani_skill.utils.structs import Pose
    robot_pose = Pose(torch.tensor([[-6.1500e-01,  7.2760e-11, -1.4901e-08,  1.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00]]))
    evaluator = Evaluator(action_encoder, camera_fixed, encoder_labels=action_encoder_labels, robot_pose_fixed=robot_pose)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred    
        metric = []
        whole_text = []
        for i in range(len(predictions)):
            tmp_labels, tmp_preds = labels[i], predictions[i]
            valid_mask = tmp_labels != -100  # -100 tokens are not valid (either padding or ignored tokens)
            first_valid_idx = np.argmax(valid_mask) # first index that is valid - determines the start of the generated text
            
            labels_to_decode = tmp_labels[valid_mask]
            decoded_labels = processor.decode(labels_to_decode, skip_special_tokens=True)

            if len(tmp_preds) - first_valid_idx >= max_tokens:
                preds_to_decode = tmp_preds[first_valid_idx + max_tokens:] # take only the part of the prediction that is valid
                preds_to_decode[preds_to_decode == -100] = processor.tokenizer.pad_token_id # if -100 is present, replace it with pad token id
                decoded_preds = processor.decode(preds_to_decode, skip_special_tokens=True)
            else:
                preds_to_decode = []
                decoded_preds = ""

            # start statistics
            num_tokens_in_pred = len(extract_tokens(decoded_preds))
            num_tokens_in_label = len(extract_tokens(decoded_labels))
            metric.append(int(num_tokens_in_pred == num_tokens_in_label))
            whole_text.append(int(len(decoded_preds) == len(decoded_labels)))
            evaluator.evaluate(decoded_preds, decoded_labels)

        final_metrics = evaluator.report_stats()
        final_metrics["valid_samples_ratio"] = np.sum(metric) / len(metric)
        final_metrics["whole_text_ratio"] = np.sum(whole_text) / len(whole_text)
        evaluator.reset()

        return final_metrics
    return compute_metrics


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
    os.system(f"cp {__file__} {save_path_final}/hf_image_condition.py")

    with open(save_path_final / "cvla_info.json","w") as f_obj:
        clva_info = dict(return_depth=args.depth, action_encoder=args.action_encoder)
        json.dump(clva_info, f_obj)


def get_trainer(args, model, processor, train_dataset, eval_sim_dataset, eval_real_dataset, collate_fn, save_path, eval_sim_dummy_camera, 
                eval_real_dummy_camera, action_encoder, eval_sim_action_encoder, eval_real_action_encoder):
    # FT ONLY THE SELF-ATTENTION LAYERS
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
            
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            if args.ft_more_params:
                if "self_attn" in name or "mlp" in name:
                    param.requires_grad = True
                    print("set to True:", name)
                else:
                    param.requires_grad = False
            else:
                if "self_attn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    TRAIN_EXAMPLES = len(train_dataset) * args.num_repeats
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_DEV = args.batch_size_dev # on l40 was 8
    GRAD_ACCUM = int(round(BATCH_SIZE / BATCH_SIZE_DEV))
    if args.max_steps > 0:
        TRAIN_STEPS = args.max_steps
    else:
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


    if args.eval_dataset == "double":
        trainer = MultiEvalSeq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_sim_dataset,
            data_collator=collate_fn,
            args=args_jax,
            compute_metrics=get_compute_metrics_fn(processor, SEQLEN, eval_sim_dummy_camera, action_encoder, eval_sim_action_encoder),
        )

        trainer.eval_dataset_real = eval_real_dataset
        trainer.compute_metrics_real = get_compute_metrics_fn(processor, SEQLEN, eval_real_dummy_camera, action_encoder, eval_real_action_encoder)
        trainer.compute_metrics_sim = get_compute_metrics_fn(processor, SEQLEN, eval_sim_dummy_camera, action_encoder, eval_sim_action_encoder)
    else:
        if args.eval_dataset == "real":
            eval_dataset = eval_real_dataset
            eval_dummy_camera = eval_real_dummy_camera
            eval_action_encoder = eval_real_action_encoder
        elif args.eval_dataset == "sim":
            eval_dataset = eval_sim_dataset
            eval_dummy_camera = eval_sim_dummy_camera
            eval_action_encoder = eval_sim_action_encoder
        else:
            raise ValueError(f"Unknown eval dataset: {args.eval_dataset}")
        
        trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            args=args_jax,
            compute_metrics=get_compute_metrics_fn(processor, SEQLEN, eval_dummy_camera, action_encoder, eval_action_encoder),
        )

    trainer.model.config.use_cache = False

    return trainer


def get_datasets(args, dataset_location, valid_dataset_location):
    
    # SETTING UP THE DATASET
    action_encoder = args.action_encoder
    bg_image_dataset = ImageFolderDataset("/tmp/indoorCVPR/Images", transform=transforms.RandomResizedCrop((448, 448)))
    randomize_background = RandomizeBackgrounds(p=args.p_background, background_images=bg_image_dataset)
    forced_image_augs = RandomizeBackgrounds(p=1.0, background_images=bg_image_dataset)
    
    return_depth = False
    if args.depth or args.depth_on_both:
        return_depth = True

    if args.no_augs:
        augment_rgbds = None
        augment_rgb=None
        augment_text=None
        augment_depth=None
    else:
        augment_rgbds=randomize_background
        augment_rgb = augment_image_rgb
        if args.conditioning == "trajectory":
            augment_text = None
        else:
            augment_text = complexify_text
        augment_depth = None
    
    if "mix30obj" in args.dataset_version:
        from torch.utils.data import ConcatDataset
        if "mix30obj-8" in args.dataset_version:
            cvla_dataset = "cvla-obja-8"
            clevr_dataset = "cvla-clevr-8"
        elif "mix30obj-camF-sceneF-9" in args.dataset_version:
            clevr_dataset = "cvla-clevr-camF-sceneF-9"
            cvla_dataset = "cvla-obja-camF-sceneF-9"
        elif "mix30obj-camRF-sceneF-9" in args.dataset_version:
            clevr_dataset = "cvla-clevr-camRF-sceneF-9"
            cvla_dataset = "cvla-obja-camRF-sceneF-9"
        elif "mix30obj-camRF-sceneR-9" in args.dataset_version:
            clevr_dataset = "cvla-clevr-camRF-sceneR-9"
            cvla_dataset = "cvla-obja-camRF-sceneR-9"
        elif "mix30obj-camRS-sceneF-9" in args.dataset_version:
            clevr_dataset = "cvla-clevr-camRS-sceneF-9"
            cvla_dataset = "cvla-obja-camRS-sceneF-9"
        else:
            raise ValueError(f"Unknown dataset version: {args.dataset_version}")
        
        dataset_location = f"/tmp/{clevr_dataset}"
        dataset_location1 = f"/tmp/{clevr_dataset}"
        dataset_location2 = f"/tmp/{cvla_dataset}"

        dataset1 = H5Dataset(dataset_location1, return_depth=return_depth, action_encoder=action_encoder, limit_samples=100_000,
                             augment_rgbds=augment_rgbds, augment_rgb=augment_rgb, augment_text=augment_text, augment_depth=augment_depth)
        dataset2 = H5Dataset(dataset_location2, return_depth=return_depth, action_encoder=action_encoder, limit_samples=50_000,
                             augment_rgbds=augment_rgbds, augment_rgb=augment_rgb, augment_text=augment_text, augment_depth=augment_depth)
        raw_dataset = ConcatDataset([dataset1, dataset2])
        assert dataset1.action_encoder.NAME == dataset2.action_encoder.NAME, f"Action encoders are different: {dataset1.action_encoder.NAME} vs {dataset2.action_encoder.NAME}"
        raw_dataset.action_encoder = dataset1.action_encoder
        print("Concatenated the datasets of lengths:", len(dataset1), len(dataset2), "total:", len(raw_dataset))
    else:
        raw_dataset = H5Dataset(dataset_location, return_depth=return_depth, action_encoder=action_encoder,
                                augment_rgbds=augment_rgbds, augment_rgb=augment_rgb, augment_text=augment_text, augment_depth=augment_depth)
    
    clean_text = CleanText(truncate_len=75)
    if args.conditioning == "text":
        run_name = f"{args.dataset_version}_text_lr{args.lr}" + args.extra_run_name
        train_dataset = raw_dataset
        
        eval_sim_dataset = H5Dataset(valid_dataset_location, action_encoder=action_encoder, return_depth=return_depth, limit_samples=200,
                                     augment_rgbds=None, augment_rgb=None, augment_text=None, augment_depth=None)
        eval_sim_dummy_camera = eval_sim_dataset[0][1]["camera"]
        eval_sim_action_encoder = None
        eval_dataset_location  = Path("/data/lmbraid19/argusm/datasets/cvla-droid-block-simple-v4")
        train_ratio = 0.0 if "droid-block" in str(eval_dataset_location) else 0.8
        eval_real_dataset = JSONLDataset(jsonl_file_path=eval_dataset_location, action_encoder=action_encoder,
                                         augment_text=clean_text, return_depth=return_depth, split="valid", train_ratio=train_ratio)

        eval_real_dummy_camera = eval_real_dataset[0][1]["camera"]
        eval_real_action_encoder = eval_real_dataset.action_encoder

    elif args.conditioning == "trajectory":
        num_images_in_context = args.num_images_in_context
        image_order = args.image_order

        run_name = f"_img_{num_images_in_context}_pr_{image_order}_enc_{action_encoder}"
        eval_run_name = run_name
        run_name = f"{args.dataset_version}_{run_name}"
        load_presampled_pairs_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"{run_name}_new.pkl"
        run_name += f"maxTokens{args.max_tokens}_lr{args.lr}" + args.extra_run_name  

        train_dataset = PairedDataset(raw_dataset, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=load_presampled_pairs_path,
                                    mode="train", p_copy=args.p_copy, apply_copy_augs=args.apply_copy_augs, p_sort_by_l2_distance=args.p_sort_by_l2_distance, 
                                    sort_criteria=args.sort_criteria, presampled_path=None, sampling_type=args.sampling_type)
        
        eval_dataset_location  = Path("/data/lmbraid19/argusm/datasets/cvla-droid-block-simple-v4")
        eval_real_load_presampled_pairs_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"{eval_dataset_location.name}_dataset_{eval_run_name}_new.pkl"
        presampled_eval_sequences_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"{eval_dataset_location.name}_{eval_run_name}_pCopy0_pSorting0_presampled_eval_sequences.pkl"
        
        train_ratio = 0.0 if "droid-block" in str(eval_dataset_location) else 0.8
        real_raw_eval_dataset = JSONLDataset(jsonl_file_path=eval_dataset_location, augment_text=clean_text, return_depth=return_depth, split="valid", train_ratio=train_ratio, action_encoder=action_encoder)
        
        
        eval_real_dataset = PairedDataset(real_raw_eval_dataset, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=eval_real_load_presampled_pairs_path,
                                    mode="test", p_copy=0, apply_copy_augs=False, p_sort_by_l2_distance=0, 
                                    sort_criteria=args.sort_criteria, presampled_path=presampled_eval_sequences_path)
        eval_real_dummy_camera = eval_real_dataset[0][1][0]["camera"]
        eval_real_action_encoder = real_raw_eval_dataset.action_encoder

        sim_raw_dataset_for_eval = H5Dataset(valid_dataset_location, augment_rgbds=None, augment_rgb=None, augment_text=None, augment_depth=None, 
                                     return_depth=return_depth, action_encoder=action_encoder)
        sim_load_presampled_pairs_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"{valid_dataset_location.name}_{eval_run_name}_new.pkl"
        sim_presampled_eval_sequences_path = Path("/data/lmbraid21/bratulic/max_pali/datasets") / f"{valid_dataset_location.name}_{eval_run_name}_pCopy0_pSorting0_presampled_eval_sequences.pkl"

        eval_sim_dataset = PairedDataset(sim_raw_dataset_for_eval, num_images_in_context=num_images_in_context, image_order=image_order, load_presampled_pairs_path=sim_load_presampled_pairs_path,
                                    mode="test", p_copy=0, apply_copy_augs=False, p_sort_by_l2_distance=0, 
                                    sort_criteria=args.sort_criteria, presampled_path=sim_presampled_eval_sequences_path)
        
        eval_sim_dummy_camera = eval_sim_dataset[0][1][0]["camera"]
        eval_sim_action_encoder = sim_raw_dataset_for_eval.action_encoder

        # Initial test which tasks we have
        # print(eval_sim_dataset.task_lookup.keys())
        # print(eval_real_dataset.task_lookup.keys())

    else:
        raise ValueError(f"Unknown conditioning type: {args.conditioning}")
    
    assert train_dataset.action_encoder.NAME == eval_sim_dataset.action_encoder.NAME, f"Action encoders are different: {train_dataset.action_encoder.NAME} vs {eval_sim_dataset.action_encoder.NAME}"
    assert train_dataset.action_encoder.NAME == eval_real_dataset.action_encoder.NAME, f"Action encoders are different: {train_dataset.action_encoder.NAME} vs {eval_real_dataset.action_encoder.NAME}"

    print("dataset_location:", dataset_location,"samples:", len(raw_dataset), "paired_samples:", len(train_dataset))

    return train_dataset, eval_sim_dataset, eval_real_dataset, run_name, eval_sim_dummy_camera,  eval_real_dummy_camera, raw_dataset.action_encoder, eval_sim_action_encoder, eval_real_action_encoder


def load_data_to_node(data_location="/work/dlclarge2/bratulic-cvla/", dataset_version="cvla-clevr-8"):
    # DATA COPY-PASTING AND CHECK
    if not os.path.exists('/tmp/indoorCVPR'):
        cmd1 = (
        f"rsync -a --progress {data_location}/indoorCVPR_09.tar /tmp/ && "
        "mkdir -p /tmp/indoorCVPR && "
        "tar -xf /tmp/indoorCVPR_09.tar -C /tmp/indoorCVPR"
        )
        subprocess.run(cmd1, shell=True, check=True)

        cmd2 = "file /tmp/indoorCVPR"
        result1 = subprocess.run(cmd2, shell=True, check=True, capture_output=True, text=True)
        print(result1.stdout)
    else:
        print('Data already copied.')

    if "cvla-clevr" in dataset_version:
        clevr_dataset = dataset_version
        cvla_dataset = None
    elif "mix30obj-8" in dataset_version:
        cvla_dataset = "cvla-obja-8"
        clevr_dataset = "cvla-clevr-8"
    elif "mix30obj-camF-sceneF-9" in dataset_version:
        clevr_dataset = "cvla-clevr-camF-sceneF-9"
        cvla_dataset = "cvla-obja-camF-sceneF-9"
    elif "mix30obj-camRF-sceneF-9" in dataset_version:
        clevr_dataset = "cvla-clevr-camRF-sceneF-9"
        cvla_dataset = "cvla-obja-camRF-sceneF-9"
    elif "mix30obj-camRF-sceneR-9" in dataset_version:
        clevr_dataset = "cvla-clevr-camRF-sceneR-9"
        cvla_dataset = "cvla-obja-camRF-sceneR-9"
    elif "mix30obj-camRS-sceneF-9" in dataset_version:
        clevr_dataset = "cvla-clevr-camRS-sceneF-9"
        cvla_dataset = "cvla-obja-camRS-sceneF-9"
    else:
        raise ValueError(f"Unknown dataset version: {dataset_version}")
    
    if cvla_dataset is not None:
        if not os.path.exists(f'/tmp/{cvla_dataset}'):
            cmd3 = f"rsync -a --progress {data_location}/{cvla_dataset}/ /tmp/{cvla_dataset}/"
            subprocess.run(cmd3, shell=True, check=True)

            cmd4 = f"file /tmp/{cvla_dataset}"
            result2 = subprocess.run(cmd4, shell=True, check=True, capture_output=True, text=True)
            print(result2.stdout)
        else:
            print(f'Data {cvla_dataset} already copied.')

    if not os.path.exists(f'/tmp/{clevr_dataset}'):
        cmd3 = f"rsync -a --progress {data_location}/{clevr_dataset}/ /tmp/{clevr_dataset}/"
        subprocess.run(cmd3, shell=True, check=True)

        cmd4 = f"file /tmp/{clevr_dataset}"
        result2 = subprocess.run(cmd4, shell=True, check=True, capture_output=True, text=True)
        print(result2.stdout)
    else:
        print(f'Data {clevr_dataset} already copied.')

    valid_dataset = clevr_dataset + "-valid"
    
    if not os.path.exists(f'/tmp/{valid_dataset}'):
        cmd3 = f"rsync -a --progress {data_location}/{valid_dataset}/ /tmp/{valid_dataset}/"
        subprocess.run(cmd3, shell=True, check=True)

        cmd4 = f"file /tmp/{valid_dataset}"
        result2 = subprocess.run(cmd4, shell=True, check=True, capture_output=True, text=True)
        print(result2.stdout)
    else:
        print(f'Data {valid_dataset} already copied.')


def get_args():
    parser = argparse.ArgumentParser()

    # model and task specific options
    parser.add_argument("--model_id", type=str, default="google/paligemma2-3b-pt-224")
    parser.add_argument("--conditioning", type=str, choices=["text", "trajectory"], default="text")
    parser.add_argument("--action_encoder", type=str, default="xyzrotvec-cam-512xy128d", help="Encoder to use for the model")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--depth_on_both", action="store_true", help="Use depth on both images in context and query image")
    parser.add_argument("--max_tokens", type=int, default=13, help="Max tokens for generation (basically sequence length)")
    parser.add_argument("--dataset_version", type=str, choices=["cvla-clevr-8", 
                                                                "cvla-clevr-camF-sceneF-9", "cvla-clevr-camRF-sceneF-9",  
                                                                "cvla-clevr-camRF-sceneR-9", "cvla-clevr-camRS-sceneF-9",
                                                                "mix30obj-8", 
                                                                "mix30obj-camF-sceneF-9", "mix30obj-camRF-sceneF-9",
                                                                "mix30obj-camRF-sceneR-9", "mix30obj-camRS-sceneF-9"], default="mix30obj-8", help="Dataset version to use")
    
    # augmentation options
    parser.add_argument("--p_background", type=float, default=0.2)
    parser.add_argument("--no_augs", action="store_true")
    
    # save and eval options
    parser.add_argument("--extra_run_name", type=str, default="debug")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_limit", type=int, default=15)
    parser.add_argument("--save_path", type=str, default="/work/dlclarge2/bratulic-cvla/models")
    parser.add_argument("--no_eval", action="store_true", help="Do not evaluate the model")
    parser.add_argument("--eval_dataset", type=str, choices=["real", "sim", "double"], default="double", help="Dataset to evaluate on")
    parser.add_argument("--data_location", type=str, default="/work/dlclarge2/bratulic-cvla/")
    
    # demo-specific options
    parser.add_argument("--num_images_in_context", type=int, default=1)
    parser.add_argument("--image_order", type=str, choices=["interleaved", "images_first"], default="interleaved")
    parser.add_argument("--sampling_type", type=str, choices=["random", "all"], default="all", help="Sampling type for the images in context. Random will sample randomly, all will try to not reuse the same image combination twice")
    parser.add_argument("--p_copy", type=float, default=0.0, help="Percentage of pairs with direct copy of images in context")
    parser.add_argument("--apply_copy_augs", action="store_true", help="Apply augmentations to the copy of the images in context")
    parser.add_argument("--p_sort_by_l2_distance", type=float, default=0.0, help="Sort the images in context by L2 distance to the query image for some percentage")
    parser.add_argument("--sort_criteria", type=str, choices=["camera_position", "trajectory_shape"], default="camera_position", help="Sort the images in context by camera position or trajectory shape")
    
    # optimization options
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_dev", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for training, -1 for using one epoch of defined data")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--ft_more_params", action="store_true", help="Fine-tune more parameters in the model")
    
    return parser.parse_args()


def main():
    current_time =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = get_args()

    if "mix30obj" in args.dataset_version:
        dataset_location = "_" + args.dataset_version
        if args.dataset_version == "mix30obj-8":
            valid_dataset_location = Path("/tmp") / "cvla-clevr-8-valid"
        elif "mix30obj-camF-sceneF-9" in args.dataset_version:
            valid_dataset_location = Path("/tmp") / "cvla-clevr-camF-sceneF-9-valid"
        elif "mix30obj-camRF-sceneF-9" in args.dataset_version:
            valid_dataset_location = Path("/tmp") / "cvla-clevr-camRF-sceneF-9-valid"
        elif "mix30obj-camRF-sceneR-9" in args.dataset_version:
            valid_dataset_location = Path("/tmp") / "cvla-clevr-camRF-sceneR-9-valid"
        elif "mix30obj-camRS-sceneF-9" in args.dataset_version:
            valid_dataset_location = Path("/tmp") / "cvla-clevr-camRS-sceneF-9-valid"

    else:
        dataset_location = Path("/tmp") / args.dataset_version
        valid_dataset_location = Path("/tmp") / (args.dataset_version + "-valid") 

    load_data_to_node(args.data_location, args.dataset_version)

    # SETTING UP THE DATASETS
    train_dataset, eval_sim_dataset, eval_real_dataset, run_name, eval_sim_dummy_camera, eval_real_dummy_camera, action_encoder, eval_sim_action_encoder, eval_real_action_encoder = get_datasets(args, dataset_location, valid_dataset_location)
    
    # SETTING UP THE SAVE PATHS
    save_path_final = Path(args.save_path) / (run_name + "_" + current_time)
    save_path = Path(f"/tmp/cvla_{getpass.getuser()}") / (run_name + "_" + current_time)
    save_hyperparams(save_path_final, save_path, args)

    # SETTING UP THE MODEL
    DEVICE, TORCH_DTYPE = torch.device('cuda'), torch.bfloat16
    processor, model = get_model(model_type=args.model_id, TORCH_DTYPE=TORCH_DTYPE, DEVICE=DEVICE, checkpoint=args.checkpoint)
    collate_fn = get_collate_fn(processor, args.num_images_in_context, args.image_order, TORCH_DTYPE, args, DEVICE)

    # SETTING UP THE TRAINER
    trainer = get_trainer(args, model, processor, train_dataset, eval_sim_dataset, eval_real_dataset, collate_fn, save_path, eval_sim_dummy_camera, eval_real_dummy_camera, action_encoder, eval_sim_action_encoder, eval_real_action_encoder)

    # TRAINING THE MODEL
    import traceback
    try:
        trainer.train(resume_from_checkpoint=args.checkpoint)
    except: #Exception as e:  # Catch all exceptions, including AssertionError, but not keyboard ??
        # 
        #print(f"Encountered error {e.__class__.__name__}")
        #print(e)
        #traceback.print_exc()  # Prints the full traceback
        print("Training failed. Saving model and exiting...")
    print("Done training.")

    # TRANSFER THE MODEL TO FINAL LOCATION
    os.system(f"mv {save_path}/* {save_path_final}/")
    print(f"done copying {save_path} to {save_path_final}.")

    os.system(f"rm -r {save_path}")
    print("done cleanup.")
    

if __name__ == "__main__":
    main()
