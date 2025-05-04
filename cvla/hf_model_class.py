"""
A wrapper for the huggingeface model
"""

import json
import re
import numpy as np
from pathlib import Path
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from cvla.utils_traj_tokens import getActionEncInstance
from cvla.data_augmentations import depth_to_color

def load_config_from_txt(filepath):
    def convert_value(value):
        value = value.strip()
        if value.lower() == 'none':
            return None
        elif value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        try:
            # Try int first
            return int(value)
        except ValueError:
            try:
                # Then float
                return float(value)
            except ValueError:
                # Otherwise keep as string
                return value

    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = convert_value(value)

    return config


def get_robot_state(prefix):
    pattern = r"((?:<loc\d+>|<seg\d+>)+)"
    match = re.search(pattern, prefix)
    if match:
        extracted = match.group(1)
        return extracted
    return ""


class cVLA_wrapped:
    def __init__(self, model_path):
        # some processing
        info_file = model_path.parent / "cvla_info.json"

        args_file = model_path.parent / "args.txt"

        if args_file.exists():
            args = load_config_from_txt(args_file)
            self.conditioning = args.get("conditioning", None)
        else:
            args = None
            self.conditioning = "text"

        if self.conditioning == "trajectory":
            action_enoder = args.get("action_encoder", "xyzrotvec-cam-1024xy")
            enc_model = getActionEncInstance(action_enoder)
            self.depth_on_query = args.get("depth", False)
            self.depth_on_both = args.get("depth_on_both", False)
            return_depth = self.depth_on_query or self.depth_on_both
        else:
            try:
                with open(info_file, "r") as f:
                    model_info = json.load(f)
            except FileNotFoundError:
                model_info = None

            if model_info is not None:
                return_depth = model_info["return_depth"]
                action_enoder = model_info["action_encoder"]
                enc_model = getActionEncInstance(action_enoder)
            else:
                print("Warning: loading default encoder: xyzrotvec-cam-1024xy")
                enc_model = getActionEncInstance("xyzrotvec-cam-1024xy")
                return_depth = False
                if "_depth" in str(model_path):
                    return_depth = True

            
        self.model_path = model_path

        self.load_model(model_path, return_depth)
        self.enc_model = enc_model
        self.model_path = model_path
        self.return_depth = return_depth

        self.conditioning_dataset = None
        self.task_lookup = None

    def print_summary(self):
        model_name = self.model_path.parent.name
        print("model:".ljust(10), model_name,"\t", self.model_path)
        print("encoder:".ljust(10), self.enc_model.NAME)
        print("depth:".ljust(10), self.return_depth)



    def set_conditioning_dataset(self, conditioning_dataset):
        self.conditioning_dataset = conditioning_dataset
        self.task_lookup = conditioning_dataset.task_lookup

    def load_model(self, model_path, return_depth):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TORCH_DTYPE = torch.bfloat16
        print('Using device:', DEVICE)

        MODEL_ID ="google/paligemma2-3b-pt-224"
        processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
        print("loaded processor.")
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=TORCH_DTYPE, device_map="auto")

        if self.conditioning == "trajectory":
             def collate_fn(batch):
                images, labels, depths = zip(*batch)        # images will be lists of lists since one batch input has multiple images
                prefixes = []
                for i in range(len(labels)):
                    tmp_prefix = ""
                    if self.depth_on_both:
                        tmp_prefix += "<image>"
                    tmp_prefix += "<image>" + get_robot_state(labels[i][0]["prefix"]) + " " + labels[i][0]["suffix"]
                    tmp_prefix += "<image>"
                    if return_depth:
                        tmp_prefix += "<image>"
                    tmp_prefix += get_robot_state(labels[i][-1]["prefix"]) + " "
                    
                    prefixes.append(tmp_prefix)
                
                if return_depth:
                    images_flat = []
                    for image_tuples, depth_tuples in zip(images, depths):
                        if self.depth_on_both:
                            for image_tuple, depth_tuple in zip(image_tuples, depth_tuples):
                                images_flat.append(depth_tuple)
                                images_flat.append(image_tuple)
                        else:
                            images_flat.extend(image_tuples[:-1])
                            images_flat.append(depth_tuples[-1])
                            images_flat.append(image_tuples[-1])
                else:
                    images_flat = [image for images_list in images for image in images_list]
                inputs = processor(
                    text=prefixes,
                    images=images_flat,
                    return_tensors="pt",
                    padding="longest",

                ).to(TORCH_DTYPE).to(DEVICE)

                return inputs
        else:
            if return_depth:
                def collate_fn(batch):
                    images, labels = zip(*batch)
                    prefixes = ["<image><image>" + label["prefix"] for label in labels]
                    #suffixes = [label["suffix"] for label in labels]
                    images_flat = [img for img_list_x in images for img in img_list_x]
                    inputs = processor(
                        text=prefixes,
                        images=images_flat,
                        return_tensors="pt",
                        #suffix=suffixes,
                        padding="longest"
                    ).to(TORCH_DTYPE).to(DEVICE)
                    return inputs
                
            else:
                def collate_fn(batch):
                    images, labels = zip(*batch)
                    prefixes = ["<image>" + label["prefix"] for label in labels]
                    inputs = processor(
                        text=prefixes,
                        images=images,
                        return_tensors="pt",
                        padding="longest"
                    ).to(TORCH_DTYPE).to(DEVICE)
                    return inputs
            
        self.processor = processor
        self.model = model
        self.collate_fn = collate_fn
        
    def predict_from_paired(self, images, entry, depth):
        assert self.conditioning == "trajectory"
        max_new_tokens = 13
        batch = [(images, entry, depth)]
        inputs = self.collate_fn(batch)

        prefix_length = inputs["input_ids"].shape[-1]
        try:
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
                decoded = [self.processor.decode(x, skip_special_tokens=True) for x in generation[:, prefix_length:]]
        except:
            print("Failed to generate for prefix:", entry)
            return [""]
        
        return decoded[0]

    def predict(self, images, prefix, robot_state=None):
        #if robot_state is None:
        #    robot_state = "<loc0137><loc0794><loc0057><seg058><seg034><seg017>"
        #assert robot_state is not None
        #prefix = action_text + " " + robot_state
        max_new_tokens = 13
        if self.conditioning == "trajectory":
            batch = self.get_trajectory_inputs(images, prefix)
            if self.return_depth:
                if batch is not None:
                    images, entries, depths = batch[0]
                    new_depths = []
                    for depth in depths:
                        depth = depth.detach().cpu().numpy().squeeze()
                        depth = np.clip(depth, 0, 1023)
                        depth = depth_to_color(depth)
                        depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)
                        new_depths.append(depth)
                    batch = [(images, entries, new_depths)]
            
        else:
            entry_dict = dict(prefix=prefix)

            depth, image = images
            if self.return_depth:
                depth = depth.detach().cpu().numpy().squeeze()
                depth = np.clip(depth, 0, 1023)  # depth im [mm]
                depth = depth_to_color(depth)
                depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)
                images = (depth, image)
            else:
                images = image
            #print(image.shape, depth.shape)
            batch = [(images, entry_dict)]  # batch size of 1

        if batch is None:   # Automatic failure if no demonstration image is found
            print("No demonstration image found for task:", prefix)
            return [""]
        
        inputs = self.collate_fn(batch)
        
        prefix_length = inputs["input_ids"].shape[-1]    
        try:
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
                decoded = [self.processor.decode(x, skip_special_tokens=True) for x in generation[:, prefix_length:]]

        except:
            print("Failed to generate for prefix:", prefix)
            return [""]

        return decoded
    
    # this is for the maniskill env, see run_env.py
    def make_predictions(self, image_before, prefix):
        decoded = self.predict(image_before, prefix)
        return None, None, None, decoded[0]

    def predict_trajectory(self, images, action_text, camera=None, robot_state=None):
        pred_text = self.predict(images, action_text=action_text)
        traj = self.enc_model.decode_trajectory(pred_text[0], camera=camera)
        return traj

    def get_trajectory_inputs(self, query_images, prefix):
        if isinstance(query_images, list) or isinstance(query_images, tuple):
            query_depth = query_images[0]
            query_image = query_images[1]
        else:
            query_depth = None
            query_image = query_images
        task = prefix.split("<")[0].strip()
        demonstration_image, demonstration_entry, demonstration_depth = self.get_demonstration_image(task)
        
        if demonstration_image is None: # No such task, can't sample demonstration
            return None
        
        query_entry = dict(prefix=prefix)

        images = [demonstration_image, query_image]
        entries = [demonstration_entry, query_entry]
        depths = [demonstration_depth, query_depth]

        batch = [(images, entries, depths)]

        return batch

    def get_demonstration_image(self, task):
        if self.task_lookup is not None and task in self.task_lookup:
            available_images = self.task_lookup[task]["images"]
            random_demonstration = np.random.choice(available_images)
            image, entry = self.conditioning_dataset.raw_dataset[random_demonstration]
            if isinstance(image, list):
                depth = image[0]
                image = image[1]
            else:
                depth = None
        else:
            image = None
            entry = None
            depth = None
        return image, entry, depth



if __name__ == "__main__":
    from data_loader_jsonl import JSONLDataset

    # define paths
    dataset_location = "/data/lmbraid19/argusm/datasets/cvla-droid-block-simple-v3"
    model_location = Path("/data/lmbraid19/argusm/models/")
    model_path = model_location / "clevr-act-7-depth_depthaug" / "checkpoint-4687"    

    # load model
    model_inst = cVLA_wrapped(model_path)

    # load some test data
    return_depth = model_inst.return_depth
    test_dataset = JSONLDataset(
        jsonl_file_path=f"{dataset_location}/_annotations.valid.jsonl",
        image_directory_path=f"{dataset_location}/dataset",
        clean_prompt=True,
        return_depth=return_depth
    )
    test_dataset.action_encoder = getActionEncInstance("xyzrotvec-cam-1024xy")

    print("dataset len", len(test_dataset))
    print(model_path)

    # access some images
    images, entry = test_dataset[0]
    action_text = entry["prefix"].split("<")[0].strip()

    print(action_text)  # e.g. put the yellow block in the blue cup
    print(images[0].shape, images[0].dtype)  # (720, 1280, 3) uint8 depth image encoded
    print(images[1])  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x715AEF7667B0>

    # camera_extrinsic = [[[1, 0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    # camera_intrinsic = [[[260.78692626953125, 0.0, 322.3820495605469],[ 0.0, 260.78692626953125, 180.76370239257812],[0.0, 0.0, 1.0]]]
    # camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)

    # make a prediction
    traj = model_inst.predict_trajectory(images, action_text=action_text, camera=entry["camera"])
    print(traj)


