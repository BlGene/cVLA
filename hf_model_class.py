"""
A wrapper for the huggingeface model
"""

import json
from pathlib import Path
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from utils_traj_tokens import getActionEncInstance


class cVLA_wrapped:
    def __init__(self, model_path):
        # some processing
        info_file = model_path.parent / "cvla_info.json"
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
            enc_model = getActionEncInstance("xyzrotvec-cam-1024xy")
            return_depth = False
            if "_depth" in str(model_path):
                return_depth = True

        model_name = model_path.parent.name    
        if model_path.is_dir():
            print("model:".ljust(10), model_name,"\t", model_path)
            print("depth:".ljust(10), return_depth)

        self.load_model(model_path, return_depth)
        self.enc_model = enc_model
        self.model_path = model_path
        self.return_depth = return_depth


    def load_model(self, model_path, return_depth):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TORCH_DTYPE = torch.bfloat16
        print('Using device:', DEVICE)

        MODEL_ID ="google/paligemma2-3b-pt-224"
        processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
        print("loaded processor.")
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=TORCH_DTYPE, device_map="auto")

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
        

    def predict(self, images, action_text, robot_state=None):
        if robot_state is None:
            robot_state = "<loc0137><loc0794><loc0057><seg058><seg034><seg017>"

        prefix = action_text + " " + robot_state
        entry_dict = dict(prefix=prefix)

        batch = [(images, entry_dict)]  # batch size of 1
        inputs = self.collate_fn(batch)
        
        prefix_length = inputs["input_ids"].shape[-1]    
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=False)
            decoded = [self.processor.decode(x, skip_special_tokens=True) for x in generation[:, prefix_length:]]

        return decoded

    def predict_trajectory(self, images, action_text, camera=None, robot_state=None):
        pred_text = self.predict(images, action_text=action_text)
        traj = self.enc_model.decode_trajectory(pred_text[0], camera=camera)
        return traj


if __name__ == "__main__":
    from data_loader_jsonl import JSONLDataset

    # define paths
    dataset_location = "/data/lmbraid19/argusm/datasets/clevr-real-block-simple-v3"
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


