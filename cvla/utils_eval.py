import numpy as np
import torch
import re

from scipy.spatial.transform import Rotation as R


def check_if_valid(decoded_preds, decoded_labels):
    all_predicted_tokens = re.findall(r"<((?:loc|seg)\d+)>", decoded_preds)
    all_label_tokens = re.findall(r"<((?:loc|seg)\d+)>", decoded_labels)

    if len(all_predicted_tokens) != len(all_label_tokens):
        return False
    # format should be loc loc loc seg seg seg loc loc loc seg seg seg
    # check if when 12 tokens, we have correct order
    if len(all_predicted_tokens) == 12:
        # check if loc loc loc seg seg seg loc loc loc seg seg seg
        expected_format = ["loc", "loc", "loc", "seg", "seg", "seg", "loc", "loc", "loc", "seg", "seg", "seg"]
        #for pred_token, expected_token in zip(all_predicted_tokens, expected_format):
        #    if expected_token not in pred_token:
        #        return False
            
    # if len 12 and passed the check, we have correct order
    return True


class Evaluator:
    def __init__(self, encoder, camera_fixed, encoder_labels=None, robot_pose_fixed=None):
        self.decode_caption_labels = encoder.decode_caption
        self.decode_caption_preds = encoder.decode_caption
        self.decode_trajectory_labels = encoder.decode_trajectory
        self.decode_trajectory_preds = encoder.decode_trajectory
        if encoder_labels is not None:
            self.decode_caption_labels = encoder_labels.decode_caption
            self.decode_trajectory_labels = encoder_labels.decode_trajectory
        self.camera_fixed = camera_fixed
        self.camera_fixed.extrinsic_matrix = torch.tensor([[[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]])
        self.h_image = self.camera_fixed.height
        self.w_image = self.camera_fixed.width
        self.robot_pose_fixed = robot_pose_fixed

        self.all_data = dict(
            cam=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[])),
            cart=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[]))
            )
        
        self.valid_counter = 0
        self.total_counter = 0
        self.action_labels = ["x", "y", "d", "orn"]
        # define max L1 and L2 distances as corners of images based on image size
        self.max_l1 = self.w_image + self.h_image
        self.max_l2 = np.sqrt(self.w_image**2 + self.h_image**2)


    def evaluate(self, decoded_preds: str, decoded_labels: str, camera=None, robot_pose=None):
        """
        Arguments:
            decoded_preds: predictions decoded from the llm (so as strings)
            decoded_labels: labels decoded from the llm (so as strings)
        """
        self.total_counter += 1 
        
        
        if not check_if_valid(decoded_preds, decoded_labels):   # either not enough tokens or wrong order
            return
        
        self.valid_counter += 1

        if camera is None:
            camera = self.camera_fixed
            
        if robot_pose is None:
            robot_pose = self.robot_pose_fixed

        for mode in ("cam", "cart"):    
            if mode == "cam":
                dec_func_preds, dec_func_lab = self.decode_caption_preds, self.decode_caption_labels
            elif mode == "cart":
                dec_func_preds, dec_func_lab = self.decode_trajectory_preds, self.decode_trajectory_labels

            try:
                pos_data, orn_data = dec_func_lab(decoded_labels, camera=camera, robot_pose=robot_pose)
                pos_pred, orn_pred = dec_func_preds(decoded_preds, camera=camera, robot_pose=robot_pose)
            except ValueError:
                print("skipping")
                continue

            if mode == "cart":
                pos_data, orn_data = pos_data[0], orn_data[0]
                pos_pred, orn_pred = pos_pred[0], orn_pred[0]
                
            self.all_data[mode]["data"]["pos"].append(pos_data.numpy())
            self.all_data[mode]["pred"]["pos"].append(pos_pred.numpy())
            self.all_data[mode]["data"]["orn"].append(R.from_quat(orn_data.numpy(), scalar_first=True))
            self.all_data[mode]["pred"]["orn"].append(R.from_quat(orn_pred.numpy(), scalar_first=True))
    
    def report_stats(self):
        # if there was no data, return max values
        if self.valid_counter == 0:
            return_stats_dict = dict()
            for mode in ("cam", "cart"):
                for i, action_label in enumerate(self.action_labels):
                    return_stats_dict[f"{mode}_{action_label}_l2"] = self.max_l2
                    return_stats_dict[f"{mode}_{action_label}_l1"] = self.max_l1
                return_stats_dict[f"{mode}_l1"] = self.max_l1
                return_stats_dict[f"{mode}_l2"] = self.max_l2
                return_stats_dict[f"{mode}_l1_depth"] = self.max_l1
                return_stats_dict[f"{mode}_l1_depth_obj"] = self.max_l1
            return_stats_dict["valid_counter"] = 0
            return return_stats_dict
        
        # elif valid_counter > 0
        for mode in self.all_data:
            for split in self.all_data[mode]:
                self.all_data[mode][split]["pos"] = np.array(self.all_data[mode][split]["pos"])

        valid_diffs = dict()
        return_stats_dict = dict()
        for mode in ("cam", "cart"):
            valid_diff = self.all_data[mode]["data"]["pos"] - self.all_data[mode]["pred"]["pos"]
            if mode == "cart":
                valid_diff = valid_diff * 100
            if mode == "cam":
                valid_diff[:, :, 2] = valid_diff[:, :, 2] * 100
            valid_orn_diffs = [(R.inv(r1) * r2) for r1, r2 in zip(self.all_data[mode]["data"]["orn"], self.all_data[mode]["pred"]["orn"])]
            valid_orn_diffs_deg = np.array([r1.magnitude() for r1 in valid_orn_diffs]) * 180 / np.pi
            valid_orn_diffs_r = [r1.as_rotvec() for r1 in valid_orn_diffs]
            valid_diffs[mode] = np.concatenate((valid_diff, valid_orn_diffs_deg[:, :, np.newaxis]), axis=-1)

            for i, action_label in enumerate(self.action_labels):
                return_stats_dict[f"{mode}_{action_label}_l2"] = np.linalg.norm(valid_diffs[mode][:, :, i])
                return_stats_dict[f"{mode}_{action_label}_l1"] = np.mean(np.abs(valid_diffs[mode][:, :, i]))
            l1 = np.mean(np.abs(valid_diffs[mode]))
            l2 = np.linalg.norm(valid_diffs[mode])
            l1_depth = np.mean(np.abs(valid_diffs[mode][:, :, 2]))
            l1_depth_obj = np.mean(np.abs(valid_diffs[mode][:, 0, 2]))
            return_stats_dict[f"{mode}_l1"] = l1
            return_stats_dict[f"{mode}_l2"] = l2
            #return_stats_dict[f"{mode}_depth_l1"] = l1_depth
            return_stats_dict[f"{mode}_d_obj_l1"] = l1_depth_obj

        return_stats_dict["valid_counter"] = self.valid_counter / self.total_counter
    
        self.valid_diffs = valid_diffs
        return return_stats_dict
    
    def reset(self):
        self.valid_counter = 0
        self.total_counter = 0
        self.all_data = dict(
            cam=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[])),
            cart=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[]))
            )