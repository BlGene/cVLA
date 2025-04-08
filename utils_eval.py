import numpy as np
import torch

from scipy.spatial.transform import Rotation as R


class Evaluator:
    def __init__(self, encoder, eval_dummy_camera):
        self.decode_caption = encoder.decode_caption
        self.decode_trajectory = encoder.decode_trajectory
        self.eval_dummy_camera = eval_dummy_camera
        self.eval_dummy_camera.extrinsic_matrix = torch.tensor([[[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0]]])
        self.h_image = self.eval_dummy_camera.height
        self.w_image = self.eval_dummy_camera.width

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


    def evaluate(self, decoded_preds, decoded_labels):
        self.total_counter += 1
        
        if len(decoded_preds) != len(decoded_labels):
            return
        
        self.valid_counter += 1

        for mode in ("cam", "cart"):    
            if mode == "cam":
                dec_func = self.decode_caption
            elif mode == "cart":
                dec_func = self.decode_trajectory

            try:
                pos_data, orn_data = dec_func(decoded_labels, camera=self.eval_dummy_camera)
                pos_pred, orn_pred = dec_func(decoded_preds, camera=self.eval_dummy_camera)
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
        else:
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
                return_stats_dict[f"{mode}_l1_depth"] = l1_depth
                return_stats_dict[f"{mode}_l1_depth_obj"] = l1_depth_obj

            return_stats_dict["valid_counter"] = self.valid_counter / self.total_counter

        return return_stats_dict
    
    def reset(self):
        self.valid_counter = 0
        self.total_counter = 0
        self.all_data = dict(
            cam=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[])),
            cart=dict(pred=dict(orn=[], pos=[]), data=dict(orn=[], pos=[]))
            )
