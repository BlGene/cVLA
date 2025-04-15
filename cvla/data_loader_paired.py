import tqdm
from pathlib import Path
import numpy as np
import pickle
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


def compute_camera_position(extrinsics):
    extrinsics = np.array(extrinsics)
    # If extrinsics has an extra dimension, remove it:
    if extrinsics.ndim == 3 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    R = extrinsics[:3, :3]  # rotation matrix (3x3)
    t = extrinsics[:3, 3]   # translation vector (3,)
    # For a world-to-camera transformation, the camera center is computed as:
    camera_center = -R.T @ t
    return camera_center


class PairedDataset(Dataset):
    def __init__(self, 
                 raw_dataset, 
                 num_images_in_context=1, 
                 image_order="interleaved", 
                 load_presampled_pairs_path=None,
                 p_copy=0.0,
                 apply_copy_augs=False,
                 p_sort_by_l2_distance=0.0,
                 plot_statistics=False,
                 sort_criteria="camera_position",   # camera or trajectory coords
                 mode="train",
                 presampled_path=None):
        self.raw_dataset = raw_dataset
        self.num_images_in_context = num_images_in_context
        self.image_order = image_order
        self.load_presampled_pairs_path = load_presampled_pairs_path
        self.p_copy = p_copy
        self.apply_copy_augs = apply_copy_augs
        self.p_sort_by_l2_distance = p_sort_by_l2_distance
        self.just_indices = False
        assert self.image_order in ["interleaved", "images_first"], \
            "image_order must be either 'interleaved' or 'images_first'"
        
        if presampled_path is not None and presampled_path.exists() and mode == "test":
            print("Loaded presampled pairs from file.")
            with open(presampled_path, "rb") as f:
                self.presampled_pairs = pickle.load(f)

            # Also load task look-up because we will need it for simulation
            if self.load_presampled_pairs_path is not None and self.load_presampled_pairs_path.exists():
                print(f"Loading pre-sampled pairs from {load_presampled_pairs_path}")
                
                with open(load_presampled_pairs_path, "rb") as f:
                    self.task_lookup = pickle.load(f)
        else:
            self.presampled_pairs = None

            # If a presampled pairs file exists, load it
            if self.load_presampled_pairs_path is not None and self.load_presampled_pairs_path.exists():
                print(f"Loading pre-sampled pairs from {load_presampled_pairs_path}")
                
                with open(load_presampled_pairs_path, "rb") as f:
                    self.task_lookup = pickle.load(f)

                camera_position_path = Path(str(load_presampled_pairs_path).replace(".pkl", "_camera_similarities.pkl"))
                trajectory_shape_path = Path(str(load_presampled_pairs_path).replace(".pkl", "_trajectory_similarities.pkl"))

                with camera_position_path.open(mode="rb") as f:
                    self.task_images_camera_similarities = pickle.load(f)

                with trajectory_shape_path.open(mode="rb") as f:
                    self.task_images_trajectory_similarities = pickle.load(f)
            else:
                # Setup: define a lookup table for task prefixes that stores image indices and camera positions.
                # Use a lambda so that each new key gets a dictionary with the desired structure.
                self.task_lookup = {}
                self.task_images_camera_similarities = {}  # store for every task the pairwise distances matrix
                self.task_images_trajectory_similarities = {}  # store for every task the pairwise distances matrix
                
                # Enable returning only the prefix to speed up processing
                self.raw_dataset.return_only_prefix = True
                for i in tqdm.tqdm(range(len(self.raw_dataset))):
                    entry = self.raw_dataset[i]
                    prefix = entry["prefix"].split("<")[0].strip()  # represents the task
                    if prefix not in self.task_lookup:
                        self.task_lookup[prefix] = {"images": [], "camera_positions": [], "trajectory_coords": []}
                    camera_position = compute_camera_position(entry["camera_extrinsic"])
                    whd, _ = self.raw_dataset.action_encoder.decode_caption(entry["prefix"], entry["camera"])   # given in pixels
                    self.task_lookup[prefix]["images"].append(i)
                    self.task_lookup[prefix]["camera_positions"].append(camera_position)
                    self.task_lookup[prefix]["trajectory_coords"].append(whd)
                    
                self.raw_dataset.return_only_prefix = False

                if self.load_presampled_pairs_path is not None and not self.load_presampled_pairs_path.exists():
                    print(f"Saving pre-sampled pairs to {load_presampled_pairs_path}")

                    with open(load_presampled_pairs_path, "wb") as f:
                        pickle.dump(self.task_lookup, f)

                # For every task, compute an images x images matrix with camera distances
                for task, task_data in self.task_lookup.items():
                    camera_positions = np.array(task_data["camera_positions"])
                    # Compute pairwise Euclidean distances
                    dists = np.linalg.norm(camera_positions[:, None] - camera_positions[None, :], axis=-1)
                    self.task_images_camera_similarities[task] = dists

                    # Compute pairwise trajectory distances
                    trajectory_coords = np.array(task_data["trajectory_coords"])
                    # Compute pairwise Euclidean distances
                    traj_dists = np.linalg.norm(trajectory_coords[:, None] - trajectory_coords[None, :], axis=-1)
                    self.task_images_trajectory_similarities[task] = traj_dists

                # Save the presampled pairs lookup if a path was provided and does not exist yet
                if self.load_presampled_pairs_path is not None:
                    camera_position_path = Path(str(load_presampled_pairs_path).replace(".pkl", "_camera_similarities.pkl"))
                    trajectory_shape_path = Path(str(load_presampled_pairs_path).replace(".pkl", "_trajectory_similarities.pkl"))

                    if not camera_position_path.exists():
                        print(f"Saving camera position similarities to {camera_position_path}")
                
                        with open(camera_position_path, "wb") as f:
                            pickle.dump(self.task_images_camera_similarities, f)

                    if not trajectory_shape_path.exists():
                        print(f"Saving trajectory shape similarities to {trajectory_shape_path}")

                        with open(trajectory_shape_path, "wb") as f:
                            pickle.dump(self.task_images_trajectory_similarities, f)
        
            # Remove tasks with only one image
            self.task_lookup = {k: v for k, v in self.task_lookup.items() if len(v["images"]) > 1}

            # Precompute and store additional properties to re-use later
            self.saved_indices = {task: data["images"] for task, data in self.task_lookup.items()}
            self.data_properties = {task: data["camera_positions"] for task, data in self.task_lookup.items()}
            if sort_criteria == "camera_position":
                self.similarity_metric = self.task_images_camera_similarities
            else:
                self.similarity_metric = self.task_images_trajectory_similarities
        
        # Total number of samples (here defined as the sum of images over tasks)
        if mode == "train":
            self.paired_len = sum([len(v["images"]) for v in self.task_lookup.values()])
        else:
            if self.presampled_pairs is not None:
                self.paired_len = len(self.presampled_pairs)
            else:
                print("Creating presampled pairs for validation/test set.")
                # presample the pairs
                self.paired_len = 1000
                self.just_indices = True
                tmp_indices = []
                for i in range(1000):
                    tmp_indices.append(self[i])

                self.presampled_pairs = tmp_indices
                self.just_indices = False
                # Save the presampled pairs
                if presampled_path is not None:
                    with open(presampled_path, "wb") as f:
                        pickle.dump(self.presampled_pairs, f)
        
        # Print and plot statistics about the dataset
        print("Statistics about the paired dataset:")
        #print(f"Number of tasks with >1 image: {len(self.task_lookup)}")
        print(f"Total number of images (across tasks): {self.paired_len}")
        if plot_statistics:
            self.plot_statistics()

    def plot_statistics(self):
        # Plot histogram of images per task
        task_image_counts = [len(data["images"]) for data in self.task_lookup.values()]
        plt.figure(figsize=(8, 5))
        plt.hist(task_image_counts, bins=range(1, max(task_image_counts) + 2), align='left', rwidth=0.8)
        plt.xlabel("Number of Images per Task")
        plt.ylabel("Frequency")
        plt.title("Histogram of Images per Task")
        plt.xticks(range(1, max(task_image_counts) + 1))
        plt.show()
        
        # Plot histogram of camera pose differences (pairwise distances) across all tasks
        all_distances = []
        for dists in self.task_images_camera_similarities.values():
            # We take the upper triangle (excluding diagonal) to avoid duplicates
            triu_indices = np.triu_indices_from(dists, k=1)
            all_distances.extend(dists[triu_indices])
        if all_distances:
            plt.figure(figsize=(8, 5))
            plt.hist(all_distances, bins=20, rwidth=0.8)
            plt.xlabel("Euclidean Distance Between Camera Positions")
            plt.ylabel("Frequency")
            plt.title("Histogram of Camera Pose Differences")
            plt.show()
            
            print("Camera Pose Distance Statistics:")
            print("Mean distance:", np.mean(all_distances))
            print("Median distance:", np.median(all_distances))
            print("Min distance:", np.min(all_distances))
            print("Max distance:", np.max(all_distances))
        else:
            print("Not enough camera position data to compute pose differences.")
    
    def __len__(self):
        return self.paired_len
            
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        if self.presampled_pairs is not None:
            sequence_to_sample = self.presampled_pairs[idx]
            images = []
            entries = []
            for i in sequence_to_sample:
                image, entry = self.raw_dataset[i]
                images.append(image)
                entries.append(entry)

        else:
        
            # Uniformly sample a task (TODO: consider weighted sampling if needed)
            task = np.random.choice(list(self.task_lookup.keys()))
            
            if np.random.rand() < self.p_copy:
                # Copy sampling: context and query are exactly the same image.
                chosen_idx = np.random.choice(self.task_lookup[task]["images"], 1, replace=False)[0]
                if self.just_indices:
                    return [chosen_idx] * (self.num_images_in_context + 1)
                image, entry = self.raw_dataset.getitem_func(chosen_idx)   # self.raw_dataset[chosen_idx]
                images = [image] * (self.num_images_in_context + 1)
                entries = [entry] * (self.num_images_in_context + 1)
                # Optionally, apply augmentation on the copies
                if self.apply_copy_augs:
                    augmented_copy = self.raw_dataset.getitem_func(chosen_idx, force_augs=True)
                    images[-1] = augmented_copy[0]
                    
            elif np.random.rand() < self.p_sort_by_l2_distance:
                # Sample one query image, then choose context images based on similarity (L2 distance)
                task_img_indices = self.task_lookup[task]["images"]
                query_pos = np.random.choice(len(task_img_indices), 1, replace=False)[0]
                # Get the distances of all images to the chosen query image
                distances = self.task_images_camera_similarities[task][query_pos]
                # Sort indices; skip the query itself (first index after sorting should be itself)
                sorted_indices = np.argsort(distances)
                context_indices = sorted_indices[1:self.num_images_in_context + 1]
                
                if self.just_indices:
                    return [task_img_indices[i] for i in context_indices] + [task_img_indices[query_pos]]

                images, entries = [], []
                for i in context_indices:
                    img_idx = task_img_indices[i]
                    image, entry = self.raw_dataset[img_idx]
                    images.append(image)
                    entries.append(entry)
                
                # Get the query image and append it at the end
                query_img_idx = task_img_indices[query_pos]
                query_image, query_entry = self.raw_dataset[query_img_idx]
                images.append(query_image)
                entries.append(query_entry)
            
            else:
                # Uniformly sample random images from the task (without sorting by pose)
                task_img_indices = self.task_lookup[task]["images"]
                # Sample without replacement the required number of images (context + query)
                chosen_indices = np.random.choice(task_img_indices, self.num_images_in_context + 1, replace=False)
                if self.just_indices:
                    return chosen_indices.tolist()
                images, entries = [], []
                for idx in chosen_indices:
                    image, entry = self.raw_dataset[idx]
                    images.append(image)
                    entries.append(entry)
        
        # Optionally, you can also return additional info (indices or similarity metrics) if needed.
        return images, entries