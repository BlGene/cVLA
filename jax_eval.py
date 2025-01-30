import functools
import io
import warnings
import os
import sys
from typing import List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

import tensorflow as tf
import sentencepiece

from PIL import Image

if not os.path.exists("big_vision_repo"):
  raise ValueError("!git clone --quiet --branch=main --depth=1 \
     https://github.com/google-research/big_vision big_vision_repo")

# Append big_vision code to python import path
if "big_vision_repo" not in sys.path:
  sys.path.append("big_vision_repo")

# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding

from pdb import set_trace

class VLAModel:
    TOKENIZER_PATH = "./paligemma_tokenizer.model"
    SEQLEN = 32

    def __init__(self, model_path, gpu_index=None):
        # Don't let TF use the GPU or TPUs
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")

        backend = jax.lib.xla_bridge.get_backend()
        print(f"JAX version:  {jax.__version__}")
        print(f"JAX platform: {backend.platform}")
        print(f"JAX devices:  {jax.device_count()}")

        if gpu_index == None:
            my_gpu_devices = [jax.devices()[0]]
        else:
            assert gpu_index in [x.id for x in jax.devices()]
            my_gpu_devices = [jax.devices()[gpu_index]]

        print(jax.devices())
        print("my gpu", my_gpu_devices)

        # Define model
        model_config = ml_collections.FrozenConfigDict({
            "llm": {"vocab_size": 257_152},
            "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
        })
        model = paligemma.Model(**model_config)
        tokenizer = sentencepiece.SentencePieceProcessor(self.TOKENIZER_PATH)

        # Load params - this can take up to 1 minute in T4 colabs.
        print("loading params from: ", model_path)
        model_path = str(model_path)  # get rid of posix paths
        params = paligemma.load(None, model_path, model_config)

        # Define `decode` function to sample outputs from the model.
        decode_fn = predict_fns.get_all(model)['decode']
        decode = functools.partial(decode_fn, devices=my_gpu_devices, eos_token=tokenizer.eos_id())

        # Create a pytree mask of the trainable params.
        def is_trainable_param(name, param):  # pylint: disable=unused-argument
            if name.startswith("llm/layers/attn/"):  return False
            if name.startswith("llm/"):              return False
            if name.startswith("img/"):              return False
            raise ValueError(f"Unexpected param name {name}")
        trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

        # If more than one device is available (e.g. multiple GPUs) the parameters can
        # be sharded across them to reduce HBM usage per device.
        mesh = jax.sharding.Mesh(my_gpu_devices, ("data"))

        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("data"))

        params_sharding = big_vision.sharding.infer_sharding(
            params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

        # Yes: Some donated buffers are not usable.
        warnings.filterwarnings(
            "ignore", message="Some donated buffers were not usable")

        @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
        def maybe_cast_to_f32(params, trainable):
            return jax.tree.map(lambda p, m: p.astype(jnp.float32) if m else p,
                                params, trainable)

        # Loading all params in simultaneous - albeit much faster and more succinct -
        # requires more RAM than the T4 colab runtimes have by default.
        # Instead we do it param by param.
        params, treedef = jax.tree.flatten(params)
        sharding_leaves = jax.tree.leaves(params_sharding)
        trainable_leaves = jax.tree.leaves(trainable_mask)
        for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
            params[idx] = big_vision.utils.reshard(params[idx], sharding)
            params[idx] = maybe_cast_to_f32(params[idx], trainable)
            params[idx].block_until_ready()
        params = jax.tree.unflatten(treedef, params)

        # Print params to show what the model is made of.
        def parameter_overview(params):
            for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
                print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

        print(" == Model params == ")
        parameter_overview(params)

        self.tokenizer = tokenizer
        self.decode = decode
        self.params = params
        self.data_sharding = data_sharding


    def preprocess_image(self, image, size=224):
        # Model has been trained to handle images of different aspects ratios
        # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
        # options are helpful to improve quality in some tasks.
        image = np.asarray(image)
        if image.ndim == 2:  # Convert image without last channel into greyscale.
            image = np.stack((image,)*3, axis=-1)
        image = image[..., :3]  # Remove alpha layer.
        assert image.shape[-1] == 3

        image = tf.constant(image)
        image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
        return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

    def preprocess_tokens(self, prefix, suffix=None, seqlen=None):
        # Model has been trained to handle tokenized text composed of a prefix with
        # full attention and a suffix with causal attention.
        separator = "\n"
        tokens = self.tokenizer.encode(prefix, add_bos=True) + self.tokenizer.encode(separator)
        mask_ar = [0] * len(tokens)    # 0 to use full attention for prefix.
        mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

        if suffix:
            suffix = self.tokenizer.encode(suffix, add_eos=True)
            tokens += suffix
            mask_ar += [1] * len(suffix)    # 1 to use causal attention for suffix.
            mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

        mask_input = [1] * len(tokens)    # 1 if it's a token, 0 if padding.
        if seqlen:
            padding = [0] * max(0, seqlen - len(tokens))
            tokens = tokens[:seqlen] + padding
            mask_ar = mask_ar[:seqlen] + padding
            mask_loss = mask_loss[:seqlen] + padding
            mask_input = mask_input[:seqlen] + padding

        return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

    def postprocess_tokens(self, tokens):
        tokens = tokens.tolist()  # np.array to list[int]
        try:  # Remove tokens at and after EOS if any.
            eos_pos = tokens.index(self.tokenizer.eos_id())
            tokens = tokens[:eos_pos]
        except ValueError:
            pass
        return self.tokenizer.decode(tokens)
    
    def make_predictions(self, image, prefix, *, seqlen=SEQLEN, sampler="greedy"):
        
        image = self.preprocess_image(image)

        prefix = prefix.lower()
        suffix = ""
        tokens, mask_ar, _, mask_input = self.preprocess_tokens(prefix, seqlen=self.SEQLEN)
        label, _, _, _ = self.preprocess_tokens(suffix, seqlen=self.SEQLEN)

        example = {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_input": np.asarray(mask_input),
        }

        examples = [example]
        examples[-1]["_mask"] = np.array(True)  # Indicates true example.
        
        # Convert list of examples into a dict of np.arrays and load onto devices.
        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, self.data_sharding)

        # Make model predictions
        tokens = self.decode({"params": self.params}, batch=batch,
                        max_decode_len=seqlen, sampler=sampler)

        # Fetch model predictions to device and detokenize.
        tokens, mask = jax.device_get((tokens, batch["_mask"]))
        tokens = tokens[mask]  # remove padding examples.
        texts = [self.postprocess_tokens(e["text"]) for e in examples]
        labels = [self.postprocess_tokens(e["label"]) for e in examples]
        prediction = [self.postprocess_tokens(t) for t in tokens]
        
        for example, text, label, pred,  in zip(examples, texts, labels, prediction):
            return example["image"], text, label, pred
            
def read_n_lines(file_path: str, n: int) -> List[str]:
    with open(file_path, 'r') as file:
        lines = [next(file).strip() for _ in range(n)]
    return lines


def get_model_base():
    import kagglehub
    MODEL_PATH = "./pt_224_128.params.f16.npz"
    if not os.path.exists(MODEL_PATH):
        print("Downloading the checkpoint from Kaggle, this could take a few minutes....")
        # Note: kaggle archive contains the same checkpoint in multiple formats.ROBOFLOW_API_KEY
        # Download only the float16 model.
        MODEL_PATH = kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224', 'paligemma-3b-pt-224.f16.npz')
        print(f"Model path: {MODEL_PATH}")

    model = VLAModel(MODEL_PATH)
    return model

def test_on_dataset():
    dataset_location = "/tmp/clevr-act-6-var-cam"
    #model_path = "/data/lmbraid19/argusm/models/clevr-act-6-var-cam_jax-native/model/paligemma-3b-pt-224-final.f16.npz"
    #model = VLAModel(model_path, gpu_index=1)  # takes a while
    

    dataset_location = Path(dataset_location)
    images = []
    lines = read_n_lines(dataset_location / "dataset/_annotations.train.jsonl", 25)
    for example_str in lines:
        example = json.loads(example_str)
        image = Image.open(dataset_location / "dataset" / example["image"])
        prefix = example["prefix"]
        set_trace()
        #res = model.make_predictions(image, prefix)
        set_trace()


def test_on_env():
    # get the model
    model_path = "/data/lmbraid19/argusm/models/clevr-act-6-var-cam_jax-native/model/paligemma-3b-pt-224-final.f16.npz"
    model = VLAModel(model_path, gpu_index=1)  # takes a while
    
    import tyro
    import os
    import json
    from mani_skill.examples.run_env import Args, iterate_env, save_dataset

    parsed_args = tyro.cli(Args)
    parsed_args.env_id = "ClevrMove-v1"
    parsed_args.render_mode = "rgb_array"
    parsed_args.control_mode = "pd_joint_pos"
    parsed_args.shader = "rt"
    

    dataset_path = Path("/data/lmbraid19/argusm/CLUSTER/cvla/clevr-act-6-cam2")
    #model_path = "/tmp/clevr-act-5/model/paligemma-3b-pt-224-step002730.f16.npz"
    #model_path = "/tmp/clevr-act-5/model/paligemma-3b-pt-224-final.f16.npz"
    seed_fn = "/ihome/argusm/lang/ManiSkill/mani_skill/examples/seeds_valid.json"
    parsed_args.record_dir = dataset_path / "record"

    with open(seed_fn, "r") as f_obj:
        seeds = json.load(f_obj)
    parsed_args.seed = seeds
    N_samples = len(seeds)

    # maniskill recording
    os.makedirs(parsed_args.record_dir, exist_ok=True)

    # cVLA recording
    dataset_path = Path(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    save_dataset(iterate_env(parsed_args, vis=False, model=model), N=int(N_samples), dataset_path=dataset_path)

    
if __name__ == "__main__":
    import json
    from pathlib import Path
    from pdb import set_trace

    #test_on_env()
    test_on_dataset()
    
    

    