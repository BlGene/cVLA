# CVLA Training

## Install

Create a conda env (python3.12) and install both my ManiSkill version and cVLA in the same directory.
All versions should be flexible other than `transformers` and `accelerate` with which I've had issues.

```
cd {WORKING_DIR}
conda create -n "paligemma" python=3.12
conda activate paligemma
git@github.com:BlGene/ManiSkill.git
cd ManiSkill
pip install -e .
cd ..
git clone git@github.com:BlGene/cVLA.git
cd cVLA
pip install -e .  # has not been tested -.-
```

## Copying Datasets

Before training data must be moved to the local machine, otherwise it will be too slow. The data is located on the network.
Current dataset is e.g. `clever-act-7-depth` or `clevr-act-7-small` to test a few images.

```
# ssh cluster_machine
rsync -a --progress /data/lmbraid19/argusm/datasets/indoorCVPR_09.tar /tmp/ && mkdir -p /tmp/indoorCVPR && tar -xf /tmp/indoorCVPR_09.tar -C /tmp/indoorCVPR
rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-act-7-depth /tmp/
file /tmp/indoorCVPR
file /tmp/clevr-act-7-depth
```

## Running Inference and Training

1. In case you want to use `VS Code`, or something to work on remote machines.
2. Create a huggingface account, and generate a token. Don't forget to add read-rights.
3. Agree to agree to Paligemma2 licence (https://huggingface.co/google/paligemma2-3b-pt-224)
4. For the first run, uncomment the following lines `#from huggingface_hub import notebook_login; notebook_login()`
5. Now you can do an evaluation or training run. 

Current configurations are  is e.g. `hf_finetune_paligemma2.ipynb` and `hf_finetune_paligemma2-depth.ipynb`.

```
conda activate paligemma
jupyter-lab {WORKING_DIR}/cVLA/hf_eval_notebook.ipynb  # evaluation
jupyter-lab {WORKING_DIR}/cVLA/hf_finetune_paligemma2.ipynb  # traning
```

## Running Online-Evaluation

Once a model has been obtained, it can be evaluated in simulation.

```
conda activate maniskill
cd project_dir/ManiSkill/mani_skill/examples
python gen_dataset.py -e "ClevrMove-v1"   --render-mode="rgb_array" -c "pd_joint_pos"
python -m mani_skill.examples.motionplanning.panda.run --env-id ClevrMove-v1 --traj-name="trajectory" --only-count-success --save-video -n 1
```


## Cluster

This is some code to enable training via jupyter notebooks on cluster machines, by creating an ssh tunnel.

```
ssh kislogin3.rz.ki.privat
srun -p lmbdlc2_gpu-l40s --gpus-per-node=1 --mem=40G --time=3:00:00 --pty bash --nodelist=dlc2gpu08
conda activate paligemma
cd /ihome/argusm/lang/cVLA/
jupyter-lab hf_finetune_paligemma2-l40.ipynb
# copy files in notebook not, ssh.
rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-real-1of5c-v1 /tmp/
rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-act-7-depth /tmp/
ssh -L 8890:localhost:8888 argusm@dlc2gpu08  # ssh -L local_port:remote_host:remote_port user@remote_server
```

## Dataset Creation

To create new and different datasets see my ManiSkill repo, specifically `ManiSkill/mani_skill/examples/README_cvla.md`.
