# CVLA Training


## Copying dataset to node
```
# on cluster machin 
cd /tmp
rsync -a --progress /data/lmbraid19/argusm/datasets/clevr-act-*.zip .
```

## Running training

Create a conda env (python version 3.12 because of tensorboard)
```
conda create env -n paligemma python=3.12
```

1. Use VS Code, set up remote connection to e.g. bud
2. Install Python and Jupyter packages on remote and set notebook to `paligemma`
3. Run notebook and follow initial instructions for Kaggle / model access
3. Get a kaggle key and create `~/.kaggle/kaggle.json` with `{"username":"XXX","key":"YYY"}`
3. Run `./cVLA/finetune_paligemma_robotflow.ipynb`


## Running motion planning

```
conda activate maniskill
cd project_dir/ManiSkill/mani_skill/examples
python gen_dataset.py -e "ClevrMove-v1"   --render-mode="rgb_array" -c "pd_joint_pos"
python -m mani_skill.examples.motionplanning.panda.run --env-id ClevrMove-v1 --traj-name="trajectory" --only-count-success --save-video -n 1
```


## Dataset Creation

See the ManiSkill repo, ManiSkill/mani_skill/examples/README_cvla.md. 


