## Data parquet creation
Using your llava dataset created with RadVLM repo, you can create the parquet data files by executing the following command (change as you like):
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic
```

## Reward function 
The reward function that worked for me using BLEU metric was created here: `custom_rewards/bleu_reward.py`. You can create other reward files, always specifying the name of the function that calls it (currently `compute_score` in the training scripts). 

## Slurm script 
You will find what worked for me until now for running GRPO in `jobs` folder. 

## Checkpoint conversion to HF
```
python -m verl.model_merger merge --backend fsdp --local_dir /your/checkpoint/path/global_step_X/actor --target_dir /your/checkpoint/path/huggingface/
```



