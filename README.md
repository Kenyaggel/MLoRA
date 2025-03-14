# MoE-MLoRA for Multi-Domain CTR Prediction: Efficient Adaptation with Expert Specialization


## Our Team
* Aviel Ben Siman Tov
* Ken Yagel
* Eyal German

## Requirements

```
tensorflow-gpu==1.12.0
requests==2.26.0
tqdm==4.62.2
pandas==1.1.5
scikit-learn==0.24.2
numpy==1.16.6
deepctr==0.9.0
RTX 2080Ti + 64G RAM
python:3.6
Ubuntu 20.04
```

## Dataset Preprocess

[//]: # (### Amazon dataset)

[//]: # ()
[//]: # (Enter `dataset/Amazon`)

[//]: # ()
[//]: # (* Download raw dataset at : [Amazon review data]&#40;https://nijianmo.github.io/amazon/index.html#complete-data&#41;)

[//]: # (* unzip the dataset into `raw_data`)

[//]: # ()
[//]: # (* the split rule in `config_*.json`, such as `config_6.json`)

[//]: # (* run `split_py --config config_*.json`, it will automatically split the domains and create the dataset.)

### Taobao dataset

Enter `dataset/Taobao`

* Download Taobao raw dataset at : https://tianchi.aliyun.com/dataset/9716.
* unzip the dataset into `raw_data`
  1. theme_click_log.csv
  2. theme_item_pool.csv
  3. user_item_purchase_log.csv
  4. item_embedding.csv
  5. user_embedding.csv
* change the dataset config in `config_*.json`. theme_num = -1 denotes using all domains.
* run `split_py --config config_*.json`, it will automatically split the domains and create the dataset.

### Movielens dataset

Enter `dataset/Movielens

- Download Movielens raw dataset at : [MovieLens | GroupLens](https://grouplens.org/datasets/movielens/), named MovieLens 1M Dataset.

- unzip the dataset into `raw_data`
  1. users.dat
  2. movies.dat
  3. ratings.dat
- change the dataset config in `config_*.json`. 
- run `split_py --config config_*.json`, it will automatically split the domains and create the dataset.

## Run experiments

The model configuration files are located in the `config` folder, categorized into three datasets: Amazon, Taobao, and Movielens. Each base model is divided into the original model and the injected MLoRA versions. For example, `nfm.json` and `nfm_mlora.json`. `nfm.json` represents the original NFM model, while `nfm_mlora.json` signifies the NFM model with the help of MLoRA

### Run MLoRA baseline

```
python run.py config/Taobao_10/nfm_mlora.json
python run.py config/Taobao_10/dcn_mlora.json
python run.py config/Taobao_10/deepfm_mlora.json
python run.py config/Taobao_10/wdl_mlora.json
......
```

### Run our MoE-MLoRA

```
python run_moe.py config/Taobao_10/nfm_mlora.json
python run_moe.py config/Taobao_10/dcn_mlora.json
python run_moe.py config/Taobao_10/deepfm_mlora.json
python run_moe.py config/Taobao_10/wdl_mlora.json
......
```

#### Running Experiments
To run an experiment, use:
```
python run_experiments.py --jobid <job_id>
```
Job ID Mapping:
* **Job ID 1-3**: Runs experiments on the **Movielens** dataset, varying the number of experts and domain splits (by gender, age, or occupation).
* **Job ID 4**: Runs an experiment on the **Taobao_10** dataset with a theme-based domain split.
* **Job ID 5**: Tests different expert configurations on **Movielens** (split by gender).


To run all experiments using SLURM, submit a job array (1-5), and run:
```
python run_experiments.py
```

### Config example

```json
{
  "model": {
    "name": "autoint_mlora",
    "norm": "none",
    "dense": "dense",
    "lora_r": -1,
    "lora_reduce":16,
    "num_experts": 2,
    "auxiliary_net": false,
    "user_dim": 128,
    "item_dim": 128,
    "domain_dim": 128,
    "auxiliary_dim": 128,
    "hidden_dim": [
      256,
      128,
      64
    ],
    "dropout": 0.5
  },
  "train": {
    "train_from_ckp": false,
    "load_pretrain_emb": true,
    "emb_trainable": false,
    "epoch": 99999,
    "learning_rate": 0.001,
       "meta_learning_rate": 0.1,
    "domain_meta_learning_rate": 0.1,
    "merged_method": "plus",
    "sample_num": 5,
    "add_query_domain": true,
    "finetune_every_epoch": false,
    "shuffle_sequence": true,
    "meta_sequence": "random",
    "target_domain": -1,
    "domain_regulation_step": 0,
    "meta_train_step": 0,
    "meta_finetune_step": 0,
    "meta_split": "train-train",
    "meta_split_ratio": 0.8,
    "average_meta_grad": "none",
    "meta_parms": [
      "emb",
      "kernel_shared",
      "bias_shared"
    ],
    "result_save_path": "result",
    "checkpoint_path": "checkpoint",
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "patience": 3,
    "val_every_step": 1,
    "histogram_freq": 0,
    "shuffle_buff_size": 10000
  },
  "dataset": {
    "name": "Taobao",
    "dataset_path": "dataset/Taobao",
    "domain_split_path": "split_by_theme_10",
    "batch_size": 1024,
    "shuffle_buffer_size": 10000,
    "num_parallel_reads": 8,
    "seed": 123
  }
}
```

### Config explanation

```
"name": 'model_single' represents the original deep learning model, while 'model_lora' represents the deep learning model with MLoRA.
"num_experts": The number of experts in the model. It must be at least equal to the number of domains in the dataset. A higher value allows multiple experts per domain.
"lora_r": the rank of LoRA
"lora_reduce": LoRA's temperature coefficient \alpha in our paper.
"user_dim", "item_dim" and "domain_dim": The vector dimensions, the Taobao dataset provides user embeddings and item embeddings, both with a dimension of 128. The dimensions for the other two datasets can be modified as needed.
"hidden_dim": The dimensions of the three-layer MLP, are set to [256, 128, 64] by default.
"load_pretrain_emb“ and "emb_trainable": In Taobao dataset, user embeddings and item embeddings are provided, with load_pretrain_emb is "True" and "emb_trainable" is "False"
"patience": Our training strategy will save the parameters of the model with the best performance during training. 
Training would stop when the model's performance failed to exceed the best performance achieved in previous training epochs for a consecutive number of times. More details in our paper.
```

## Acknowledgments

Parts of this project utilize code from the following open-source projects:

1. **MLoRA**: [Repository URL](https://github.com/gaohaining/MLoRA)
2. **DeepCTR**: [Repository URL](https://github.com/shenweichen/DeepCTR)
2. **MAMDR**: [Repository URL](https://github.com/RManLuo/MAMDR)

   