# BrainMass (Brain network analysis via mask modeling and feature Alignment by Self-Supervised learning)

# Brain network foundation model

## Pre-training:

For pre-training, you should change the "path" and "csv" in the config file (e.g., config/normal_sch1.yaml)
   
   The "path" is the path of timeseries data
   
   The csv is the list of pre-training files, like:

        file,
        ukb_001,
        ukb_002,
        ...

(This code can also be trained for a single dataset and achieves promising performance.)

you can  run pre-training like:
```shell
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config config/normal_sch1.yaml
```

or use the checkpoint to pre-training like:
```shell
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config config/normal_sch1.yaml --resume --model_path checkpoints/checkpoints_test/test.pth 
```


## Downstream evaluation:
1. You need to change the root of the evaluation files in "02-eval_byot_svm_alldata.py" 43 to your path:

```python
root = "/path/to/alldata"
```
2. You need to prepare your own data in the down_stream folder:
   
   These csv files are like:

        new_nam,dx,is_train,site
        xuanwu_001,0,1,0
        xuanwu_002,1,0,1
        ...

3. For downstream evaluation, you can run like:
```shell
python 02-eval_byot_svm_alldata.py -c config/normal_sch1.yaml -d abide1 -f checkpoints/checkpoints_test/test.pth
```


We find that different fMRI preprocessing steps might also lead to performance fluctuations. 
