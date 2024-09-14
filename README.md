# BrainMass (Brain network analysis via mask modeling and feature Alignment by Self-Supervised learning)

Brain network foundation model

preparation:

you need to change the path of data in 01-ddp_byol.py and 02-eval_byot_svm_alldata.py

We currently upload the training code. This code can also be trained for a single dataset and achieves promising performance.

1. you can run preprocessing like:
```shell
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config config/normal_sch1.yaml
```

2. or use the checkpoint to preprocessing like:
```shell
python -m torch.distributed.launch --nproc_per_node=4 01-ddp_byol.py --config config/normal_sch1.yaml --resume --model_path checkpoints/checkpoints_test/test.pth 
```

3. For downstream evaluation, you can run like:
```shell
python 02-eval_byot_svm_alldata.py -c config/normal_sch1.yaml -d abide1 -f 
```


We find that different fMRI preprocessing steps might also lead to performance fluctuations. 
We would provide a way for you to fine-tune using your own data using LORA or just fine-tuning the whole model.
