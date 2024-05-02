# BrainMass (Brain network analysis via mask modeling and feature Alignment by Self-Supervised learning)

Brain network foundation model


We currently upload the training code. This code can also be trained for a single dataset and achieves promising performance.

you can run the code like:
```shell
python -m torch.distributed.launch --nproc_per_node=8 01-ddp_byol.py --config config/normal_sch1.yaml
```


We are now cleaning the code and try to upload the checkpoints (qwq poor network)


We find that different fMRI preprocessing steps might also lead to performance fluctuations. 
We would provide a way for you to fine-tune using your own data using LORA or just fine-tuning the whole model.
