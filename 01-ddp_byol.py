import torch
import yaml
from network_dataset import Task1Data
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from byol_trainer import BYOLTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import  DistributedSampler

from timm.utils import NativeScaler

import random
import numpy as np

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ddp", action = "store_true",default = False)
    parser.add_argument("--resume", action = "store_true",default = False)
    parser.add_argument("--config", type=str,default = "")
    parser.add_argument("--model_path", type=str,default = "")
    parser.add_argument("--use_ddp",type=bool)
    parser.add_argument("--local-rank", default=-1)

    FLAGS = parser.parse_args()
    FLAGS.use_ddp = not FLAGS.no_ddp
    return FLAGS

def init_ddp(FLAGS):
    local_rank = FLAGS.local_rank
    torch.cuda.set_device(int(local_rank))
    dist.init_process_group(backend='nccl') 

def adjust_learning_rate(optimizer, epoch, final_lr,warmup_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr_warmup = [i / warmup_epochs * final_lr for i in range(1, int(warmup_epochs +1))]
    if epoch < warmup_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup[epoch]
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr

def main():
    FLAGS = get_parse()
    if FLAGS.use_ddp is True:
        print("Init ddp")
        init_ddp(FLAGS)

    config = yaml.load(open(FLAGS.config, "r"), Loader = yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = str(config['data']['path'])
    csv_path = str(config['data']['csv'])
    time_len = int(config['data']['time_len'])
    mask_len = int(config['data']['time_mask'])
    mask_way = str(config['data']['mask_way'])
    train_dataset = Task1Data(root = data_path,csv = csv_path, mask_way=mask_way,mask_len=mask_len, time_len=time_len)
    if FLAGS.use_ddp is True:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'],num_workers=config['trainer']['num_workers'],pin_memory=True,sampler = train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'],num_workers=config['trainer']['num_workers'])

    feature_size = config['network']['feature_dim']
    depth = config['network']['depth']
    heads = config['network']['heads']
    dim_feedforward = config['network']['dim_feedforward']
    mm = str(config['network']['mm'])
    clf_mask = int(config['network']['clf_mask'])
    mse_mask = int(config['network']['mse_mask'])
    model = BYOLTrainer(depth,heads,config['trainer']['m'],feature_size,dim_feedforward,mm=mm, clf_mask = clf_mask, mse_mask = mse_mask)
    model.cuda().train()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has parameters: {n_parameters / 1e6}M")
    model_checkpoints_folder = config['saving']['checkpoint_dir']

    log_dir = config['saving']['log_dir']

    if FLAGS.resume is True:
        checkpoint = torch.load(FLAGS.model_path, map_location = 'cpu')
        model.load_state_dict(checkpoint['model'])

    if FLAGS.use_ddp is True:
        model = DDP(model, find_unused_parameters=True)
        model_call = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model_call = model

    optimizer = torch.optim.AdamW(model_call.get_parameters(),lr= config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])

    if not os.path.exists(model_checkpoints_folder) and dist.get_rank() == 0:
        os.makedirs(model_checkpoints_folder)
 
    loss_scaler = NativeScaler()
    
    if FLAGS.use_ddp and dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    model_call.initialize_target()
    best_train_loss = 99999.

    acc_lambda = float(config['trainer']['acc_lambda'])
    mse_lambda = float(config['trainer']['mse_lambda'])
    warmup_epochs = int(config['trainer']['warmup_epochs'])

    # for test
    #for epoch_counter in range(1):
    for epoch_counter in range(config['trainer']['max_epochs']):
        epoch_counter = epoch_counter #+ epoch_from
        model.train()
        if FLAGS.use_ddp is True:
            train_loader.sampler.set_epoch(epoch_counter)
        if FLAGS.resume is not True:
            adjust_learning_rate(optimizer,epoch_counter,config['optimizer']['lr'],warmup_epochs)

        header = 'Epoch: [{}]'.format(epoch_counter)

        n_steps = 50

        total_loss = 0.
        byol_loss = 0.
        nce_loss = 0.
        mcl_acc = 0.

        count = 0.
        niter = 0.

        st = time.time()
        calc_st = time.time()

        for step, (batch_view_1, batch_view_2) in enumerate(train_loader):
            B = len(batch_view_1)

            batch_view_1 = batch_view_1.to(device,non_blocking=True).float()
            batch_view_2 = batch_view_2.to(device,non_blocking=True).float()

            loss_byol, acc, nce, mse = model(batch_view_1, batch_view_2)

            if mm == 'byol':
                loss =  loss_byol
            elif mm == 'byol+clf':
                loss =  loss_byol + acc_lambda* nce
            elif mm == 'byol+mse':
                loss =  loss_byol + mse_lambda* mse
            else:
                loss = loss_byol + acc_lambda* nce + mse_lambda * mse

            optimizer.zero_grad()

            loss_scaler(loss, optimizer, parameters=model.parameters(),clip_grad=1,clip_mode='value')

            #loss.backward()
            #optimizer.step()

            model_call.update_target()  # update the key encoder

            total_loss += len(batch_view_1) * float(loss)
            byol_loss += len(batch_view_1) * float(loss_byol)
            nce_loss += len(batch_view_1) * float(nce)
            mcl_acc += len(batch_view_1) * float(acc)

            count += len(batch_view_1)
            if FLAGS.use_ddp is False or (FLAGS.use_ddp is True and dist.get_rank() == 0):
                if step % n_steps == 0:
                    end = time.time()
                    print(f"Epoch: {epoch_counter} [{step}/{len(train_loader)}]: byol: {loss_byol:.5f}, nce: {nce:.5f}, mse: {mse:.5f} time: {end-st}")
                    st = time.time()
            if step %n_steps == 0 and step != 0:
                if FLAGS.use_ddp is False or (FLAGS.use_ddp is True and dist.get_rank() == 0):
                    need_time = (time.time() - calc_st) /n_steps * len(train_loader)/60./60.
                    print(f"precalc time: {need_time} hours by batch: {config['trainer']['batch_size']}")
                    calc_st = time.time()
        total_loss /= count
        byol_loss /= count
        nce_loss /= count
        mcl_acc /= count


        if writer is not None:

            writer.add_scalar('Acc', mcl_acc, global_step=epoch_counter)
            writer.add_scalar('Nce', nce_loss, global_step=epoch_counter)
            writer.add_scalar('MSE', mse, global_step=epoch_counter)
            writer.add_scalar('byol_loss', byol_loss, global_step=epoch_counter)
            writer.add_scalar('total_loss', total_loss, global_step=epoch_counter)

        if total_loss <= best_train_loss:
            best_train_loss = total_loss
            model_call.save_model(os.path.join(model_checkpoints_folder, 'best_model.pth'))
            
        if epoch_counter % config['saving']['n_epochs'] == 0:
            model_call.save_model(os.path.join(model_checkpoints_folder, f'model_{epoch_counter}.pth'))

    # save checkpoints
    model_call.save_model(os.path.join(model_checkpoints_folder, 'last_model.pth'))
    
if __name__ == '__main__':
    main()