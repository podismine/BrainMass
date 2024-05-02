import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BNTF, MLPHead, get_sinusoid_encoding_table

import torch.distributed as dist
import random
def accuracy(output,target,top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res
class BYOLTrainer(nn.Module):
    def __init__(self,depth,heads,m,feature_dim,dim_feedforward,mm,clf_mask=10, mse_mask=5):
        super().__init__()
        self.m = m
        self.model_mode = mm
        self.online_network = BNTF(feature_dim,depth,heads,dim_feedforward)
        self.target_network = BNTF(feature_dim,depth,heads,dim_feedforward)
        self.predictor = MLPHead(feature_dim, feature_dim*2,feature_dim)

        roi_num = 100
        self.roi_num = roi_num

        # for mcl
        self.mcl_mask = clf_mask
        self.mrm_mask = mse_mask
        self.token_num = 2

        self.mask_embed = nn.Parameter(torch.zeros([1, 1, roi_num]))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, roi_num))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, roi_num))

        self.pos_embed = get_sinusoid_encoding_table(roi_num + self.token_num, roi_num)

        self.norm =  nn.LayerNorm(roi_num)

        self.cpred = nn.Sequential(nn.Linear(roi_num, dim_feedforward), nn.LeakyReLU(), nn.Linear(dim_feedforward, roi_num)) #256
        self.gpred = nn.Sequential(nn.Linear(roi_num, dim_feedforward), nn.LeakyReLU(), nn.Linear(dim_feedforward, roi_num)) #1024
        
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        self.init_params()
        self.check_values()

    
    def init_params(self):
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)
        self.cls_token = torch.nn.init.xavier_normal_(self.cls_token)
        self.dist_token = torch.nn.init.xavier_normal_(self.dist_token)

    def check_values(self):
        if self.model_mode not in ["byol","byol+clf","byol+mse","byol+clf+mse",'moco', 'rrp',"moco+clf+mse"]:
            raise KeyError(f"{self.model_mode} value error, should be in [byol,byol+clf,byol+mse,byol+clf+mse]")

        if self.mcl_mask > self.roi_num:
            raise KeyError(f"{self.mcl_mask} value error, mcl_mask should be smaller than roi_num")

        if self.mrm_mask > self.roi_num:
            raise KeyError(f"{self.mrm_mask} value error, mrm_mask should be smaller than roi_num")

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward_mcl(self,x):
        B, T, C = x.shape
        mask = self.mcl_mask
        device = x.device
        token_num = 2

        encode_samples = torch.empty((B,mask,C),device = device, requires_grad=False).float()
        mask_index = torch.empty((B,mask),device = device, requires_grad=False).long()
        mask_dense = torch.ones([B,T,C],device = device)

        for i in range(B):
            mask_id = torch.tensor(random.sample(range(0, T), mask))
            mask_index[i] = mask_id
            encode_samples[i] = x[i, mask_index[i]].clone().detach()
            mask_dense[i,mask_index[i]] = 0

        mask_tokens = self.mask_embed.expand(B, T, -1)

        new_x = x * mask_dense + (1-mask_dense) * mask_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        dist_tokens = self.dist_token.expand(B, -1, -1) 
        new_x = torch.cat((cls_tokens,dist_tokens, new_x), dim=1)
        new_x = new_x + self.pos_embed.type_as(new_x).to(x.device).clone().detach()

        x_vis = self.online_network(new_x,forward_with_mlp=False)

        pred = torch.empty((B,mask, C),device = device).float()
        for i in range(B):
            pred[i]=self.cpred(x_vis[i,mask_index[i]+token_num])
        
        nce = torch.tensor(0.0).to(device)
        correct = torch.tensor(0.0).to(device)

        for i in range(B):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask, device=device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor

        acc = 1. * correct / (B * mask)
        nce = nce / (-1. * B * mask)

        return acc, nce

    def forward_mrm(self,x):
        B, T, C = x.shape
        mask = self.mrm_mask
        device = x.device
        token_num = 2

        mask_index = torch.empty((B,mask),device = device, requires_grad=False).long()
        mask_dense = torch.ones([B,T,C],device = device)

        for i in range(B):
            mask_id = torch.tensor(random.sample(range(0, T), mask))
            mask_index[i] = mask_id
            mask_dense[i,mask_index[i]] = 0

        mask_tokens = self.mask_embed.expand(B, T, -1)

        new_x = x * mask_dense + (1-mask_dense) * mask_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        dist_tokens = self.dist_token.expand(B, -1, -1) 
        new_x = torch.cat((cls_tokens,dist_tokens, new_x), dim=1)
        new_x = new_x + self.pos_embed.type_as(new_x).to(x.device).clone().detach()

        x_vis = self.online_network(new_x,forward_with_mlp=False)

        pred = torch.empty((B,mask, C),device = device).float()
        target = torch.empty((B,mask, C),device = device).float()
        for i in range(B):
            pred[i]=self.gpred(x_vis[i,mask_index[i]+token_num])
            target[i] = x[i, mask_index[i], :]

        mse = torch.mean((pred - target) ** 2)
        
        return mse

    def forward(self, batch_view_1, batch_view_2,returns = 'all'):
        
        if self.model_mode == 'byol':
            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
            acc = 0.
            nce = 0.
            mse = 0.
            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse
        elif self.model_mode == 'rrp':
            predictions_from_view_1 = self.online_network(batch_view_1)
            output = self.clf(predictions_from_view_1)
            loss = nn.CrossEntropyLoss()(output, batch_view_2)
            acc = accuracy(output, batch_view_2[:, 1])[0]
            return loss.mean(), acc, 0., 0.
        elif self.model_mode == 'moco':
            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
            acc = 0.
            nce = 0.
            mse = 0.
            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.contrastive_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.contrastive_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse

        elif self.model_mode == 'byol+clf':
            acc,nce = self.forward_mcl(batch_view_1)
            
            mse = 0.

            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.contrastive_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.contrastive_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse

        elif self.model_mode == 'byol+mse':
            acc, nce = 0., 0.
            
            mse = self.forward_mrm(batch_view_1)

            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse

        elif self.model_mode == 'byol+clf+mse':
            acc,nce = self.forward_mcl(batch_view_1)
            
            mse = self.forward_mrm(batch_view_1)

            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse

        elif self.model_mode == 'moco+clf+mse':
            acc,nce = self.forward_mcl(batch_view_1)
            
            mse = self.forward_mrm(batch_view_1)

            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

            with torch.no_grad():
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)

            loss = self.contrastive_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.contrastive_loss(predictions_from_view_2, targets_to_view_2)

            return loss.mean(),acc,nce,mse
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / 0.07
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * 0.07)

    @torch.no_grad()
    def initialize_target(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def get_parameters(self):
        return list(self.online_network.parameters()) + list(self.predictor.parameters())

    def save_model(self, path):
        try:
            if dist.get_rank() == 0:
                torch.save({
                    'model':self.state_dict(),
                    'online_network_state_dict': self.online_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                }, path)
        except:
            torch.save({
                    'model':self.state_dict(),
                    'online_network_state_dict': self.online_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                }, path)
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output