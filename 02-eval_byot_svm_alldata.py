from network_dataset import Task2Data
import torch
import yaml
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from utils import BNTF, MLPHead
import random
import argparse
from sklearn.svm import SVC 
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from utils import get_data

parser = argparse.ArgumentParser()
parser.add_argument("--seed",'-s', type=int,default = 42)
parser.add_argument("--config_path",'-c', type=str,default = "")
parser.add_argument("--data",'-d', type=str,default = "")
parser.add_argument("--csv",'-f', type=str,default = "")

args = parser.parse_args()
args = get_data(args)


shuffle_seed = int(args.seed) #42
batch_size = 64
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = args.config_path 


config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

root = "/path/to/alldata"
mask_way=config['data']['mask_way']
mask_len=int(config['data']['time_mask'])
time_len=int(config['data']['time_len'])

train_dataset = Task2Data(root, args.csv, mask_way,mask_len,time_len,shuffle_seed=shuffle_seed,is_train=True,is_test=False)
val_dataset = Task2Data(root, args.csv,mask_way,mask_len,time_len,shuffle_seed=shuffle_seed,is_train=False,is_test=False)
test_dataset = Task2Data(root, args.csv,mask_way,mask_len,time_len,shuffle_seed=shuffle_seed,is_train=False,is_test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=4, drop_last=False, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=4, drop_last=False, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          num_workers=4, drop_last=False, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = config['network']['feature_dim']
depth = config['network']['depth']
heads = config['network']['heads']
dim_feedforward = config['network']['dim_feedforward']

encoder = BNTF(feature_size,depth,heads,dim_feedforward).to(device)

# test.pth for test. best_model.pth should be used
load_params = torch.load(os.path.join(config['saving']['checkpoint_dir'],'test.pth'), map_location='cpu')['online_network_state_dict']
encoder.load_state_dict(load_params)
print("Parameters successfully loaded.")

encoder = encoder.to(device)
def get_features_from_encoder(encoder, loader,times = 1):
    
    x_train = []
    y_train = []
    encoder.eval()
    for _ in range(times):
    # TTA is not used
        for i, (x, y) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).long()
            with torch.no_grad():
                bz, _, _, = x.shape
                for atten in encoder.attention_list:
                    x = atten(x)
                node_feature = encoder.dim_reduction(x)
                feature_vector = node_feature.reshape((bz, -1))
                x_train.extend(feature_vector.detach())
                y_train.extend(y.detach())
    x_train = torch.stack(x_train).detach().cpu()
    y_train = torch.stack(y_train).detach().cpu()
    return x_train, y_train

encoder.eval()
clf = SVC(probability=True) # 5

x_train, y_train = get_features_from_encoder(encoder, train_loader,1)
x_val, y_val = get_features_from_encoder(encoder, val_loader,1) 
x_test, y_test = get_features_from_encoder(encoder, test_loader,1)
print("Loading features loaded.")

x_train = x_train.detach().cpu().numpy()
x_val = x_val.detach().cpu().numpy()
x_test = x_test.detach().cpu().numpy()

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

clf.fit(x_train, y_train.detach().cpu().numpy()[:,1])
pred_test = clf.predict(x_test)

acc = accuracy_score(pred_test, y_test.detach().cpu().numpy()[:,1])
cm = confusion_matrix(pred_test, y_test.detach().cpu().numpy()[:,1])
sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)

res_string = f"acc: {acc:.4f}  sen: {sen:.4f} spe: {spe:.4f}"
print(res_string)
with open(f"res.txt", 'a') as f:
    f.write(f"data:[{args.data}] \t seed:[{shuffle_seed}] \t {res_string} \n")