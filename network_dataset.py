#coding:utf8
import os
from torch.utils import data
import numpy as np
import nibabel as nib
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import warnings
from nilearn.connectome import ConnectivityMeasure
from sklearn.utils  import shuffle

warnings.filterwarnings("ignore")

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):

        self.input_size = input_size

        self.num_patches = self.input_size
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

def mask_timeseries(timeser, mask = 30):
    rnd = np.random.random()
    time_len = timeser.shape[1]
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]
def mask_timeseries_per(timeser, mask = 30):
    rnd = np.random.random()

    time_len = timeser.shape[1]
    mask_len = int(mask * time_len /100)
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask_len))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]

def random_timeseries(timeser,sample_len):
    time_len = timeser.shape[1]
    st_thres = 1
    if time_len <= sample_len + st_thres:
        return timeser

    select_range = time_len - sample_len
    if select_range < 1:
        return timeser

    st = random.sample(list(np.arange(st_thres,select_range)),1)[0]
    return timeser[:,st:st+sample_len]

class Task1Data(data.Dataset):

    def __init__(self, root = None,csv = None, mask_way='mask',mask_len=10, time_len=30):
        self.template = 'sch'
        self.root = root
        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len
        df = pd.read_csv(csv)
        self.names = list(df['file'])

        print(f"Finding files: {len(self.names)}")
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __getitem__(self,index):
        name = self.names[index]
        img = np.load(os.path.join(self.root, name))
        if self.mask_way == 'mask':
            slices = [mask_timeseries(img,mask=self.mask_len).T, mask_timeseries(img,mask=self.mask_len).T]
        elif self.mask_way == 'mask_per':
            slices = [mask_timeseries_per(img,mask=self.mask_len).T, mask_timeseries_per(img,mask=self.mask_len).T]
        elif self.mask_way == 'random':
            slices = [random_timeseries(img,sample_len=self.time_len).T, random_timeseries(img,sample_len=self.time_len).T]
        else:
            raise KeyError(f"mask way error, your input is {self.mask_way}")
        correlation_matrix = self.correlation_measure.fit_transform(slices)
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix[0], correlation_matrix[1]

    def __len__(self):
        return len(self.names)
        
class Task3Data(data.Dataset):

    def __init__(self, root= None, csv = None, mask_way='mask',mask_len=10, time_len=30,shuffle_seed=42,is_train = True, is_test = False):
        self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        self.root = root

        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len

        self.df = pd.read_csv(csv)

        self.names = list(self.df['new_name'])
        test_length = int(len(self.df) * 0.15)

        all_data = np.array(self.names)
        lbls = np.array(list([1 if f == 1 else 0 for f in self.df['dx'] ]))
        sites = np.array(self.df['site']) if 'site' in self.df.columns else lbls
        train_index = self.df[self.df['is_train']==1].index
        rest_index = self.df[self.df['is_train']==0].index

        data_train = all_data[train_index]
        labels_train = lbls[train_index]

        rest_data = all_data[rest_index]
        rest_site = sites[rest_index]
        rest_label = lbls[rest_index]


        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(rest_data, rest_site):
            data_test, labels_test = rest_data[test_index], rest_label[test_index]
            data_val, labels_val = rest_data[valid_index], rest_label[valid_index]

        if is_test is True:
            print("Testing data:")
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
            # self.imgs, self.lbls = np.concatenate([data_train, data_val],0), np.concatenate([labels_train, labels_val],0),
        else:
            print("Val data:")
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')


    def __getitem__(self,index):
        name = self.imgs[index]
        lbl = self.lbls[index]
        img = np.load(os.path.join(self.root, f"{self.template}_{name}.npy"))
        if self.is_train is True:
            if self.mask_way == 'mask':
                slices = [mask_timeseries(img,self.mask_len).T]
            elif self.mask_way =='random':
                slices = [random_timeseries(img,self.time_len).T]
            elif self.mask_way =='mask_per':
                slices = [mask_timeseries_per(img,mask=self.mask_len).T]
            else:
                slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        elif self.is_test is False:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        else:
            # slices = [img.T]
            slices = [mask_timeseries_per(img,mask=self.mask_len).T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix,onehot_lbl
    
    def __len__(self):
        return len(self.imgs)