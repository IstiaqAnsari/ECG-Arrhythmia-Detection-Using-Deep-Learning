import numpy as np, pandas as pd,os
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import random_split
from Sampler import BalancedBatchSampler
class MITData(Dataset):
    def __init__(self, root_dir,filename,classes = 18):
        self.classes = classes
        self.path = os.path.join(root_dir,filename)
        data_store = pd.HDFStore('data/Data_RRR.h5')
        # Put DataFrame into the object setting the key as 'preprocessed_df'
        self.df = data_store['df']
        data_store.close()
    def __len__(self):
        return len(self.df.index)
    def __getitem__(self, idx):
        x = self.df.iloc[idx].values
        y = int(x[-1])
        x = x[:-1].reshape(1,256)
        return x,y

def getDataLoader(root_dir='data',filename='DataWithClass.h5',split=1,batch_size=126,sampled = True,shuffle=False):

    if sampled:shuffle = False
    else: shuffle = True
    dataset = MITData(root_dir,filename)
    print("Total Data Samples ",dataset.__len__())
    train_size = int(split* dataset.__len__())
    test_size = dataset.__len__() - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    if sampled:
    	train_loader = DataLoader(train_dataset,sampler=BalancedBatchSampler(train_dataset),batch_size=batch_size)
    else:
    	train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    if split<1:
    	test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
    	return train_loader,test_loader
    else : return train_loader
