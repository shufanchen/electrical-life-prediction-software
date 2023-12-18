import os
import pandas as pd
from torch.autograd import Variable
import torch
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn import preprocessing
from torch.utils.data import TensorDataset,DataLoader
import torch.utils.data.dataloader as Data
import joblib

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

def time_series_windowize(time_series,seqLen):
    """每seqLen个样本拼接成一个新的序列,这个新的序列作为一个大样本，一个大样本配对一个RUL"""
    window_data = []
    for i in range(seqLen, len(time_series)):
        window_data.append(time_series[i - seqLen:i, :])
    return window_data

def make_loader(file_directory,seqLen,batch_size,input_size,model_type):
    """将数据装进loader,方便后续提取batch"""
    data_raw = pd.read_csv(file_directory)

    X, Y = data_raw.iloc[:, [i for i in range(24)] + [25, 26, 27, 28]], data_raw.iloc[:, [24]]

    selector = SelectKBest(mutual_info_regression, k=input_size)
    selector.fit(X, Y)
    joblib.dump(selector,'./model/contactor/'+model_type+'_con_feature_select')
    X = selector.transform(X)  # 特征选择
    standardscaler = preprocessing.StandardScaler()
    standardscaler.fit(X)
    joblib.dump(standardscaler,'./model/contactor/'+model_type+'_con_standardscaler')
    X = standardscaler.transform(X)
    training_Set_length = int(len(data_raw) * (5 / 6))
    X_train = X[:training_Set_length, :]

    X_train_window = time_series_windowize(X_train, seqLen=seqLen)
    Y_train_window = Y[seqLen:training_Set_length].values

    X_val = X[training_Set_length:, :]
    X_val_window = time_series_windowize(X_val, seqLen=seqLen)
    Y_val_window = Y[seqLen + training_Set_length:].values

    X_train = Variable(torch.Tensor(X_train_window).float())
    Y_train = Variable(torch.Tensor(Y_train_window).float())
    X_val = Variable(torch.Tensor(X_val_window).float())
    Y_val = Variable(torch.Tensor(Y_val_window).float())

    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = Data.DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle=True)
    return train_loader,X_val,Y_val


def rul_change(x):
    # 将开关寿命定性分析，改成离散的等级制
    if x > 0.8:
        return 1
    elif x > 0.7 and x <= 0.8:
        return 2
    elif x > 0.6 and x <= 0.7:
        return 3
    elif x > 0.5 and x <= 0.6:
        return 4
    elif x > 0.4 and x <= 0.5:
        return 5
    elif x > 0.3 and x <= 0.4:
        return 6
    elif x > 0.2 and x <= 0.3:
        return 7
    elif x <= 0.2:
        return 8

def label_dis(data):
    """标签离散化，赋予等级制标签"""
    data['rul'] = data['rul'].apply(lambda x: rul_change(x))
    return data

def make_classify_dataset(path):
    '''制作分类器所需的数据集'''
    train_set_list = []
    listdir(path, train_set_list)
    dataset = pd.DataFrame()
    for i in train_set_list:
        temp = pd.read_csv(i)
        temp = label_dis(temp)
        dataset = pd.concat([dataset,temp[temp['rul']>4]])
    dataset.to_csv('classify_dataset.csv',index=False)

if __name__ == '__main__':
    make_classify_dataset('./dataset')
