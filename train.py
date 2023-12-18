import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from data_process import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import datetime
from Network import *
import json
def myScore(Target, Pred):
    """针对寿命预测的模型评估函数"""
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp

def print_model_params(model_name,params,train_set,val_loss,val_score,acc):
    """打印训练模型日志"""
    with open(params['model_type']+'_model_training_log.txt', 'a') as f:
        f.write('/' + '*' * 80 + '/\n')
        f.write('训练日期:\t'+str(datetime.datetime.now())+'\n')
        f.write('模型名称:\t'+model_name+'\n')
        f.write('训练集名称:\t' + train_set + '\n')
        f.write('模型参数:\t')
        js = json.dumps(params)
        f.write(js)
        f.write('\n')
        if params['model_type']=='RNN':
            f.write('accuracy:\t%.06f\n' % acc)
        else:
            f.write('val_loss:\t%.06f\n'% val_loss )
            f.write('val_score:\t%.02f\n'% val_score)
        f.write('/' + '*' * 80 + '/\n')
    return 0


def train_con(file_directory, params, save_path = ''):
    """训练预测连续寿命的模型"""
    if params['model_type'] == 'LSTM':
        seqLen = params['seqLen']
        batch_size = params['batch_size']
        input_size = params['input_size']
        lr = params['learning_rate']
        epochs = params['epochs']
        max_rul = params['max_rul']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        model = Model(input_size, hidden_size, num_layers)
    if params['model_type'] == 'DAST':
        seqLen = params['seqLen']
        max_rul = params['max_rul']
        batch_size = params['batch_size']
        input_size = params['input_size']
        lr = params['learning_rate']
        epochs = params['epochs']
        n_heads = params['n_heads']
        n_encoder_layers = params['n_encoder_layers']
        dim_val = 64
        dim_attn = 64
        dim_val_t = 64
        dim_attn_t = 64
        dim_val_s = 64
        dim_attn_s = 64
        #n_heads = 4
        n_decoder_layers = 1
        #n_encoder_layers = 2
        #epochs = 30
        dec_seq_len = 4
        output_sequence_length = 1
        model = DAST(dim_val_s, dim_attn_s, dim_val_t, dim_attn_t, dim_val, dim_attn, seqLen, input_size,
                     dec_seq_len,
                     output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
    train_loader, X_val, Y_val = make_loader(file_directory, seqLen, batch_size,
                                             input_size,model_type=params['model_type'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training  and valing
    loss_list = []
    train_loss_list = []
    train_time = []
    temp_model = []
    model_loss = 100000

    for epoch in range(epochs):
        # training
        model.train()
        start1 = time.time()
        for i, (X, Y) in enumerate(train_loader):
            out = model(X)
            loss = torch.sqrt(criterion(out * max_rul, Y * max_rul))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        end1 = time.time()
        train_time.append(end1 - start1)
        loss_eopch = np.mean(np.array(loss_list))
        train_loss_list.append(loss_eopch)
        print('epoch = ', epoch,
              'train_loss = ', loss_eopch.item())

        # valing
        model.eval()
        prediction_t = model(X_val)
        prediction_np = prediction_t.detach().numpy()
        Y_val_numpy = Y_val.detach().numpy()
        val_loss = torch.sqrt(criterion(prediction_t * max_rul, Y_val * max_rul))
        val_score = myScore(Y_val_numpy * max_rul, prediction_np * max_rul)
        print('val_loss = ', val_loss.item(),
              'val_score = ', val_score)

        # Model save
        if epoch % 10 == 0:
            if val_loss.item() < model_loss:
                model_loss = val_loss.item()
                temp_model = model
        if epoch == epochs - 1:#最后一个epoch保存模型
            File_Path = '.' + '\\' + 'model/contactor'
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            if save_path != '':
                model_name = save_path + '/' + params['model_type'] + '_prediciton_model.pkl'
                torch.save(temp_model, model_name)
            else:
                model_name = File_Path + '/' + params['model_type'] + '_prediciton_model.pkl'
                torch.save(temp_model, model_name)
            plt.plot(range(len(Y_val)), prediction_t.detach().numpy())
            plt.plot(range(len(Y_val)), Y_val)
            plt.show()
            print_model_params(model_name,params,file_directory,val_loss,val_score,0)

    train_time_sum = np.sum(train_time)
    print('Train_time:', train_time_sum)


def train_dis(file_directory, params, save_path = ''):
    """训练预测寿命阶段的模型(本版本特指训练RNN模型)"""
    feature_num = params['feature_num']
    neighbor_num = params['neighbor_num']
    data_train = pd.read_csv('classify_dataset.csv')
    X_train, Y_train = data_train.iloc[:, [i for i in range(24)] + [25, 26, 27, 28]], data_train.iloc[:, [24]]
    data_val = pd.read_csv(file_directory)
    data_val = label_dis(data_val)
    data_val = data_val.iloc[int(0.5 * len(data_val)):, :]
    X_val, Y_val = data_val.iloc[:, [i for i in range(24)] + [25, 26, 27, 28]], data_val.iloc[:, [24]]
    # 数据归一化
    standardscaler = StandardScaler()
    standardscaler.fit(X_train)
    joblib.dump(standardscaler, './model/contactor/rnn_standardscaler')
    X_train = standardscaler.transform(X_train)
    X_val = standardscaler.transform(X_val)
    # 特征选择
    selector = SelectKBest(mutual_info_classif, k=feature_num)
    selector.fit(X_train, Y_train)
    joblib.dump(selector, './model/contactor/rnn_feature_select')
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    Y_train = Y_train.values

    kNN_reg = KNeighborsClassifier(n_neighbors=neighbor_num)
    # 只有经过fit之后才有best_estimator_
    kNN_reg.fit(X_train, Y_train)
    Y_pre = kNN_reg.predict(X_val)
    plt.plot(range(len(Y_val)), Y_pre)
    plt.plot(range(len(Y_val)), Y_val)
    plt.show()
    Y_val = Y_val.to_numpy()
    acc = (Y_pre == Y_val.squeeze(1)).sum()/len(Y_val)
    if save_path == '' :
        model_name = 'model' + '/contactor/' + 'rnn_model.pkl'
        torch.save(kNN_reg, model_name)
    else:
        model_name = save_path +'/' + 'rnn_model.pkl'
        torch.save(kNN_reg, model_name)
    print_model_params(model_name,params,'classify_dataset.csv',1,1,acc=acc)
    return 0


if __name__ == '__main__':
    params = {'seqLen': 40, 'batch_size': 64, 'input_size': 10, 'learning_rate': 0.001, 'epochs': 10, 'max_rul': 1000,
              'hidden_size': 160, 'num_layers': 2, 'model_type': 'LSTM'}
    train_con('./dataset/b2_database.csv', params=params)
    # params = {'feature_num':3,'neighbor_num':11,'model_type':'RNN'}
    # file_directory = './b3_database.csv'
    # train_dis(file_directory,params)
