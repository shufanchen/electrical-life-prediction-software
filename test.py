import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data_process import time_series_windowize
from train import myScore
import datetime
def data_preprocess(path, type):
    """将数据转化成模型所需的格式"""
    data = pd.read_csv(path)
    X, Y = data.iloc[:, [i for i in range(24)] + [25, 26, 27, 28]], data.iloc[:, [24]]
    selector = joblib.load('./model/contactor/' + type + 'feature_select')
    standar = joblib.load('./model/contactor/' + type + 'standardscaler')
    X = selector.transform(X)
    X = standar.transform(X)
    return X


def test(model_type, file_directory, device_type='contactor',seqlen=60):
    """用户调用的预测寿命接口"""
    if model_type == 'LSTM':
        test_data = data_preprocess(file_directory, 'LSTM_con_')
        test_data = time_series_windowize(test_data, seqlen)
        test_data = Variable(torch.Tensor(test_data).float())
        model = torch.load('./model/'+device_type+'/'+'LSTM_prediciton_model.pkl')
        pre_data = model(test_data)
    elif model_type == 'DAST':
        test_data = data_preprocess(file_directory, 'DAST_con_')
        test_data = time_series_windowize(test_data, seqlen)
        test_data = Variable(torch.Tensor(test_data).float())
        model = torch.load('./model/'+device_type+'DAST_prediciton_model.pkl')
        pre_data = model(test_data)
    elif model_type == 'RNN':
        test_data = data_preprocess(file_directory, 'dis_')
        model = torch.load('./model/'+device_type+'RNN_prediciton_model.pkl')
        pre_data = model(test_data)

    return pre_data

def print_test_log(model_name,test_set,test_score):
    """打印测试模型日志"""
    with open('model_testing_log.txt', 'a') as f:
        f.write('/' + '*' * 80 + '/\n')
        f.write('训练日期:\t'+str(datetime.datetime.now())+'\n')
        f.write('模型名称:\t'+model_name+'\n')
        f.write('训练集名称:\t' + test_set + '\n')
        f.write('val_score:\t%.02f\n'% test_score)
        f.write('/' + '*' * 80 + '/\n')
    return 0

def test_developer(file_directory,model_path,selector_path,standar_path,seqlen):#下个版本该函数要加入序列长度参数
    """开发者调用的测试模型接口"""
    test_data = pd.read_csv(file_directory)
    X, Y = test_data.iloc[:, [i for i in range(24)] + [25, 26, 27, 28]], test_data.iloc[:, [24]]
    model = torch.load(model_path)
    selector = joblib.load(selector_path)
    standar = joblib.load(standar_path)
    X = selector.transform(X)
    X = standar.transform(X)
    X = time_series_windowize(X,seqlen)
    X = Variable(torch.Tensor(X).float())
    prediction = model(X)
    max_rul = len(test_data)
    test_score = myScore(Y.iloc[seqlen:].values * max_rul, prediction.detach().numpy() * max_rul)
    plt.plot(range(len(Y)-seqlen), prediction.detach().numpy())
    plt.plot(range(len(Y)-seqlen), Y.iloc[seqlen:].values)
    plt.show()
    print_test_log(model_path,file_directory,test_score)
    return test_score

if __name__ == '__main__':
    seqlen = 60
    pre = test('LSTM', 'b3_database.csv')
    y = pd.read_csv('b3_database.csv')
    y = y.iloc[seqlen:, [24]]
    plt.plot(range(len(y)), y)
    plt.plot(range(len(y)), pre.detach().numpy())
    plt.show()
