import pandas as pd
import matplotlib.pyplot as plt
import os


class switch_Data:
    def __init__(self, filename, t_s=13000, t_e=13800):
        """数据对象初始化，每一个存储一次开和关过程数据的CSV文件被实例化成一个对象"""
        data = pd.read_excel(filename)
        self.ua = data.iloc[:, 6].values
        self.ia = data.iloc[:, 0].values
        self.ub = data.iloc[:, 7].values
        self.ib = data.iloc[:, 1].values
        self.uc = data.iloc[:, 8].values
        self.ic = data.iloc[:, 2].values
        self.period = 200  # 吸合时电压电流周期 单位:采样点数
        self.u_th = 80  # 电压突变阈值，如超过这个阈值，则怀疑该点是燃弧开始点
        self.i_th = 10  # 电流过0阈值，电流多个时刻绝对值小于该阈值，则怀疑已经开闸(燃弧开始)
        self.ua_end = 0
        self.ua_start = 0
        self.ub_end = 0
        self.ub_start = 0
        self.uc_end = 0
        self.uc_start = 0
        self.t_s = t_s
        self.t_e = t_e
        self.t_close = self.t_s - 1000

    def compute_contact(self, time_close, u, ii):
        """计算某一相的接触电阻、接触电压、接触电流"""
        u_power_sum = 0
        i_power_sum = 0
        for i in range(self.period):
            u_power_sum += u[time_close + i] * u[time_close + i]
            i_power_sum += ii[time_close + i] * ii[time_close + i]
        r_contact = (u_power_sum / i_power_sum) ** 0.5
        u_contact = (u_power_sum / self.period) ** 0.5
        i_contact = (i_power_sum / self.period) ** 0.5
        return [r_contact, u_contact, i_contact]

    def time_slice(self, t_s, t_e):
        """对开关数据对象的关键时间段进行切片"""
        self.ua = self.ua[t_s:t_e]
        self.ia = self.ia[t_s:t_e]
        self.ub = self.ub[t_s:t_e]
        self.ib = self.ib[t_s:t_e]
        self.uc = self.uc[t_s:t_e]
        self.ic = self.ic[t_s:t_e]

    def find_end(self, u, i):
        """寻找燃弧结束点"""
        u = u.tolist()
        i = i.tolist()
        t_end = 0  # 燃弧结束时间点
        index_temp = 1
        i_th = 10
        for u_t in u:
            if abs(u_t - u[index_temp + 1]) > self.u_th:
                if abs(i[index_temp]) < i_th and abs(i[index_temp + 1]) < i_th and abs(i[index_temp + 2]) < i_th and \
                        abs(i[index_temp + 3]) < i_th and abs(i[index_temp + 4]) < i_th:
                    if abs(i[index_temp - 10]) > 20:
                        t_end = index_temp
            index_temp = index_temp + 1
            if abs(len(u) - index_temp) < 6:
                break
        return t_end

    def find_start(self, t_end, u):
        """寻找燃弧起始点"""
        U_th1 = 10  # 电压过零阈值点，电压低于此值，可认为电压即将过零
        t_start = 0  # 燃弧开始时间点
        for index_temp1 in range(4, t_end):
            if abs(u[index_temp1 - 4]) < U_th1 and abs(u[index_temp1 - 3]) < U_th1 and abs(
                    u[index_temp1 - 2]) < U_th1 and \
                    abs(u[index_temp1 - 1]) < U_th1 and abs(u[index_temp1]) < U_th1:
                t_start = index_temp1 + 1
        return t_start

    def is_reburn(self):
        """判断是否燃弧重燃"""
        t_start = [self.ua_start, self.ub_start, self.uc_start]
        t_end = [self.ua_end, self.ub_end, self.uc_end]
        u = [self.ua, self.ub, self.uc]
        flag_reburn = [0, 0, 0]  # 燃弧重燃标志，0代表燃弧阶段电弧没有重燃，1代表有
        for i in range(3):
            index_temp3 = t_start[i] - 2
            change_index = t_start[i]
            for u_temp in u[i][t_start[i]:t_end[i] + 1]:
                if abs(u_temp - u[i][index_temp3]) > self.u_th:
                    if abs(index_temp3 - change_index) > 5:
                        change_index = index_temp3
                        flag_reburn[i] += 1
                index_temp3 += 1
            if flag_reburn[i] > 1:
                flag_reburn[i] = 1
            else:
                flag_reburn[i] = 0
        return flag_reburn

    def compute_arc_time_energy_power(self):
        """计算燃弧时间、燃弧能量、燃弧功率"""
        delta_t = 0.5 / 16400  # 采样间隔，单位：秒
        time_arc_burn = []
        dots_burn = [self.ua_end - self.ua_start, self.ub_end - self.ub_start, self.uc_end - self.uc_start]
        a = (dots_burn[0]) * delta_t * 1000
        time_arc_burn.append(a)
        b = (dots_burn[1]) * delta_t * 1000
        time_arc_burn.append(b)
        c = (dots_burn[2]) * delta_t * 1000
        time_arc_burn.append(c)
        energy_burn = [0, 0, 0]
        uu = [self.ua, self.ub, self.uc]
        ii = [self.ia, self.ib, self.ic]
        t_start = [self.ua_start, self.ub_start, self.uc_start]
        choose = 0
        for j in dots_burn:
            for i in range(j):
                energy_burn[choose] += abs(uu[choose][t_start[choose] + i] * ii[choose][t_start[choose] + i] * delta_t)
            choose += 1
        arc_power = [0, 0, 0]
        for index in range(3):
            if dots_burn[index] == 0:
                continue
            arc_power[index] = energy_burn[index] / (dots_burn[index] * delta_t)
        return time_arc_burn, energy_burn, arc_power

    def creat_feature(self):
        """创建领域特征"""
        a_contact = self.compute_contact(self.t_close, self.ua, self.ia)
        b_contact = self.compute_contact(self.t_close, self.ub, self.ib)
        c_contact = self.compute_contact(self.t_close, self.uc, self.ic)
        self.time_slice(self.t_s, self.t_e)
        self.ua_end = self.find_end(self.ua, self.ia)
        self.ua_start = self.find_start(self.ua_end, self.ua)
        self.ub_end = self.find_end(self.ub, self.ib)
        self.ub_start = self.find_start(self.ub_end, self.ub)
        self.uc_end = self.find_end(self.uc, self.ic)
        self.uc_start = self.find_start(self.uc_end, self.uc)
        time_arc_burn, energy_burn, arc_power = self.compute_arc_time_energy_power()
        print(time_arc_burn)
        flag_reburn = self.is_reburn()
        feature = a_contact + b_contact + c_contact + time_arc_burn + energy_burn + arc_power + flag_reburn
        return feature


# test1 = switch_Data('D:/低压电气开关数据/第六批实验数据/A/a6/2021年03月02日10时05分24.88秒 第18次.xlsx')
# feature1 = test1.creat_feature()


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def database_make(file_directory, t_s=13000, t_e=13800):
    """生成数据集，一个原始数据对象生成一行数据"""
    list_name = []
    listdir(file_directory, list_name)
    energy_burn_sum = [0, 0, 0]
    i = 1
    data_input = pd.DataFrame(columns=['a_r_contact', 'a_u_contact', 'a_i_contact',
                                       'b_r_contact', 'b_u_contact', 'b_i_contact',
                                       'c_r_contact', 'c_u_contact', 'c_i_contact',
                                       'a_time_arc', 'b_time_arc', 'c_time_arc',
                                       'a_energy', 'b_energy', 'c_energy',
                                       'a_arc_power', 'b_arc_power', 'c_arc_power',
                                       'a_is_reburn', 'b_is_reburn', 'c_is_reburn',
                                       'a_energy_sum', 'b_energy_sum', 'c_energy_sum',
                                       'rul'])
    for filename in list_name:
        temp_switch = switch_Data(filename, t_s, t_e)
        data1 = temp_switch.creat_feature()
        if i > 21:
            data1 = data_clean(data1, data_input)
        for index in range(3):
            energy_burn_sum[index] += data1[12 + index]
            data1.append(energy_burn_sum[index])  # 添加燃弧累计能量特征
        data1.append((len(list_name) - i) / (len(list_name)))  # 添加归一化剩余使用寿命标签
        pd_temp = pd.DataFrame(data=[data1], columns=['a_r_contact', 'a_u_contact', 'a_i_contact',
                                                      'b_r_contact', 'b_u_contact', 'b_i_contact',
                                                      'c_r_contact', 'c_u_contact', 'c_i_contact',
                                                      'a_time_arc', 'b_time_arc', 'c_time_arc',
                                                      'a_energy', 'b_energy', 'c_energy',
                                                      'a_arc_power', 'b_arc_power', 'c_arc_power',
                                                      'a_is_reburn', 'b_is_reburn', 'c_is_reburn',
                                                      'a_energy_sum', 'b_energy_sum', 'c_energy_sum',
                                                      'rul'])
        data_input = pd.concat([data_input, pd_temp], ignore_index=True)
        i += 1
        # if i == 5 :
        #     break
    return data_input


def add_complex_feature(data):
    """添加潜在特征，潜在特征需要根据已生成的领域特征计算得出"""
    data_length = len(data)
    energy_mean = [data.iloc[-1, 21] / data_length, data.iloc[-1, 22] / data_length, data.iloc[-1, 23] / data_length]
    window_size = 50
    high_energy_rate = pd.DataFrame(columns=['a_high_energy_rate', 'b_high_energy_rate', 'c_high_energy_rate'])
    for j in range(data_length):
        if j < window_size:
            high_num = [0, 0, 0]
            for index in range(window_size):
                for choose in range(3):
                    if data.iloc[index, 12 + choose] > 1.4 * energy_mean[choose]:  # 单次燃弧能量大于均值的1.4倍被认为是高能燃弧
                        high_num[choose] += 1
            high_rate = [high_num[0] / window_size, high_num[1] / window_size, high_num[2] / window_size]
            pd_temp1 = pd.DataFrame(data=[high_rate],
                                    columns=['a_high_energy_rate', 'b_high_energy_rate', 'c_high_energy_rate'])
        if j >= window_size:
            high_num = [0, 0, 0]
            for index in range(window_size):
                for choose in range(3):
                    if data.iloc[j - window_size + index + 1, 12 + choose] > 1.3 * energy_mean[choose]:
                        # 单次燃弧能量大于均值的1.3倍被认为是高能燃弧
                        high_num[choose] += 1
            high_rate = [high_num[0] / window_size, high_num[1] / window_size, high_num[2] / window_size]
            pd_temp1 = pd.DataFrame(data=[high_rate],
                                    columns=['a_high_energy_rate', 'b_high_energy_rate', 'c_high_energy_rate'])
        high_energy_rate = pd.concat([high_energy_rate, pd_temp1], ignore_index=True)
    data_out = pd.concat([data, high_energy_rate], axis=1, join='inner')
    # 添加重燃燃弧占比复合特征
    arc_reburn_rate = pd.DataFrame(columns=['arc_reburn_rate'])
    for j in range(data_length):
        if j < window_size:
            reburn_num = 0
            for index in range(window_size):
                reburn_num += sum(data.iloc[index, [18, 19, 20]])
            reburn_rate = reburn_num / (window_size * 3)
            pd_temp = pd.DataFrame(data=[reburn_rate], columns=['arc_reburn_rate'])
        if j >= window_size:
            reburn_num = 0
            for index in range(window_size):
                reburn_num += sum(data.iloc[j - window_size + index + 1, [18, 19, 20]])
            reburn_rate = reburn_num / (window_size * 3)
            pd_temp = pd.DataFrame(data=[reburn_rate], columns=['arc_reburn_rate'])
        arc_reburn_rate = pd.concat([arc_reburn_rate, pd_temp], ignore_index=True)
        print(arc_reburn_rate.iloc[j, :].values)
    data_out = pd.concat([data_out, arc_reburn_rate], axis=1, join='inner')
    return data_out


def data_clean(data_c, data_input_c):
    """数据清洗，剔除燃弧时间明显低于平均值的样本，并用之前20个正常值的平均值代替"""
    for i in range(3):
        time_mean = sum(data_input_c.iloc[-20:, 9 + i].values) / 20
        energy_mean = sum(data_input_c.iloc[-20:, 12 + i].values) / 20
        power_mean = sum(data_input_c.iloc[-20:, 15 + i].values) / 20
        if data_c[i + 9] < 0.5:
            data_c[i + 9] = time_mean
            data_c[i + 12] = energy_mean
            data_c[i + 15] = power_mean
    return data_c


def creat_file_name(file_info):
    """返回要生成数据集的文件名"""
    file_name = []
    flag = 0
    for i in file_info[::-1]:
        if (i == '/' or i=='\\') and flag == 0:
            flag += 1
            continue
        elif (i == '/' or i=='\\') and flag >= 1:
            break
        else:
            flag += 1
            file_name.append(i)
    file_name.reverse()
    output_file_name = ''
    for i in file_name:
        output_file_name += i
    output_file_name1 = output_file_name + '_database.csv'
    return output_file_name1, output_file_name


def begin(file_info, path, t_s, t_e):
    data_0 = database_make(file_info, t_s, t_e)
    output_file_name, _ = creat_file_name(file_info)
    path = list(path)
    for i in range(len(path)):
        if path[i] == '\\':
            path[i] = '/'
    temp = ''
    for j in path:
        temp += j
    path = temp
    if path[-1] != '/':
        path = path + '/'
    path = path + output_file_name
    data_1 = add_complex_feature(data_0)

    data_1.to_csv(path, index=False)
    return output_file_name


if __name__ == "__main__":
    a,_1 = creat_file_name('D:\低压电气开关数据\第六批实验数据\C\c4')
    print()
    # print("提示:数据文件目录是存放正序的开关测试数据(xlsx格式)的目录，数据关键段始末位置是指燃弧段开始和结束的大致采样点数(默认为13000和13800)")
    # file_info = 'D:/低压电气开关数据/第六批实验数据/A/a8'
    # # file_info = input("请输入数据文件所在目录:")
    # t_s = eval(input("请输入数据关键段开始位置:"))
    # t_e = eval(input("请输入数据关键段开始位置:"))
    # data_0 = database_make(file_info, t_s, t_e)
    # output_file_name, _ = creat_file_name(file_info)
    # data_1 = add_complex_feature(data_0)
    # data_1.to_csv(output_file_name, index=False)
