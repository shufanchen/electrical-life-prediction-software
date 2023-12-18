import tkinter.filedialog
from tkinter import *
from tkinter import ttk
import pandas as pd
from train import Model
from test import *
from ttkbootstrap import Style
from Data_refine import *
from PIL import ImageTk, Image
style = Style(theme='journal')
root = style.master
root.geometry('800x800')
root.title('低压电气开关电寿命预测软件——用户界面')


def resize(w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片
    w, h = pil_image.size  # 获取图像的原始大小
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.LANCZOS)


image2 = Image.open(r'图片\xiaohui~1.gif')
image2_resize = resize(120, 120, image2)
background_image = ImageTk.PhotoImage(image2_resize)

canvas = Canvas(root, width=500, height=500, bd=0, highlightthickness=0)
canvas.create_image(70, 70, image=background_image)
canvas.place(relx=0.8, rely=0, relwidth=0.2, relheight=0.2)

lb = Label(root, text='欢迎使用低压开关电寿命预测软件', font=('华文新魏', 25))
lb.place(relx=0.05, rely=0.05)
lb1 = Label(root, text='低压开关类型：', font=('华文新魏', 18))
lb1.place(relx=0.05, rely=0.15)
ddl = ttk.Combobox(root, font=('华文新魏', 15))
ddl['value'] = ('contactor', '继电器', '断路器', '隔离开关')
ddl.current(0)
ddl.place(relx=0.25, rely=0.15, relwidth=0.15)

lb2 = Label(root, text='选择预测寿命模型：', font=('华文新魏', 18))
lb2.place(relx=0.45, rely=0.15)
ddl1 = ttk.Combobox(root, font=('华文新魏', 15))
ddl1['value'] = ('LSTM', 'DAST', 'RNN')
ddl1.current(0)
ddl1.place(relx=0.71, rely=0.15, relwidth=0.15)

lb3 = Label(root, text='最少可提供的开关实验数据个数(xlsx格式)：', font=('华文新魏', 18))
lb3.place(relx=0.05, rely=0.25)
ddl2 = ttk.Combobox(root, font=('华文新魏', 15))
ddl2['value'] = ('100', '90', '80', '70', '60', '50', '40', '30')
ddl2.current(5)
ddl2.place(relx=0.62, rely=0.25, relwidth=0.15)
#
lb4 = Label(root, text='开关数据存储路径:', font=('华文新魏', 18))
lb4.place(relx=0.05, rely=0.32)
inp1 = Entry(root, font=('华文新魏', 20))
inp1.place(relx=0.31, rely=0.32, relwidth=0.6)
#
lb4 = Label(root, text='燃弧段大致起点:', font=('华文新魏', 18))
lb4.place(relx=0.05, rely=0.41)
lb5 = Label(root, text='燃弧段大致结束点:', font=('华文新魏', 18))
lb5.place(relx=0.5, rely=0.41)
inp2 = Entry(root, font=('华文新魏', 20))
inp2.place(relx=0.3, rely=0.4, relwidth=0.12, relheight=0.06)
inp3 = Entry(root, font=('华文新魏', 20))
inp3.place(relx=0.77, rely=0.4, relwidth=0.12, relheight=0.06)
txt = Text(root)
txt.place(relx=0.05, rely=0.58, relwidth=0.9)




def creat_temp_dataset():
    """为用户创建数据集，并返回数据集的地址"""
    temp_database_name = begin(inp1.get(), 'dataset/', int(inp2.get()), int(inp3.get()))
    return 'dataset/' + temp_database_name


def start_user():
    set_name_predict = creat_temp_dataset()
    #set_name_predict  = 'dataset/b3_database.csv'
    prediction = test(ddl1.get(),set_name_predict,ddl.get(),int(ddl2.get()))
    prediction = prediction.detach().numpy()[-1]
    advice = {'good': '该开关状态良好，无需维修和保养', 'normal': '该开关运行正常，无需维修但请注意保养',
              'danger': '该开关有损坏迹象，请进行维修', 'broken': '该开关已经濒临失效，请勿使用并更换开关'}
    flag = ''
    if 1 >= prediction >= 0.7:
        flag = 'good'
    elif 0.7 > prediction >= 0.5:
        flag = 'normal'
    elif 0.5 > prediction >= 0.3:
        flag = 'danger'
    else:
        flag = 'broken'
    txt.insert(END, '*' * 80 + '\n')
    txt.insert(END, '预测模型:' + ddl1.get() + '\n' + '数据文件:' + set_name_predict + '\n')
    txt.insert(END, '该设备剩余使用寿命为:%.02f\n' % prediction)
    txt.insert(END, '开关使用建议:' + advice[flag] + '\n')
    txt.insert(END, '*' * 80 + '\n')
    pass


bt_start = Button(root, text='开始预测', command=start_user)
bt_start.place(relx=0.42, rely=0.5, relwidth=0.12, relheight=0.06)

root.mainloop()
