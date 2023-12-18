from tkinter import *
import torch
from Data_refine import *
import tkinter.filedialog
from train import *
from test import *
from ttkbootstrap import Style
from PIL import ImageTk, Image
from tkinter import ttk
def newwind():
    winNew = Toplevel(root)
    winNew.geometry('900x800')
    winNew.title('生成数据集模块')
    msg1 = Label(winNew, text='提示:原始数据文件路径是存放正序的开关测试数据\n(xlsx格式)的目录，数据关键段始末位置是指燃弧段\n开始和结束的大致采样点数(默认为13000和13800)',
                 font=('华文新魏', 16), relief=SUNKEN)
    msg1.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.25)
    lb2 = Label(winNew, text='原始数据文件路径:', font=('华文新魏', 16))
    lb2.place(relx=0.1, rely=0.3)
    lb3 = Label(winNew, text='数据集存储路径:', font=('华文新魏', 16))
    lb3.place(relx=0.1, rely=0.5)
    inp1 = Entry(winNew, font=('华文新魏', 20))
    inp1.place(relx=0.1, rely=0.35, relwidth=0.8, relheight=0.1)
    inp2 = Entry(winNew, font=('华文新魏', 20))
    inp2.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)
    lb3 = Label(winNew, text='燃弧段大致起点:', font=('华文新魏', 16))
    lb3.place(relx=0.1, rely=0.65)
    lb4 = Label(winNew, text='燃弧段大致结束点:', font=('华文新魏', 16))
    lb4.place(relx=0.7, rely=0.65)
    inp3 = Entry(winNew, font=('华文新魏', 20))
    inp3.place(relx=0.1, rely=0.7, relwidth=0.2, relheight=0.1)
    inp4 = Entry(winNew, font=('华文新魏', 20))
    inp4.place(relx=0.7, rely=0.7, relwidth=0.2, relheight=0.1)
    var = StringVar()
    var.set('状态：尚未开始')
    lb5 = Label(winNew, textvariable=var, fg='red', font=('华文新魏', 18), relief=GROOVE)
    lb5.place(relx=0.35, rely=0.85)

    def start():
        database_name = begin(inp1.get(), inp2.get(), int(inp3.get()), int(inp4.get()))
        var.set('状态:已生成训练集:\n' + database_name)

    bt_start = Button(winNew, text='开始生成', command=start)
    bt_start.place(relx=0.1, rely=0.85, height=70, width=100)
    btClose = Button(winNew, text='关闭', command=winNew.destroy)
    btClose.place(relx=0.7, rely=0.85, height=70, width=100)


database_name_train = ''


def newwind1():
    winNew1 = Toplevel(root)
    winNew1.geometry('600x400')
    winNew1.title('训练模型模块')


    def subwin1():
        def xz():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb.config(text='您选择的训练集是' + filename)
            else:
                lb.config(text='您没有选择任何文件')
            global database_name_train
            database_name_train = filename
        subwin_1 = Toplevel(winNew1)
        subwin_1.geometry('900x640')
        subwin_1.title('训练LSTM模型')

        btn = Button(subwin_1, text='选择训练集', command=xz)
        btn.pack()
        lb = Label(subwin_1, text='', font=('华文新魏', 15))
        lb.pack()

        inp1 = Entry(subwin_1, font=('华文新魏', 20))
        inp1.place(relx=0.1, rely=0.18, relwidth=0.8, relheight=0.08)
        lb3 = Label(subwin_1, text='模型保存路径:', font=('华文新魏', 15))
        lb3.place(relx=0.1, rely=0.13)

        lb4 = Label(subwin_1, text='训练模型参数设定:', font=('华文新魏', 20), fg='red')
        lb4.place(relx=0.1, rely=0.28)
        # 参数设定N_HIDDEN1, N_LAYER1, N_EPOCH1, LR1, batch_size1, seqLen1, input_size1:
        ddl0 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl0['value'] = ('160', '120', '80', '60')
        ddl0.current(0)
        ddl0.place(relx=0.25, rely=0.35, relwidth=0.1, relheight=0.08)
        lb5 = Label(subwin_1, text='Hidden_size:', font=('华文新魏', 15))
        lb5.place(relx=0.08, rely=0.37)

        ddl1 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl1['value'] = ('1', '2', '3')
        ddl1.current(0)
        ddl1.place(relx=0.7, rely=0.35, relwidth=0.1, relheight=0.08)
        lb6 = Label(subwin_1, text='Num_layers:', font=('华文新魏', 15))
        lb6.place(relx=0.55, rely=0.37)

        ddl2 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl2['value'] = ('30', '50', '70')
        ddl2.current(0)
        ddl2.place(relx=0.25, rely=0.5, relwidth=0.1, relheight=0.08)
        lb7 = Label(subwin_1, text='Num_Epoch:', font=('华文新魏', 15))
        lb7.place(relx=0.1, rely=0.52)

        ddl3 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl3['value'] = ('0.001', '0.01', '0.1')
        ddl3.current(0)
        ddl3.place(relx=0.7, rely=0.5, relwidth=0.1, relheight=0.08)
        lb8 = Label(subwin_1, text='Learning_Rate:', font=('华文新魏', 15))
        lb8.place(relx=0.52, rely=0.52)

        ddl4 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl4['value'] = ('64', '128')
        ddl4.current(0)
        ddl4.place(relx=0.25, rely=0.65, relwidth=0.1, relheight=0.08)
        lb9 = Label(subwin_1, text='Batch_Size:', font=('华文新魏', 15))
        lb9.place(relx=0.11, rely=0.67)

        ddl5 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl5['value'] = ('30', '40','50','60','70')
        ddl5.current(0)
        ddl5.place(relx=0.7, rely=0.65, relwidth=0.1, relheight=0.08)
        lb10 = Label(subwin_1, text='Seq_Len:', font=('华文新魏', 15))
        lb10.place(relx=0.58, rely=0.67)

        ddl6 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl6['value'] = ('5', '10', '15', '20')
        ddl6.current(0)
        ddl6.place(relx=0.25, rely=0.8, relwidth=0.1, relheight=0.08)
        lb11 = Label(subwin_1, text='Input_Size:', font=('华文新魏', 15))
        lb11.place(relx=0.11, rely=0.82)

        ddl7 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl7['value'] = ('1000', '2000', '3000')
        ddl7.current(0)
        ddl7.place(relx=0.7, rely=0.8, relwidth=0.1, relheight=0.08)
        lb12 = Label(subwin_1, text='Max_rul:', font=('华文新魏', 15))
        lb12.place(relx=0.58, rely=0.82)

        def start_train():
            params = {}
            params['seqLen'] = int(ddl5.get())
            params['batch_size'] = int(ddl4.get())
            params['input_size'] = int(ddl6.get())
            params['learning_rate'] = eval(ddl3.get())
            params['epochs'] = int(ddl2.get())
            params['max_rul'] = int(ddl7.get())
            params['hidden_size'] = int(ddl0.get())
            params['num_layers'] = int(ddl1.get())
            params['model_type'] = 'LSTM'
            train_con(database_name_train,params,inp1.get())

        btn_start = Button(subwin_1, text='开始训练', command=start_train)
        btn_start.place(relx=0.45, rely=0.9, height=40, width=50)

    def subwin2():
        def xz():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb.config(text='您选择的训练集是' + filename)
            else:
                lb.config(text='您没有选择任何文件')
            global database_name_train
            database_name_train = filename
        subwin_1 = Toplevel(winNew1)
        subwin_1.geometry('900x640')
        subwin_1.title('训练DAST模型')

        btn = Button(subwin_1, text='选择训练集', command=xz)
        btn.pack()
        lb = Label(subwin_1, text='', font=('华文新魏', 15))
        lb.pack()

        inp1 = Entry(subwin_1, font=('华文新魏', 20))
        inp1.place(relx=0.1, rely=0.18, relwidth=0.8, relheight=0.08)
        lb3 = Label(subwin_1, text='模型保存路径:', font=('华文新魏', 15))
        lb3.place(relx=0.1, rely=0.13)

        lb4 = Label(subwin_1, text='训练模型参数设定:', font=('华文新魏', 20), fg='red')
        lb4.place(relx=0.1, rely=0.28)
        # 参数设定N_HIDDEN1, N_LAYER1, N_EPOCH1, LR1, batch_size1, seqLen1, input_size1:
        ddl0 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl0['value'] = ('1', '2', '3', '4')
        ddl0.current(0)
        ddl0.place(relx=0.25, rely=0.35, relwidth=0.1, relheight=0.08)
        lb5 = Label(subwin_1, text='n_encoder_layers:', font=('华文新魏', 15))
        lb5.place(relx=0.06, rely=0.37)

        ddl1 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl1['value'] = ('1', '2', '3','4')
        ddl1.current(3)
        ddl1.place(relx=0.7, rely=0.35, relwidth=0.1, relheight=0.08)
        lb6 = Label(subwin_1, text='n_heads', font=('华文新魏', 15))
        lb6.place(relx=0.55, rely=0.37)

        ddl2 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl2['value'] = ('30', '50', '70')
        ddl2.current(0)
        ddl2.place(relx=0.25, rely=0.5, relwidth=0.1, relheight=0.08)
        lb7 = Label(subwin_1, text='Num_Epoch:', font=('华文新魏', 15))
        lb7.place(relx=0.1, rely=0.52)

        ddl3 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl3['value'] = ('0.001', '0.01', '0.1')
        ddl3.current(0)
        ddl3.place(relx=0.7, rely=0.5, relwidth=0.1, relheight=0.08)
        lb8 = Label(subwin_1, text='Learning_Rate:', font=('华文新魏', 15))
        lb8.place(relx=0.52, rely=0.52)

        ddl4 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl4['value'] = ('64', '128')
        ddl4.current(0)
        ddl4.place(relx=0.25, rely=0.65, relwidth=0.1, relheight=0.08)
        lb9 = Label(subwin_1, text='Batch_Size:', font=('华文新魏', 15))
        lb9.place(relx=0.11, rely=0.67)

        ddl5 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl5['value'] = ('30', '40','50','60','70')
        ddl5.current(0)
        ddl5.place(relx=0.7, rely=0.65, relwidth=0.1, relheight=0.08)
        lb10 = Label(subwin_1, text='Seq_Len:', font=('华文新魏', 15))
        lb10.place(relx=0.58, rely=0.67)

        ddl6 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl6['value'] = ('5', '10', '15', '20')
        ddl6.current(0)
        ddl6.place(relx=0.25, rely=0.8, relwidth=0.1, relheight=0.08)
        lb11 = Label(subwin_1, text='Input_Size:', font=('华文新魏', 15))
        lb11.place(relx=0.11, rely=0.82)

        ddl7 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl7['value'] = ('1000', '2000', '3000')
        ddl7.current(0)
        ddl7.place(relx=0.7, rely=0.8, relwidth=0.1, relheight=0.08)
        lb12 = Label(subwin_1, text='Max_rul:', font=('华文新魏', 15))
        lb12.place(relx=0.58, rely=0.82)

        def start_train():
            params = {}
            params['seqLen'] = int(ddl5.get())
            params['batch_size'] = int(ddl4.get())
            params['input_size'] = int(ddl6.get())
            params['learning_rate'] = eval(ddl3.get())
            params['epochs'] = int(ddl2.get())
            params['max_rul'] = int(ddl7.get())
            params['n_encoder_layers'] = int(ddl0.get())
            params['n_heads'] = int(ddl1.get())
            params['model_type'] = 'DAST'
            train_con(database_name_train,params,inp1.get())
        btn_start = Button(subwin_1, text='开始训练', command=start_train)
        btn_start.place(relx=0.45, rely=0.9, height=40, width=50)

    def subwin3():
        def xz():
            filename = tkinter.filedialog.askopenfilename()
            if filename != '':
                lb.config(text='您选择的训练集是' + filename)
            else:
                lb.config(text='您没有选择任何文件')
            global database_name_train
            database_name_train = filename
        subwin_1 = Toplevel(winNew1)
        subwin_1.geometry('900x640')
        subwin_1.title('训练RNN模型')

        btn = Button(subwin_1, text='选择训练集', command=xz)
        btn.pack()
        lb = Label(subwin_1, text='', font=('华文新魏', 15))
        lb.pack()

        inp1 = Entry(subwin_1, font=('华文新魏', 20))
        inp1.place(relx=0.1, rely=0.18, relwidth=0.8, relheight=0.08)
        lb3 = Label(subwin_1, text='模型保存路径:', font=('华文新魏', 15))
        lb3.place(relx=0.1, rely=0.13)

        lb4 = Label(subwin_1, text='训练模型参数设定:', font=('华文新魏', 20), fg='red')
        lb4.place(relx=0.1, rely=0.28)
        # 参数设定N_HIDDEN1, N_LAYER1, N_EPOCH1, LR1, batch_size1, seqLen1, input_size1:
        ddl0 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl0['value'] = ('5', '10', '15', '20')
        ddl0.current(0)
        ddl0.place(relx=0.25, rely=0.35, relwidth=0.1, relheight=0.08)
        lb5 = Label(subwin_1, text='Feature_num:', font=('华文新魏', 15))
        lb5.place(relx=0.08, rely=0.37)

        ddl1 = ttk.Combobox(subwin_1, font=('华文新魏', 15))
        ddl1['value'] = ('5', '7', '9')
        ddl1.current(0)
        ddl1.place(relx=0.7, rely=0.35, relwidth=0.1, relheight=0.08)
        lb6 = Label(subwin_1, text='Neighbor_num:', font=('华文新魏', 15))
        lb6.place(relx=0.53, rely=0.37)


        def start_train():
            params = {}
            params['feature_num'] = int(ddl0.get())
            params['neighbor_num'] = int(ddl1.get())
            params['model_type'] = 'RNN'
            train_dis(database_name_train,params,inp1.get())

        btn_start = Button(subwin_1, text='开始训练', command=start_train)
        btn_start.place(relx=0.45, rely=0.9, height=40, width=50)


    btn_start0 = Button(winNew1, text='训练LSTM模型', command=subwin1)
    btn_start0.place(relx=0.1, rely=0.45, height=70, width=100)
    btn_start1 = Button(winNew1, text='训练DAST模型', command=subwin2)
    btn_start1.place(relx=0.4, rely=0.45, height=70, width=100)

    btn_start2 = Button(winNew1, text='训练RNN模型', command=subwin3)
    btn_start2.place(relx=0.7, rely=0.45, height=70, width=100)

model_name_eval = ''
test_set_name = ''


def newwind2():
    def xz():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb.config(text='您选择的模型是' + filename)
        else:
            lb.config(text='您没有选择任何文件')
        global model_name_eval
        model_name_eval = filename

    def xz1():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb1.config(text='您选择的测试集是' + filename)
        else:
            lb1.config(text='您没有选择任何文件')
        global test_set_name
        test_set_name = filename

    def xz2():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb2.config(text='您选择的特征选择器是' + filename)
        else:
            lb2.config(text='您没有选择任何文件')
        global feature_selector
        feature_selector = filename

    def xz3():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            lb3.config(text='您选择的归一化器是' + filename)
        else:
            lb3.config(text='您没有选择任何文件')
        global standar_name
        standar_name = filename

    def start_eval():
        s = test_developer(test_set_name,model_name_eval,feature_selector,standar_name,int(ddl1.get()))
        txt.insert(END, '*' * 40 + '\n')
        txt.insert(END, '测试模型:' + model_name_eval + '\n' + '测试集:' + test_set_name + '\n')
        txt.insert(END, '模型评分为(越小越好):%.06f\n' % s)
        txt.insert(END, '*' * 40 + '\n')

    winNew2 = Toplevel(root)
    winNew2.geometry('900x640')
    winNew2.title('模型测试模块')

    btn = Button(winNew2, text='选择预测模型', command=xz, relief=GROOVE)
    btn.pack()
    lb = Label(winNew2, text='', font=('华文新魏', 15))
    lb.pack()
    btn1 = Button(winNew2, text='选择测试集', command=xz1, relief=GROOVE)
    btn1.pack()
    lb1 = Label(winNew2, text='', font=('华文新魏', 15))
    lb1.pack()

    btn2 = Button(winNew2, text='选择特征选择器', command=xz2, relief=GROOVE)
    btn2.pack()
    lb2 = Label(winNew2, text='', font=('华文新魏', 15))
    lb2.pack()
    btn3 = Button(winNew2, text='选择归一化器', command=xz3, relief=GROOVE)
    btn3.pack()
    lb3 = Label(winNew2, text='', font=('华文新魏', 15))
    lb3.pack()
    ddl1 = ttk.Combobox(winNew2, font=('华文新魏', 15))
    ddl1['value'] = ('30', '50', '70','90')
    ddl1.current(0)
    ddl1.pack()
    lb5 = Label(winNew2, text='单词预测需要的序列长度:', font=('华文新魏', 15))
    lb5.place(relx=0.09, rely=0.33)


    btn2 = Button(winNew2, text='开始测试', command=start_eval, relief=GROOVE)
    btn2.place(relx=0.45, rely=0.41, height=70, width=100)
    txt = Text(winNew2)
    txt.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.7)




def newwind3():
    winNew3 = Toplevel(root)
    winNew3.geometry('900x640')
    winNew3.title('数据增强模块')
    m = Message(winNew3, text='针对故障诊断领域退化数据不足或缺失的现象，数据增强模块能有效弥补模型训练中数据量不足的痛点，'
                              '该功能模型预计将在本软件3.0版本上线，敬请期待！', font=('华文新魏', 15))
    m.place(relx=0.2, rely=0.2, relwidth=0.5, relheight=0.5)
    return



style = Style(theme='journal')
root = style.master
root.geometry('800x800')
root.title('低压电气开关电寿命预测软件——开发者界面')

image2 =Image.open(r'picture\nuaa2.gif')
background_image = ImageTk.PhotoImage(image2)

background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
photo_user = Image.open(r'picture\button.gif')
photo_user1 = ImageTk.PhotoImage(photo_user)

btn1 = Button(root, text='生成数据集', command=newwind, font=('华文新魏', 18), image=photo_user1, compound="center")
btn1.place(relx=0.15, rely=0.3, relwidth=0.3, relheight=0.1)

btn2 = Button(root, text='训练模型', command=newwind1, font=('华文新魏', 18), image=photo_user1, compound="center")
btn2.place(relx=0.55, rely=0.3, relwidth=0.3, relheight=0.1)

btn3 = Button(root, text='测试模型', command=newwind2, font=('华文新魏', 18), image=photo_user1, compound="center")
btn3.place(relx=0.15, rely=0.6, relwidth=0.3, relheight=0.1)

btn4 = Button(root, text='数据增强', command=newwind3, font=('华文新魏', 18), image=photo_user1, compound="center")
btn4.place(relx=0.55, rely=0.6, relwidth=0.3, relheight=0.1)

root.mainloop()
