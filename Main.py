import os
import sys
from tkinter import *
from ttkbootstrap import Style
from PIL import ImageTk, Image

style = Style(theme='journal')
root1 = style.master
root1.geometry('600x600')
root1.title('低压电气开关电寿命预测软件')


def user():
    os.system('python Main_user.py')


def developer():
    os.system('python Main_developer.py')


def help_doc():
    os.system(r'.\help\software_help.docx')


def version_introduction():
    winNew1 = Toplevel(root1)
    winNew1.geometry('400x400')
    winNew1.title('版本说明')
    m = Message(winNew1, text='当前版本2.0,与1.0相比新增两种寿命预测方法，具体请参考软件说明书，在下个版本会增加数据增强功能。', font=('华文新魏', 15))
    m.place(relx=0.2, rely=0.2, relwidth=0.5, relheight=0.5)


def contact():
    winNew2 = Toplevel(root1)
    winNew2.geometry('400x400')
    winNew2.title('联系作者')
    m = Message(winNew2, text='非常感谢您能对本软件提出宝贵建议，如有建议请联系作者：QQ756935454。', font=('华文新魏', 15))
    m.place(relx=0.2, rely=0.2, relwidth=0.5, relheight=0.5)


def exit1():
    sys.exit()


background = Image.open(r'picture\nuaa.jpg')
photo = ImageTk.PhotoImage(background)
canvas = Canvas(root1, width=800, height=800, bd=0, highlightthickness=0)
canvas.create_image(300, 300, image=photo)
canvas.pack()

photo_user = Image.open('picture/button.gif')
photo_user1 = ImageTk.PhotoImage(photo_user)
btn = Button(root1, text='用户入口',font=('华文新魏', 18),command=user, relief=GROOVE, image=photo_user1, compound="center")
btn.place(relx=0.4, rely=0.14, relwidth=0.2, relheight=0.09)
lb = Label(root1, text='', font=('华文新魏', 15))
lb.pack()
btn1 = Button(root1, text='开发人员入口', font=('华文新魏', 16), command=developer, relief=GROOVE, image=photo_user1, compound="center")
btn1.place(relx=0.4, rely=0.27, relwidth=0.2, relheight=0.09)

btn2 = Button(root1, text='帮助文档', font=('华文新魏', 18), command=help_doc, relief=GROOVE, image=photo_user1, compound="center")
btn2.place(relx=0.4, rely=0.40, relwidth=0.2, relheight=0.09)
btn3 = Button(root1, text='版本说明', font=('华文新魏', 18),command=version_introduction, relief=GROOVE, image=photo_user1, compound="center")
btn3.place(relx=0.4, rely=0.53, relwidth=0.2, relheight=0.09)

btn4 = Button(root1, text='联系作者', font=('华文新魏', 18) ,command=contact, relief=GROOVE, image=photo_user1, compound="center")
btn4.place(relx=0.4, rely=0.65, relwidth=0.2, relheight=0.09)
btn5 = Button(root1, text='退出软件', font=('华文新魏', 18),command=exit1, relief=GROOVE, image=photo_user1, compound="center")
btn5.place(relx=0.4, rely=0.78, relwidth=0.2, relheight=0.09)

root1.mainloop()
