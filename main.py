from tkinter import *   #Tk, Label, Button, Entry, Canvas
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
def clear():
    print("正在清空...")
    canvas.delete('all')
    global N
    N = np.zeros((288, 288),dtype=np.uint8)

def recognite():
    print("正在识别...")
    global N
    im = Image.fromarray(N, mode='L').convert('L')
    im_new = im.resize((28, 28))
    im_new_np = np.array(im_new).astype(np.float32) / 255.0
    # 添加通道维度并确保形状为 [1, 1, 28, 28]
    im_new_np = im_new_np.reshape((1, 1, 28, 28))
    # 使用相同的 StandardScaler 对新图像进行标准化
    global scaler
    if 'scaler' not in globals():
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 784)))  # 假设输入是 28x28 的图像
        #StandardScaler.fit(X) 只需在训练集上运行一次，然后就可以在测试集上使用它来标准化数据。
        im_new_np_flat = im_new_np.reshape(-1, 784)
        im_new_np_flat = scaler.transform(im_new_np_flat)
        im_new_np = im_new_np_flat.reshape(1, 1, 28, 28)
    else:
        im_new_np_flat = im_new_np.reshape(-1, 784)
        im_new_np_flat = scaler.transform(im_new_np_flat)
        im_new_np = im_new_np_flat.reshape(1, 1, 28, 28)

    #将Numpy 数组转换为 PyTorch 张量
    im_new_tensor = torch.tensor(im_new_np, dtype=torch.float32)
    global model
    with torch.no_grad(): #禁用梯度计算
        output = model(im_new_tensor)
        _, predicted = torch.max(output, 1)
        im_predict = predicted.item()
    print(f'预测为：{im_predict}')
    var.set(f'预测为：{im_predict}')

def paint(event):
    x, y = event.x, event.y
    
    # 设置线条宽度
    line_width = 21  # 你可以根据需要调整这个值

    # 绘制当前点
    canvas.create_oval(x - line_width // 2, y - line_width // 2, x + line_width // 2, y + line_width // 2, fill='white', outline='white')

    # 更新 N 数组中的像素值
    for i in range(x - line_width // 2, x + line_width // 2 + 1):
        for j in range(y - line_width // 2, y + line_width // 2 + 1):
            if 0 <= i < 288 and 0 <= j < 288:
                N[j, i] = 255

def load_model():
    global model
    try:
        #初始化模型
        model = Net()
        #加载模型状态字典
        model.load_state_dict(torch.load('./model_Mnist.pth'))
        model.eval() 
        print("model loaded")
    except:
        print("model not found")


if __name__ == '__main__':
    
    N = np.zeros((288, 288),dtype=np.uint8)
    root = Tk()
    root.geometry('800x600+500+200')
    root.title('手写数字识别系统')
    lbl_title = Label(root, text='手写数字识别系统V1.0')
    lbl_title.grid(row=0, column=0, columnspan=3)
    var = StringVar()
    txt_result = Entry(root, textvariable=var, bg='green')
    txt_result.grid(row=1, column=0, columnspan=3)
    btn_clear = Button(root, text='清空', command=clear)
    btn_clear.grid(row=2, column=0)
    btn_load = Button(root, text='加载模型', command=load_model)
    btn_load.grid(row=2, column=1)
    btn_recognite = Button(root, text='识别', command=recognite)
    btn_recognite.grid(row=2, column=2)
    canvas = Canvas(root, width=288, height=288, bg='black')
    canvas.bind('<B1-Motion>', paint)
    canvas.grid(row=3, column=0, columnspan=3)
    root.mainloop()