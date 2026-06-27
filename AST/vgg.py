import ast

# Sample machine learning code
ml_code = """

import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import matplotlib.pyplot as plt#作图的库
import torch#一个深度学习的库Pytorch
import torch.nn as nn#neural network,神经网络
import torch.optim as optim#一个实现了各种优化算法的库
import torch.nn.functional as F#神经网络函数库
from PIL import Image, ImageOps, ImageFilter, ImageEnhance#PIL是图像处理库 ImageEnhance数据增强,ImageFilter滤镜
import os#与操作系统交互,处理文件和目录、管理进程、获取环境变量
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')


#设置随机种子
import random
torch.backends.cudnn.deterministic = True#将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_Files(path):
    Files=sorted(os.listdir(path))
    return Files

datapath="/home/nadeem/code/archive/tiny-imagenet-50/tiny-imagenet-200/train/"

labels=get_Files(datapath)
print(f"len(labels):{len(labels)}")

def num_to_str(num):
    num=str(num)[::-1]
    for i in range(6-len(num)):
        num+="0"
    return num[::-1]

datafull_path=[]
datay=[]
for label_index in range(len(labels)):
    labelFile=datapath+labels[label_index]+"/images/"
    imageFile=get_Files(labelFile)
    for index in range(len(imageFile)):
        datafull_path.append(labelFile+imageFile[index])
        datay.append(label_index)
    if label_index %10==0:
        print(f"label_index:{label_index}")
datafull_path=np.array(datafull_path)
datay=np.array(datay)
print(f"datafull_path.shape:{datafull_path.shape},datay.shape:{datay.shape}")


#划分训练集和测试集的函数
def train_test_split(dataX,datay,shuffle=True,percentage=0.8):

    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return train_X,train_y,test_X,test_y


train_full_path,train_y,valid_full_path,valid_y=train_test_split(datafull_path,datay,percentage=0.95)
label_count=np.max(train_y)+1
#for index in range(label_count):
    #print(f"np.sum(train_y==index)={np.sum(train_y==index)}")
print(f"train_full_path.shape:{train_full_path.shape},valid_full_path.shape:{valid_full_path.shape}")

valid_X=[]
for index in range(len(valid_full_path)):
    image=Image.open(valid_full_path[index])
    image=np.array(image)
    if len(image.shape)==2:
        h,w=image.shape
        new_image=np.random.rand(h,w,3)
        for i in range(3):
            new_image[:,:,i]=image
        image=new_image
    image=np.transpose(image,(2,0,1))
    valid_X.append(image)
valid_X=np.array(valid_X)
print(f"valid_X.shape:{valid_X.shape}")

cfgs={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}
def make_feature(cfg):
    feature_layers=[]#特征图
    front=3#上一层的参数
    for v in cfg:#遍历vgg的每层网络结构
        if v=="M":#如果是最大池化层
            feature_layers.append(nn.MaxPool2d(2,2))#池化核大小为2,步长为2
        else:
            feature_layers.append(nn.Conv2d(front,v,kernel_size=3,stride=1,padding=1))
            feature_layers.append(nn.BatchNorm2d(v))
            front=v
            feature_layers.append(nn.GELU())
    return nn.Sequential(*feature_layers)
class VGG(nn.Module):
    def __init__(self):
        #继承父类的所有方法
        super(VGG,self).__init__()
        
        self.conv=make_feature(cfgs['vgg16'])
        
        self.fc=nn.Sequential(
        nn.Linear(2048,256),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Tanh(),
        nn.Linear(256,256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256,200),
        )   
        
        
    def forward(self,input):
        #卷积神经网络
        output=self.conv(input)
        #展平
        output=output.view(-1,2048)#reshape
        #全连接层,原论文是4096
        output=self.fc(output)
        output=F.log_softmax(output/5,dim=1)
        return output
    


import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()

#device='cuda' if torch.cuda.is_available() else"cpu"
device="cpu"
print(f"device:{device}")

valid_X1=torch.Tensor(valid_X).to(device)
netC=VGG()

train_accs=[]#存储训练集的准确率
valid_accs=[]#存储测试集的准确率
#训练周期为50000次
num_epochs=5
batch_size=100#一次训练100张图片
#优化器
optimizer=optim.Adam(netC.parameters(),lr=0.00025,betas=(0.5,0.999))
#损失函数
criterion=nn.NLLLoss()

netC.to(device)

for epoch in range(num_epochs):
    
    random_num=[index for index in range(len(train_full_path))]
    np.random.shuffle(random_num)
    
    train_full_path=train_full_path[random_num]
    
    train_y=train_y[random_num]
    
    train_X1=[]
    for index in range(batch_size):
        image=Image.open(train_full_path[index])
        image=np.array(image)
        if len(image.shape)==2:
            h,w=image.shape
            new_image=np.random.rand(h,w,3)
            for i in range(3):
                new_image[:,:,i]=image
            image=new_image
        image=np.transpose(image,(2,0,1))
        train_X1.append(image)
    train_X1=np.array(train_X1)
    
    train_X1=torch.Tensor(train_X1).to(device)
    train_y1=torch.Tensor(train_y[:batch_size]).long().to(device)
    
    #训练
    netC.train()
    #将数据放进去训练
    output=netC(train_X1).to(device)
    #计算每次的损失函数
    loss=criterion(output,train_y1).to(device)
    #反向传播
    loss.backward()
    #优化器进行优化(梯度下降,降低误差)
    optimizer.step()
    #将梯度清空
    optimizer.zero_grad()
    if epoch%100==0:
        netC.eval()
        with torch.no_grad():
            output=netC(train_X1).to(device)
            output=output.detach().cpu().numpy()
            train_pred=np.argmax(output,axis=1)

            output=netC(valid_X1).to(device)
            output=output.detach().cpu().numpy()
            valid_pred=np.argmax(output,axis=1)
        
        train_acc=np.sum(train_pred==train_y[:batch_size])/len(train_y[:batch_size])
        valid_acc=np.sum(valid_pred==valid_y)/len(valid_y)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        print(f"epoch:{epoch},loss:{loss} train_acc:{train_acc},valid_acc:{valid_acc}")

pr.disable()
pr.print_stats(sort='cumtime')

s = io.StringIO()
sortby = 'cumulative' # You can change the sorting key (e.g., 'time', 'calls')
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

with open('/home/nadeem/Downloads/vgg16-50.txt', 'w') as f:
    f.write(s.getvalue())
"""

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = 0
        self.load_statements = 0
        self.store_statements = 0
        self.loops = 0
        self.control=0
        

    def visit_Call(self, node):
        self.function_calls += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.load_statements += 1
        elif isinstance(node.ctx, ast.Store):
            self.store_statements += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops += 1
        self.generic_visit(node)
   
    def visit_If(self, node):
        self.control += 1
        self.generic_visit(node)  
        
    def visit_Function(self, node):
        self.control += 1
        self.generic_visit(node) 
   

# Parse the code into an AST
tree = ast.parse(ml_code)

# Analyze the AST
analyzer = CodeAnalyzer()
analyzer.visit(tree)

function_count = 0
for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1

print(f"Number of control calls: {analyzer.control}")
print(f"Number of function calls: {analyzer.function_calls}")
print(f"Number of load statements: {analyzer.load_statements}")
print(f"Number of store statements: {analyzer.store_statements}")
print(f"Number of loops: {analyzer.loops}")
print(f"Number of function: {function_count}")
