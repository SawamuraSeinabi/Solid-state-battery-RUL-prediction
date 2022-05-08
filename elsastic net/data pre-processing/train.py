import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.svm import SVR, LinearSVR
import math

save_dir = r'./Dataset/'


# 库加载
# 数据类的创建
class Data_Set(Dataset):
    def __init__(self, set_dir):
        with open(save_dir + set_dir, 'rb') as a:
            self.data = pickle.load(a)
        self.length = len(self.data.keys())

    def __getitem__(self, idx):
        x = torch.tensor([self.data[idx]['var']])
        y = torch.tensor(self.data[idx]['cycle_life'][0])
        x = torch.log(x)
        y = torch.log(y)
        x = x.float()
        y = y.float()
        return x, y

    def __len__(self):
        return self.length


train_set = Data_Set('train_data_var.pkl')
train_loader = DataLoader(train_set, batch_size=76, shuffle=False)

test_set = Data_Set('test_data_var.pkl')
test_loader = DataLoader(test_set, batch_size=48, shuffle=False)


# 网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.lin1(x)
        return x


net = Net()  # 网络实例化

num_epochs = 500  # 训练代数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2,threshold=0.01,min_lr=1e-7)  # 学习率自动调整
loss = torch.nn.MSELoss()  # 损失
train_loss_iter = []
test_loss_iter = []
for num_epoch in range(1, num_epochs + 1):
    for X, Y in train_loader:

        # Y = torch.exp(Y)
        # Y_hat = torch.exp(net(X))
        Y_hat=net(X)
        l = loss(Y_hat, Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print('epoch: %d, train_loss: %f' % (num_epoch, l))
    train_loss_iter.append(l.item())

    test_loss = 0
    with torch.no_grad():
        for X2, Y2 in test_loader:
            Y2_hat = net(X2)
            # Y2 = torch.exp(Y2)
            l2 = loss(Y2_hat, Y2)
            test_loss += l2
        MSE_0 = test_loss
        print('test_loss %.7g' % MSE_0)
    scheduler.step(MSE_0)
    test_loss_iter.append(MSE_0.item())
# 训练过程绘制
plt.figure()
plt.plot(range(1, 501), train_loss_iter)
plt.plot(range(1, 501), test_loss_iter)
plt.show()
# 预测效果检验
loss2=nn.L1Loss()
for X2, Y2 in test_loader:
    Y2_hat = net(X2)
    Y2_hat = torch.exp(Y2_hat)
    Y2 = torch.exp(Y2)
    l2=loss2(Y2_hat,Y2)

print('end')
