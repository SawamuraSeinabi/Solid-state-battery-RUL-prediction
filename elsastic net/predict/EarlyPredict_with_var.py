import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

train_rawdata = np.array(pd.read_table(r'.\Dataset2\train.txt', sep=','))
test_rawdata = np.array(pd.read_table(r'.\Dataset2\test.txt', sep=','))


# 数据集构造
class Cell_var_data(Dataset):
    def __init__(self, flag):
        if flag == 'train':
            self.label = (torch.tensor(train_rawdata[:, 1]).view(-1, 1)).float()
            self.data = torch.tensor(train_rawdata[:, 0]).view(-1, 1).float()
        elif flag == 'test':
            self.label = (torch.tensor(test_rawdata[:, 1]).view(-1, 1)).float()
            self.data = torch.tensor(test_rawdata[:, 0]).view(-1, 1).float()

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y

    def __len__(self):
        return self.label.size()[0]

    def length(self):
        return self.__len__()


testset = Cell_var_data('test')
trainset = Cell_var_data('train')
testloader = DataLoader(dataset=testset, batch_size=testset.length(), shuffle=False)
trainloader = DataLoader(dataset=trainset, batch_size=trainset.length(), shuffle=True)
test_loss_iter = []
# 网络类
class LinearRegression(nn.Module):
    """
    Linear Regression Module, the input features and output
    features are defaults both 1
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# 第二层封装网络
class Net():
    def __init__(self):
        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.learning_rate = 0.001  # default
        self.epoches = 10000
        self.loss_function = nn.L1Loss()  # MAE
        self.lambadaL1 = 0.0001
        self.create_model()

    def create_model(self):
        self.model = LinearRegression()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8,
                                                                    patience=500, threshold=1e-4, min_lr=1e-7)

    def train(self, train_dataloader, test_dataloader, model_save_path="model.pth"):
        """
        Train the model and save the parameters
        Args:
            model_save_path: saved name of model
            data: (x, y) --> logx,logy
            and logy = k logx + b
        Returns:
            None
        """
        for epoch in range(self.epoches):
            for x, y in train_dataloader:
                logx = torch.log10(x)
                logy = torch.log10(y)
                prediction = self.model(logx)
                loss1 = self.loss_function(prediction, logy)
                # L1 损失
                all_linear_params = torch.cat([para.view(-1) for para in self.model.parameters()])
                l1_regularization = self.lambadaL1 * torch.norm(all_linear_params, 1)
                loss = loss1 + l1_regularization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            for X, Y in test_dataloader:
                logX = torch.log10(X)
                Y_hat = torch.pow(10, self.model(logX))
                test_loss = self.loss_function(Y_hat, Y)
                # if epoch == 9999:
                #     np_yhat = Y_hat.detach().numpy()
                #     print(np_yhat)
                test_loss_iter.append(test_loss.item())
            self.scheduler.step(test_loss)
            print("epoch: {}, loss on test_set is: {}".format(epoch, test_loss.item()))
        torch.save(self.model.state_dict(), model_save_path)

    def predict(self, x, model_path="model.pth"):
        self.model.load_state_dict(torch.load(model_path))
        prediction = torch.pow(10, self.model(torch.log(x)))
        return prediction[0, 0].item()


# 实例化网络
lassonet = Net()
lassonet.train(trainloader, testloader, model_save_path='./Netsave/lassonet1.pth')

print('end')
plt.figure()
plt.plot(range(1, 10001), test_loss_iter)
plt.xlim(0,1e4)
plt.ylim(100,900)
plt.xlabel('Epoch')
plt.ylabel('Loss')
# # plt.title('单次循环的百次十次循环放电容量与电压曲线',fontsize=15)
# plt.savefig(fname="linear_loss3.svg", dpi=600, format="svg")
plt.show()
