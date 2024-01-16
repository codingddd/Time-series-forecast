import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Config():
    train_path = "./train_set.csv"
    val_path = "./validation_set.csv"
    test_path = "./test_set.csv"
    data_path = './ETTh1.csv'
    timestep = 96  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    input_size = 7  # 每个步长对应的特征数量
    output_size = 96  # 预测步长
    moving_avg= 25 # kernel_size
    seed = 0 #随机种子
    epochs = 20 # 迭代轮数
    best_loss = 1 # 记录损失
    learning_rate = 0.001 # 学习率
    model_name = 'Improved model' # 模型名称
    save_path = './{}_best.pth'.format(model_name) # 最优模型保存路径


class moving_avg(nn.Module):#[32, 96, 7]

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, multi_mlp=False,hidden_size=512):
        
        super(Model, self).__init__()

        self.seq_len = Config.timestep #输入时间序列长度
       
        self.pred_len = Config.output_size #输出序列长度
        
        self.decompsition = series_decomp(Config.moving_avg)
        self.multi_mlp = multi_mlp
        self.hidden_size = hidden_size
        self.channels = Config.input_size
        if  self.multi_mlp:     
            self.Linear_Seasonal=nn.Sequential(
                    nn.Linear(self.seq_len, self.hidden_size),
                    nn.ReLU(),  
                    nn.Linear(self.hidden_size, self.hidden_size) ,
                    nn.ReLU(),  
                    nn.Linear(self.hidden_size, self.pred_len)
                )
            self.Linear_Trend=nn.Sequential(
                    nn.Linear(self.seq_len, self.hidden_size),
                    nn.ReLU(),  
                    nn.Linear(self.hidden_size, self.hidden_size) ,
                    nn.ReLU(),   
                    nn.Linear(self.hidden_size, self.pred_len)
                )
   
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))


    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)


    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


def create_dataset(data,n_predictions,n_next):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next),:]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j,k])
        train_Y.append(b)
    train_X = np.array(train_X,dtype='float32')
    train_Y = np.array(train_Y,dtype='float32')
    
    train_Y=train_Y.reshape(train_Y.shape[0],-1,7)
    
    return train_X,train_Y

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)  
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    
    seed = Config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 设置NumPy随机种子
    np.random.seed(seed)
    
    predictions_list=[]
    labels_list=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Config()
   
    df_train = pd.read_csv(config.train_path, index_col = 0)
    df_val = pd.read_csv(config.val_path, index_col = 0)
    df_test = pd.read_csv(config.test_path, index_col = 0)
    
    
    scaler = StandardScaler()
    all_data=np.concatenate((np.array(df_train),np.array(df_val),np.array(df_test)),axis=0)
    sss=scaler.fit(all_data)

    train_data = scaler.transform(np.array(df_train))
    val_data = scaler.transform(np.array(df_val))
    test_data = scaler.transform(np.array(df_test))
    
    train_X, train_Y =create_dataset(train_data,config.timestep,config.output_size)
    val_X, val_Y =create_dataset(val_data,config.timestep,config.output_size)
    test_X, test_Y =create_dataset(test_data,config.timestep,config.output_size)
   
    train_dataset = MyDataset(train_X, train_Y)
    val_dataset= MyDataset(val_X, val_Y)
    test_dataset= MyDataset(test_X, test_Y)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,shuffle=False,drop_last=True)
    
    
    model = Model().to(device)
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练模型
    loss_lst=[]
    for i in range(config.epochs):
        model.train()
        for seq, labels in train_loader:
            #seq:[32, 96, 7],labels:[32, 96,7]
            optimizer.zero_grad()

            y_pred = model(seq)
            
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        print(f'Training loss after epoch {i}: {single_loss.item():10.8f}')
        loss_lst.append(single_loss.item())

        # 在每个epoch后，使用验证集评估模型
        model.eval()
        with torch.no_grad():
            for seq, labels in val_loader:
                y_pred = model(seq)
                val_loss = loss_function(y_pred, labels)
            print(f'Validation loss after epoch {i}: {val_loss.item():10.8f}')
            
        
        if i%3==1:
            # 测试模型并画图
            model.eval()
            with torch.no_grad():
                for seq, labels in test_loader:
                    y_pred = model(seq)
                    test_loss = loss_function(y_pred, labels)
                print(f'Test loss: {test_loss.item():10.8f}')
            
            if test_loss<config.best_loss:
                config.best_loss=test_loss
                torch.save(model.state_dict(),config.save_path)

                p= seq[2,:,-1].cpu()
                y_pred_oil=y_pred[2,:,-1].cpu()
                y_labels_oil=torch.cat([p,labels[2,:,-1].cpu()],dim=0)
            
                
                x1 = np.arange(config.timestep, config.timestep+config.output_size) 
                x2 = np.arange(0, config.timestep+config.output_size)
                
                plt.figure(figsize=(12, 8))
                plt.plot(x1,y_pred_oil, "b",label="pred",linewidth=2.5)
                plt.plot(x2,y_labels_oil, "r",label="ground truth",linewidth=2.5)
                plt.legend()
                
                plt.savefig('image/savefig_example_{}.png'.format("Improved model"))
                plt.show()
                print("save success!")
    
    x1 = np.arange(0, config.epochs) 
    
    plt.figure(figsize=(12, 8))
    plt.plot(x1,np.array(loss_lst), "b",label="train_loss",linewidth=2)
    plt.legend()
    
    plt.savefig('image/savefig_example_{}.png'.format("Improved model_loss"))
    plt.show()    
    
    print(f'Final training loss: {single_loss.item():10.10f}')
    
    print(f'Final test loss: {config.best_loss.item():10.10f}')
    
