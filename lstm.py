import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import math

class Config():
    data_path = './ETTh1.csv'
    train_path = "./train_set.csv"
    val_path = "./validation_set.csv"
    test_path = "./test_set.csv"
    
    timestep = 96  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 7  # 每个步长对应的特征数量
    hidden_size = 512  # 隐层大小
    output_size = 96  # 输出的时间步长
    num_layers = 2  # 层数
    epochs = 50 # 迭代轮数
    seed = 0 #随机种子
    best_loss = 1 # 记录损失
    learning_rate = 0.001 # 学习率
    model_name = 'lstm' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径


def create_dataset(data,n_predictions,n_next):
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
        self.data = torch.tensor(data, dtype=torch.float32).to(device)  # 转换为torch张量并移动到设备上
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=512, output_size=1):#input_size就是feature_size
        super().__init__()
        self.hidden_size = hidden_layer_size
        
        self.num_directions=1#单向LSTM
        
        self.num_layers=2
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size,self.num_layers,batch_first=True)
        
        self.output_size=output_size
        
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)
        self.fc5 = nn.Linear(self.hidden_size, self.output_size)
        self.fc6 = nn.Linear(self.hidden_size, self.output_size)
        self.fc7 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):#batch_size:32,timestep:96,feature:7
        #input_seq:[32, 96, 7],lstm_out:[32, 96, 100]
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size,self.hidden_size).to(device)
            
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        #lstm_out:[32, 96, 100]
        lstm_out = F.tanh(lstm_out)
        pred1 = self.fc1(lstm_out)[:,-1,:]
        pred2 = self.fc2(lstm_out)[:,-1,:]
        pred3 = self.fc3(lstm_out)[:,-1,:]
        pred4 = self.fc4(lstm_out)[:,-1,:]
        pred5 = self.fc5(lstm_out)[:,-1,:]
        pred6 = self.fc6(lstm_out)[:,-1,:]
        pred7 = self.fc7(lstm_out)[:,-1,:]
        
        pred = torch.stack([pred1, pred2, pred3, pred4, pred5, pred6, pred7], dim=-1)
        #dim=-1:pred:[32, 96, 7]
    
        return pred


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
    
    
    model = LSTM(input_size=7,output_size=config.output_size).to(device)
    loss_function = nn.MSELoss()
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
            
        
        if i%3==0:
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
                
                plt.savefig('image/savefig_example_{}.png'.format("lstm"))
                plt.show()
                print("save success!")
    
    x1 = np.arange(0, config.epochs) 
    
    plt.figure(figsize=(12, 8))
    plt.plot(x1,np.array(loss_lst), "b",label="train_loss",linewidth=2)
    plt.legend()
    
    plt.savefig('image/savefig_example_{}.png'.format("lstm_loss"))
    plt.show()    
    
    print(f'Final training loss: {single_loss.item():10.10f}')
    
    print(f'Final test loss: {config.best_loss.item():10.10f}')



