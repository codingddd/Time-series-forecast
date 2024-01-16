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
    d_model = 512
    timestep = 96  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    
    input_size = 7  # 每个步长对应的特征数量
    
    output_size = 336  # 预测步长

    epochs = 50 # 迭代轮数
    best_loss = 1 # 记录损失
    learning_rate = 0.0007 # 学习率
    seed = 0 # 随机种子
    model_name = 'transformer' # 模型名称
    save_path = './{}_best.pth'.format(model_name) # 最优模型保存路径


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
        self.data = torch.tensor(data, dtype=torch.float32).to(device)  # 转换为torch张量并移动到设备上
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(config.input_size, config.d_model)
        self.output_fc = nn.Linear(config.input_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=4,
            dim_feedforward=4 * config.input_size,
            batch_first=True,
            dropout=0.1,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=4,
            dropout=0.1,
            dim_feedforward=4 * config.input_size,
            batch_first=True,
            device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.fc = nn.Linear(config.output_size * config.d_model, config.output_size)
        self.fc1 = nn.Linear(config.timestep * config.d_model, config.d_model)
        self.fc2 = nn.Linear(config.d_model, config.output_size*config.input_size)

    def forward(self, x):
        # x:[32, 96, 7]
        batch_size=x.shape[0]
        x = self.input_fc(x)# x:[32, 96, 512]
        x = self.pos_emb(x)# x:[32, 96, 512]
        x = self.encoder(x)#[32, 96, 512]
        

        x = x.flatten(start_dim=1)#[32, 96*512]
        
        x=F.relu(x)
        
        x = self.fc1(x)
        out = self.fc2(x)
        out=out.view(x.shape[0],config.output_size,-1)

        return out#[32, 96]
    
    


if __name__ == "__main__":
    
    seed = Config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 设置NumPy随机种子
    np.random.seed(seed)
    
    predictions_list=[]
    labels_list=[]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    
    
    model = TransformerModel().to(device)
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
                
                plt.savefig('image/savefig_example_{}.png'.format("transformer"))
                plt.show()
                print("save success!")
    
    x1 = np.arange(0, config.epochs) 
    
    plt.figure(figsize=(12, 8))
    plt.plot(x1,np.array(loss_lst), "b",label="train_loss",linewidth=2)
    plt.legend()
    
    plt.savefig('image/savefig_example_{}.png'.format("transformer_loss"))
    plt.show()    
    
    print(f'Final training loss: {single_loss.item():10.10f}')
    
    print(f'Final test loss: {config.best_loss.item():10.10f}')




