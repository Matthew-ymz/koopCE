import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

class KoopmanOperator(nn.Module):
    def __init__(self, koopman_dim, delta_t, device="cpu"):
        super(KoopmanOperator, self).__init__()

        self.koopman_dim = koopman_dim
        self.delta_t = delta_t
        self.device = device
        
        self.linear_evolution = nn.Linear(self.koopman_dim, self.koopman_dim, bias=False)

    def forward(self, x, T):
        # x is B x 1 x Latent (Initial condition)
        
        Y = torch.zeros(x.shape[0], T, self.koopman_dim).to(self.device)
        
        # 获取初始状态
        y = x[:, 0, :]
        
        for t in range(T):
            y = self.linear_evolution(y)
            
            # 存储这一步的结果
            Y[:, t, :] = y

        return Y
    
class Lusch(nn.Module):
    def __init__(self,input_dim,koopman_dim,hidden_dim,delta_t=0.01,device="cpu"):
        super(Lusch,self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim,koopman_dim))

        self.decoder = nn.Sequential(nn.Linear(koopman_dim,hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim,input_dim))

        self.koopman = KoopmanOperator(koopman_dim,delta_t,device)

        self.device = device
        self.delta_t = delta_t

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.zeros((input_dim,)))
        self.register_buffer('std', torch.ones((input_dim,)))

    def forward(self,x):
        x = self.embed(x)
        x = self.recover(x)
        return x

    def embed(self,x):
        x = self._normalize(x)
        x = self.encoder(x)
        return x

    def recover(self,x):
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def koopman_operator(self,x,T=1):
        return self.koopman(x,T)

    def _normalize(self, x):
        return (x - self.mu[(None,)*(x.dim()-1)+(...,)])/self.std[(None,)*(x.dim()-1)+(...,)]

    def _unnormalize(self, x):
        return self.std[(None,)*(x.dim()-1)+(...,)]*x + self.mu[(None,)*(x.dim()-1)+(...,)]
    
def CE_loss(model):
    K_tensor = model.koopman.linear_evolution.weight
    _, S_tensor, _ = torch.linalg.svd(K_tensor, full_matrices=False)
    S_max = S_tensor.max()
    loss = 0
    for i in range(S_tensor.size()[0]):
        loss += S_tensor[i] * (S_max - S_tensor[i])
    return loss

def koopman_loss(x,model,Sp,T, alpha1=2, alpha2=1e-10, alpha_CE=0):
    # Sp < T
    MAX_T = max(Sp,T)

    encoder_x = model.embed(x)
    recover_x = model.recover(encoder_x)


    koopman_stepped = model.koopman_operator(encoder_x[:,[0],:],MAX_T)
    recover_koopman = model.recover(koopman_stepped[:,:(Sp-1),:])


    reconstruction_inf_loss = torch.norm(x-recover_x,p=float('inf'),dim=[-2,-1]).mean()
    prediction_inf_loss = torch.norm(x[:,1:Sp,:]-recover_koopman,p=float('inf'),dim=[-2,-1]).mean()


    lin_loss = F.mse_loss(encoder_x[:,1:T,:],koopman_stepped[:,:(T-1),:])
    pred_loss = F.mse_loss(recover_koopman,x[:,1:Sp,:],)
    reconstruction_loss = F.mse_loss(recover_x,x)
    inf_loss = reconstruction_inf_loss + prediction_inf_loss

    if alpha_CE > 0:
        loss = alpha1*(pred_loss + reconstruction_loss) + lin_loss + alpha2*inf_loss + alpha_CE*CE_loss(model)
    else:
        loss = alpha1*(pred_loss + reconstruction_loss) + lin_loss + alpha2*inf_loss
    return loss

def prediction_loss(x_recon,x_ahead,model):
    # Sp < T
    with torch.inference_mode():
        model.eval()
        Y = model.koopman_operator(model.embed(x_recon[:,[-1],:]),x_ahead.shape[1])
        prediction_loss = F.mse_loss(x_ahead,model.recover(Y))

    return prediction_loss

class get_dataset(Dataset):

    def __init__(self,X,horizon):

        self.X = X
        self.horizon = horizon
        self.D = X.shape[-1]
        self.T = X.shape[1]-self.horizon
        self.mu = torch.tensor([torch.mean(X[:,:,i]) for i in range(self.D)])
        self.std = torch.tensor([torch.std(X[:,:,i]) for i in range(self.D)])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is int:
            idx = [idx]

        start = torch.randint(low=0,high=self.T+1,size=(len(idx),))
        windows = torch.tensor([list(range(i,i+self.horizon)) for i in start]).unsqueeze(-1).repeat(1,1,self.D)
        x = torch.gather(self.X[idx],1,windows).squeeze()

        return x
    
def prepare_dataloader(data, max_step, batch_size=64):
    """
    将单条轨迹切片成 [Batch, Max_Time_Step + 1, Features]
    """
    num_samples = len(data) - max_step
    
    X = []
    for i in range(num_samples):
        # 截取长度为 max_step + 1 的序列
        window = data[i : i + max_step + 1]
        X.append(window)
        
    X = np.array(X) # [Samples, Time, Features]
    
    # 转 Tensor
    dataset = TensorDataset(torch.from_numpy(X).float())
    
    # 划分 Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

import torch

def create_sequences(data, seq_length, stride=1):
    """
    将长序列切分为短序列 (滑动窗口)
    
    Args:
        data: 输入数据，形状可以是 (Total_Time, Input_Dim) 或 (Total_Time,)
        seq_length: 每个短序列的时间步长 (T)
        stride: 滑动步长。默认为1 (重叠最多)，如果不想重叠则设为 seq_length
        
    Returns:
        3阶张量: (Num_Samples, seq_length, Input_Dim)
    """
    # 1. 确保数据是 Tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
        
    # 2. 如果数据是 1D (Time,), 升维成 (Time, 1)
    if data.dim() == 1:
        data = data.unsqueeze(-1)
        
    # Total_Time, Input_Dim
    num_time_steps, input_dim = data.shape
    
    if num_time_steps < seq_length:
        raise ValueError("数据总长度小于序列长度，无法切分")

    # 3. 使用 unfold 创建滑动窗口
    # data.unfold(dimension, size, step)
    # 对第0维(时间)展开。
    # 结果形状会变成: (Num_Samples, Input_Dim, seq_length)
    sequences = data.unfold(0, seq_length, stride)
    
    # 4. 调整维度顺序
    # 我们需要 (Num_Samples, seq_length, Input_Dim)
    # 目前是 (Num_Samples, Input_Dim, seq_length)，所以需要交换最后两维
    sequences = sequences.permute(0, 2, 1)
    
    return sequences