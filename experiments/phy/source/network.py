import numpy as np
import torch
from torch import nn

class RISNet(nn.Module):
 def __init__(self,F,M=64):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(F,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU())
  self.message=nn.Sequential(nn.Linear(192,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU())
  self.output=nn.Linear(96,M*4);self.M=M
 def forward(self,x):
  z=self.encoder(x);context=z.mean(1,keepdim=True).expand_as(z)
  return self.output(self.message(torch.cat([z,context],-1)).mean(1)).reshape(-1,self.M,4)
def features(hd,C):
 x=np.concatenate([hd,C.reshape(*C.shape[:2],-1)],axis=-1)
 return np.concatenate([x.real,x.imag],axis=-1).astype(np.float32)
def predict(model,hd,C,mu,sd):
 with torch.no_grad():ix=model(torch.from_numpy((features(hd,C)-mu)/sd)).argmax(-1).numpy()
 return np.exp(1j*np.pi/2*ix)
