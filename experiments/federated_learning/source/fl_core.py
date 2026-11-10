from pathlib import Path
from copy import deepcopy
import os,numpy as np
import torch
from torch import nn
class CNN(nn.Module):
 def __init__(self):
  super().__init__();self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
  self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,10))
 def forward(self,x):return self.classifier(self.features(x))
def vec(m):return torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone()
def setvec(m,x):torch.nn.utils.vector_to_parameters(x,m.parameters())
def dataset():
 raw=Path(os.environ.get('RIS_MNIST_RAW','data/MNIST/raw'))
 def read(prefix,n,seed):
  X=np.frombuffer((raw/(prefix+'-images-idx3-ubyte')).read_bytes(),np.uint8,offset=16).reshape(-1,1,28,28).copy()
  Y=np.frombuffer((raw/(prefix+'-labels-idx1-ubyte')).read_bytes(),np.uint8,offset=8).copy()
  ix=np.sort(np.random.default_rng(seed).choice(len(Y),n,replace=False))
  return torch.tensor(X[ix],dtype=torch.float32)/127.5-1,torch.tensor(Y[ix],dtype=torch.long)
 return *read('train',12000,2026),*read('t10k',3000,2027)
def partition(labels,seed):
 rng=np.random.default_rng(seed)
 for _ in range(100):
  ids=[[] for _ in range(8)]
  for cl in range(10):
   ix=np.where(labels==cl)[0];rng.shuffle(ix);f=rng.dirichlet(np.full(8,.3))
   for k,xx in enumerate(np.split(ix,(np.cumsum(f)[:-1]*len(ix)).astype(int))):ids[k]+=xx.tolist()
  if min(map(len,ids))>=20:return [np.array(x) for x in ids]
 raise RuntimeError('partition')
def evaluate(m,X,Y):
 m.eval();correct=0;loss=0
 with torch.no_grad():
  for start in range(0,len(Y),256):
   out=m(X[start:start+256]);yy=Y[start:start+256]
   loss+=nn.functional.cross_entropy(out,yy,reduction='sum').item();correct+=(out.argmax(1)==yy).sum().item()
 return loss/len(Y),correct/len(Y)
def updates(m,X,Y,plans):
 base=vec(m);out=[]
 for plan in plans:
  local=deepcopy(m);local.train();opt=torch.optim.SGD(local.parameters(),lr=.03,momentum=.9)
  for ix in plan:
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(local(X[ix]),Y[ix]);loss.backward();opt.step()
  out.append((vec(local)-base).numpy().astype(float))
 return np.stack(out)
