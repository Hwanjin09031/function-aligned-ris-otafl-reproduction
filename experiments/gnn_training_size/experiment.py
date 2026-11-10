import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
from protocol_source import samples,c,digest
from network import RISNet,features,predict
import torch
from torch import nn
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def dump(p,o):p.write_text(json.dumps(o,indent=2))
def prepare(e):
 t=time.time();out=R/'data'/str(e);out.mkdir(exist_ok=True)
 old=np.load(R/'reference'/f'gnn_{e}/teacher.npz');rep=json.loads((R/'reference'/f'gnn_{e}/report.json').read_text())
 hd,C,cv=c.moments(samples(e,stream=2));assert digest(hd,C,cv)==rep['teacher_input_hash']
 books=np.load(R/'reference/training'/f'seed_{e}/L16/books.npz')
 he,Ce,cve=c.moments(samples(e,2400,stream=6));initial=c.bank(he,Ce,books['function'],cve,.01)
 th,b,v,j,hist=c.ao(he,Ce,cve,.01,initial[0],iterations=10,bits=2)
 assert np.all(j<=initial[-1]+1e-10)
 x=np.concatenate([features(hd[:800],C[:800]),features(he,Ce)])
 labels=np.concatenate([old['labels'][:800],np.mod(np.rint(np.angle(th)/(np.pi/2)).astype(int),4)])
 np.savez_compressed(out/'train.npz',x=x,labels=labels,mu=old['mu'],sd=old['sd'],extra_teacher_phases=th,extra_teacher_mse=j,extra_teacher_history=hist)
 np.savez_compressed(out/'validation.npz',hd=hd[800:],C=C[800:],cv=cv[800:],labels=old['labels'][800:],teacher_mse=old['mse'][800:])
 ht,Ct,cvt=c.moments(samples(e,stream=3));assert digest(ht,Ct,cvt)==rep['test_input_hash']
 np.savez_compressed(out/'test.npz',hd=ht,C=Ct,cv=cvt)
 base=np.load(R/'reference'/f'gnn_{e}/test_arrays.npz')
 # Reload and evaluate original GNN under restored code before new training.
 st=torch.load(R/'reference'/f'gnn_{e}/gnn.pt',weights_only=True);net=RISNet(st['feature_dim']);net.load_state_dict(st['state']);net.eval()
 ph=predict(net,ht,Ct,old['mu'],old['sd']);nmse=c.solve(c.effective(ht,Ct,ph),cvt,.01)[-1]*8
 err=float(np.max(abs(nmse-base['gnn|conditional_nmse'])));assert err<1e-10
 dump(out/'checks.json',{'original_input_hashes_match':True,'original_gnn_test_max_abs_error':err,'train_n':len(x),'original800_nested':True,'validation_n':200,'test_n':1000,'extra_teacher_nonincrease':True,'seconds':time.time()-t})
 print('PREPARED',e,round(time.time()-t,1),flush=True)
def train(e,n,seed,full_batch=False):
 t=time.time();out=R/'runs'/(f'{e}_{n}_{seed}'+('_fullbatch' if full_batch else ''));out.mkdir(exist_ok=True)
 d=np.load(R/'data'/str(e)/'train.npz');val=np.load(R/'data'/str(e)/'validation.npz')
 X=torch.from_numpy((d['x'][:n]-d['mu'])/d['sd']);Y=torch.from_numpy(d['labels'][:n]).long()
 VX=torch.from_numpy((features(val['hd'],val['C'])-d['mu'])/d['sd']);VY=torch.from_numpy(val['labels']).long()
 torch.manual_seed(seed);net=RISNet(X.shape[-1]);init_hash=digest(*[p.detach().numpy() for p in net.parameters()]);opt=torch.optim.Adam(net.parameters(),lr=.001)
 step=0;epoch=0;seen=0;buffer=torch.empty(0,dtype=torch.long);best=float('inf');records=[];losses=[];last=0
 while step<2600:
  epoch+=1
  if full_batch:
   while len(buffer)<64:buffer=torch.cat([buffer,torch.randperm(n)])
   order=buffer[:64];buffer=buffer[64:]
  else:order=torch.randperm(n)
  for a in range(0,len(order),64):
   ids=order[a:a+64];net.train();loss=nn.functional.cross_entropy(net(X[ids]).transpose(1,2),Y[ids]);opt.zero_grad();loss.backward();opt.step()
   step+=1;seen+=len(ids);losses.append(float(loss.item()))
   if step%65==0:
    net.eval();ph=predict(net,val['hd'],val['C'],d['mu'],d['sd']);score=float(c.solve(c.effective(val['hd'],val['C'],ph),val['cv'],.01)[-1].mean())
    with torch.no_grad():vce=float(nn.functional.cross_entropy(net(VX).transpose(1,2),VY).item())
    records.append({'step':step,'epoch':((seen+n-1)//n if full_batch else epoch),'seen_samples':seen,'train_ce_interval':float(np.mean(losses)),'validation_ce':vce,'validation_mse':score});losses=[]
    if score<best:
     best=score;last=step;torch.save({'state':net.state_dict(),'feature_dim':X.shape[-1]},out/'best.pt')
   if step==2600:break
 net.load_state_dict(torch.load(out/'best.pt',weights_only=True)['state']);net.eval()
 # Test loaded only after validation selection is complete.
 test=np.load(R/'data'/str(e)/'test.npz');ph=predict(net,test['hd'],test['C'],d['mu'],d['sd']);bb,vv,j=c.solve(c.effective(test['hd'],test['C'],ph),test['cv'],.01)
 assert np.max(abs(bb))<=1+1e-12 and np.all(np.isfinite(j))
 np.savez_compressed(out/'test_results.npz',conditional_nmse=j*8,phases=ph)
 dump(out/'report.json',{'environment':e,'n_train':n,'initialization_seed':seed,'initial_weights_hash':init_hash,'selected_step':last,'selected_validation_mse':best,'test_nmse':float((j*8).mean()),'stop_step':step,'epochs_started':((seen+n-1)//n if full_batch else epoch),'samples_seen':seen,'full_batch_control':full_batch,'history':records,'seconds':time.time()-t})
 print('TRAINED',e,n,seed,round(time.time()-t,1),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(int(sys.argv[2]))
 else:train(*map(int,sys.argv[2:]),full_batch=(sys.argv[1]=='control'))
