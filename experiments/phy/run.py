import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
import training as tr
from power_solver import solve
c=tr.c;assert c.solve is solve
KINDS=('function','fairness','rate','power')
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def samples(seed,n=1000,rho=.05,stream=1,K=8,N=4):
 env=c.legacy.create_clustered_environment(np.random.default_rng(seed),K,N,64,4,kappa_G=10.)
 return c.cc.generate_clustered_samples(np.random.default_rng(seed+10000*stream),env,n,K,N,64,kappa_hr=10**.3,csi_error_ratio=rho)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def train(seed,L):
 start=time.time();out=R/'results'/f'seed_{seed}'/f'L{L}';out.mkdir(parents=True,exist_ok=True)
 ref=R/'reference/training'/f'seed_{seed}'
 if L==16:
  import shutil
  for name in ('books.npz','checkpoints.npz','frozen.json'):shutil.copy2(ref/'L16'/name,out/name)
  print('REUSED',seed,L,flush=True);return
 train=c.moments(samples(seed));val=c.moments(samples(seed,300,stream=4))
 saved=json.loads((ref/'prepared.json').read_text());assert digest(*train)==saved['train_hash'];assert digest(*val)==saved['validation_hash']
 pool=np.load(ref/'pool.npz')['pool'];scores=np.load(ref/'pool_scores.npz');books={};arrays={};hist={}
 for kind in KINDS:
  ids=tr.greedy(scores[kind],L);checks,h,seconds=tr.refine(pool[ids],*train,kind,c.legacy.phase_alphabet(2))
  vscore=[float(-tr.score_matrix(b,*val,'function').max(1).mean()*8) for b in checks];ix=int(np.argmin(vscore));books[kind]=checks[ix]
  for i,b in enumerate(checks):arrays[f'{kind}_pass{i}']=b
  hist[kind]={'selected_pass':ix,'validation_nmse':vscore,'training':h,'greedy_indices':ids.tolist(),'book_hash':digest(books[kind]),'seconds':seconds}
  print('TRAIN',seed,L,kind,round(time.time()-start,1),flush=True)
 np.savez_compressed(out/'books.npz',**books);np.savez_compressed(out/'checkpoints.npz',**arrays)
 dump(out/'frozen.json',{'seed':seed,'L':L,'train_hash':digest(*train),'validation_hash':digest(*val),'pool_hash':digest(pool),'diagnostics':hist,'elapsed_seconds':time.time()-start})
def evaluate(ss,books,net,scale,noise):
 from network import predict
 hd,C,cv=c.moments(ss);B,K,N=hd.shape
 d={k:c.bank(hd,C,books[k],cv,noise) for k in KINDS}
 for name,bits in [('ao_quantized',2),('ao_continuous',None)]:d[name]=c.ao(hd,C,cv,noise,d['function'][0],iterations=10,bits=bits)[:4]
 theta=predict(net,hd,C,scale['mu'],scale['sd']);d['gnn']=(theta,*solve(c.effective(hd,C,theta),cv,noise))
 theta=np.random.default_rng(4477).choice(c.legacy.phase_alphabet(2),(B,64));d['random']=(theta,*solve(c.effective(hd,C,theta),cv,noise))
 d['no_ris']=(np.zeros((B,64)),*solve(hd,np.broadcast_to(ss[0].csi.q_d*np.eye(N),(B,K,N,N)),noise))
 ht=np.array([s.h_d_true for s in ss]);Ct=np.array([s.C_true for s in ss]);zero=np.zeros_like(cv)
 d['perfect_ao_continuous']=c.ao(ht,Ct,zero,noise,d['function'][0],iterations=10)[:4]
 out={}
 for kind,(theta,b,v,j) in d.items():
  assert np.max(abs(b))<=1+1e-12 and np.all(np.isfinite(j))
  out[f'{kind}|conditional_nmse']=j*K
  out[f'{kind}|realized_nmse']=c.obj(c.effective(ht,Ct,theta),b,v,zero,noise,np.ones(K)/K)*K
  if kind in ('ao_quantized','ao_continuous'):assert np.all(j<=d['function'][-1]+1e-10)
 return out
def phy(seed):
 import torch
 from network import RISNet
 torch.set_num_threads(1);start=time.time();out=R/'results'/f'seed_{seed}';out.mkdir(exist_ok=True)
 ref=R/'reference'/f'gnn_{seed}';st=torch.load(ref/'gnn.pt',weights_only=True);net=RISNet(st['feature_dim']);net.load_state_dict(st['state']);net.eval();scale=np.load(ref/'teacher.npz')
 books=np.load(R/'reference/training'/f'seed_{seed}/L16/books.npz');arrays={};central=None
 for sweep,grid in [('snr',[0,10,20,30,40]),('rho',[0,.01,.05,.1,.2])]:
  for x in grid:
   rho=.05 if sweep=='snr' else x;noise=10**(-x/10) if sweep=='snr' else .01
   if (rho,noise)==(.05,.01) and central is not None:res=central
   else:res=evaluate(samples(seed,rho=rho,stream=3),books,net,scale,noise)
   if (rho,noise)==(.05,.01):central=res
   for k,v in res.items():arrays[f'{sweep}|{x}|{k}']=v
   print('PHY',seed,sweep,x,round(time.time()-start,1),flush=True)
 previous=np.load(ref/'test_arrays.npz')
 for k in previous.files:assert np.allclose(central[k],previous[k],rtol=0,atol=1e-12),(seed,k,float(np.max(abs(central[k]-previous[k]))))
 np.savez_compressed(out/'phy.npz',**arrays);dump(out/'phy_complete.json',{'seed':seed,'states_per_condition':1000,'central_previous_arrays_exact':len(previous.files),'array_count':len(arrays),'elapsed_seconds':time.time()-start})
def sizes(seed):
 out=R/'results'/f'seed_{seed}';ss=samples(seed,stream=3);hd,C,cv=c.moments(ss);ht=np.array([s.h_d_true for s in ss]);Ct=np.array([s.C_true for s in ss]);zero=np.zeros_like(cv);arrays={}
 for L in [2,4,8,16,32]:
  books=np.load(out/f'L{L}/books.npz')
  for kind in KINDS:
   theta,b,v,j=c.bank(hd,C,books[kind],cv,.01);arrays[f'L|{L}|{kind}|conditional_nmse']=j*8;arrays[f'L|{L}|{kind}|realized_nmse']=c.obj(c.effective(ht,Ct,theta),b,v,zero,.01,np.ones(8)/8)*8
 if (out/'phy.npz').exists():
  old=np.load(out/'phy.npz')
  for k in KINDS:assert np.allclose(arrays[f'L|16|{k}|conditional_nmse'],old[f'rho|0.05|{k}|conditional_nmse'],rtol=0,atol=1e-12)
 np.savez_compressed(out/'sizes.npz',**arrays);print('SIZES',seed,'PASS',flush=True)
if __name__=='__main__':
 mode=sys.argv[1];seed=int(sys.argv[2]);globals()[mode](seed,*[int(x) for x in sys.argv[3:]])
