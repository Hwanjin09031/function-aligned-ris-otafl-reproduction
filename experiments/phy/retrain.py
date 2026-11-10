"""Train the AO-imitation network on generated channel samples."""
from run import *
def gnn(seed):
 import torch
 from torch import nn
 sys.path.insert(0,str(R/'source'))
 from network import RISNet,predict,features
 torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
 start=time.time();out=R/'retrain_check'/f'gnn_{seed}';out.mkdir(exist_ok=True)
 books=np.load(R/'reference/training'/f'seed_{seed}/L16/books.npz')
 teacher=samples(seed,stream=2);hd,C,cv=c.moments(teacher)
 teacher_input_hash=digest(hd,C,cv)
 initial=c.bank(hd,C,books['function'],cv,.01)
 th,b,v,j,hist=c.ao(hd,C,cv,.01,initial[0],iterations=10,bits=2)
 assert np.all(j<=initial[-1]+1e-10)
 labels=np.mod(np.rint(np.angle(th)/(np.pi/2)).astype(int),4)
 x=features(hd,C);mu=x[:800].mean((0,1),keepdims=True);sd=np.maximum(x[:800].std((0,1),keepdims=True),1e-3)
 X=torch.from_numpy((x-mu)/sd);Y=torch.from_numpy(labels).long()
 net=RISNet(x.shape[-1]);assert sum(p.numel() for p in net.parameters())==112000
 opt=torch.optim.Adam(net.parameters(),lr=.001);best=np.inf;last=0;records=[]
 for epoch in range(1,201):
  net.train();order=torch.randperm(800);losses=[]
  for a in range(0,800,64):
   ids=order[a:a+64];loss=nn.functional.cross_entropy(net(X[ids]).transpose(1,2),Y[ids]);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.item()))
  if epoch%5==0:
   net.eval();tp=predict(net,hd[800:],C[800:],mu,sd);val=c.solve(c.effective(hd[800:],C[800:],tp),cv[800:],.01)[-1]
   score=float(val.mean());records.append({'epoch':epoch,'train_ce':float(np.mean(losses)),'validation_mse':score})
   if score<best:
    best=score;last=epoch;torch.save({'state':net.state_dict(),'feature_dim':x.shape[-1]},out/'gnn.pt')
   if epoch-last>=40:break
 net.load_state_dict(torch.load(out/'gnn.pt',weights_only=True)['state']);net.eval()
 np.savez_compressed(out/'teacher.npz',phases=th,labels=labels,mse=j,history=hist,mu=mu,sd=sd)
 # Test only after checkpoint selection is completed.
 ss=samples(seed,stream=3);ht=np.array([s.h_d_true for s in ss]);Ct=np.array([s.C_true for s in ss]);hd,C,cv=c.moments(ss)
 d={k:c.bank(hd,C,books[k],cv,.01) for k in KINDS}
 tp=predict(net,hd,C,mu,sd);d['gnn']=(tp,*c.solve(c.effective(hd,C,tp),cv,.01))
 permutation=np.random.default_rng(seed+50000).permutation(8)
 perm=predict(net,hd[:,permutation],C[:,permutation],mu,sd);mismatch=int(np.sum(abs(perm-tp)>1e-12));assert mismatch==0
 theta=np.random.default_rng(4477).choice(c.legacy.phase_alphabet(2),(1000,64));d['random']=(theta,*c.solve(c.effective(hd,C,theta),cv,.01))
 for name,bits in [('ao_quantized',2),('ao_continuous',None)]:d[name]=c.ao(hd,C,cv,.01,d['function'][0],iterations=10,bits=bits)[:4]
 arrays={};rows=[]
 for name,(theta,b,v,jj) in d.items():
  conditional=jj*8;realized=c.obj(c.effective(ht,Ct,theta),b,v,np.zeros_like(cv),.01,np.ones(8)/8)*8
  arrays[f'{name}|conditional_nmse']=conditional;arrays[f'{name}|realized_nmse']=realized
  assert np.max(abs(b))<=1+1e-12
  rows.append({'seed':seed,'method':name,'conditional_nmse':float(conditional.mean()),'realized_nmse':float(realized.mean())})
 # Input boundary check: all non-oracle designs above are formed from nominal
 # hd/C/cov only. Realized channels are accessed only in the evaluation loop.
 np.savez_compressed(out/'test_arrays.npz',**arrays)
 report={'seed':seed,'train_samples':800,'validation_samples':200,'test_samples':1000,'parameters':112000,'feature_dimension':520,'selected_epoch':last,'stop_epoch':epoch,'validation_mse':best,'teacher_validation_mse':float(j[800:].mean()),'validation_history':records,'permutation_phase_mismatches':mismatch,'teacher_input_hash':teacher_input_hash,'test_input_hash':digest(hd,C,cv),'rows':rows,'elapsed_seconds':time.time()-start,'scope':'New final-solver AO-imitation baseline; one training initialization per environment; conditional CSI only, no FL/GNN or pilot budget claim.'}
 dump(out/'report.json',report);print('GNN',seed,'PASS epoch',last,'seconds',round(time.time()-start,1),flush=True)

