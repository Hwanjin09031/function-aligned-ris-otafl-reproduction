"""Receive-array sensitivity; fit, freeze and held-out test are separate."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys,time,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import training as tr
P=json.loads((R/'analysis_plan.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2))
def dest(seed,Nr):return R/'results'/f'{seed}_n{Nr}'
def data(seed,stream,n,Nr):return tr.data(seed,stream,n,P['M'],Nr)
def nmse(book,d):return -tr.score_matrix(book,*d,'function').max(1)*P['K']
def fit(job):
 seed,Nr=job;d=dest(*job);d.mkdir(parents=True,exist_ok=True)
 if (d/'frozen.json').exists():return dict(seed=seed,Nr=Nr,cached=True)
 start=time.perf_counter();full=data(seed,1,2000,Nr);train=tuple(a[:1000] for a in full);val=data(seed,2,300,Nr)
 alphabet=tr.c.legacy.phase_alphabet(2);tr.PASSES=3;pool,random=tr.make_pool(full[0],full[1],seed,alphabet)
 rec=dict(seed=seed,Nr=Nr,M=64,bits=2,ns=1000,plan_sha256=sha(R/'analysis_plan.json'),training_digest=tr.digest(*train),validation_digest=tr.digest(*val),pool_digest=tr.digest(pool),random_digest=tr.digest(random),methods={});arrays=dict(pool=pool,random=random)
 for kind in P['methods']:
  initial=pool[tr.greedy(tr.score_matrix(pool,*train,kind),16)];cp,history,seconds=tr.refine(initial,*train,kind,alphabet)
  vals=np.stack([nmse(book,val) for book in cp]);means=vals.mean(1);arrays[kind+'_checkpoints']=np.array(cp);arrays[kind+'_validation_nmse']=vals
  assert np.array(cp).shape==(4,16,64) and max(np.min(abs(b[...,None]-alphabet),axis=-1).max() for b in cp)<1e-12
  rec['methods'][kind]=dict(history=history,validation_nmse=means.tolist(),selected3=int(means.argmin()),refinement_seconds=seconds,checkpoint_digests=[tr.digest(b) for b in cp])
 np.savez_compressed(d/'frozen.npz',**arrays);rec['fit_total_seconds']=time.perf_counter()-start;save(d/'frozen.json',rec);return dict(seed=seed,Nr=Nr,seconds=round(rec['fit_total_seconds'],1))
def freeze():
 out=dict(created_unix=time.time(),plan_sha256=sha(R/'analysis_plan.json'),driver_sha256=sha(Path(__file__)),files={})
 for s in P['environment_seeds']:
  for Nr in P['receive_antennas']:
   for name in ['frozen.npz','frozen.json']:
    p=dest(s,Nr)/name
    if not p.exists():raise RuntimeError(f'Missing fit {p}')
    out['files'][str(p.relative_to(R))]=sha(p)
 assert not any((dest(s,Nr)/'test.json').exists() for s in P['environment_seeds'] for Nr in P['new_receive_antennas'])
 save(R/'pretest_frozen_manifest.json',out);print('All configurations frozen before held-out evaluation.',flush=True)
def verify():
 f=json.loads((R/'pretest_frozen_manifest.json').read_text());assert f['plan_sha256']==sha(R/'analysis_plan.json') and f['driver_sha256']==sha(Path(__file__))
 for p,h in f['files'].items():assert sha(R/p)==h,p
def test(job):
 seed,Nr=job;d=dest(*job)
 if (d/'test.json').exists():return dict(seed=seed,Nr=Nr,cached=True)
 rec=json.loads((d/'frozen.json').read_text());f=np.load(d/'frozen.npz');testdata=data(seed,3,1000,Nr);hd,C,cov=testdata
 selected=[f[k+'_checkpoints'][rec['methods'][k]['selected3']] for k in P['methods']];values={k+'_3':nmse(b,testdata) for k,b in zip(P['methods'],selected)};values['random']=nmse(f['random'],testdata)
 initial=tr.c.bank(hd,C,np.concatenate(selected),cov,.01)[0];theta,tx,rx,mse,history=tr.c.ao(hd,C,cov,.01,initial,iterations=10,bits=2)
 assert np.min(abs(theta[...,None]-tr.c.legacy.phase_alphabet(2)),axis=-1).max()<1e-12;values['ao']=mse*8;np.savez_compressed(d/'test_values.npz',**values)
 save(d/'test.json',dict(seed=seed,Nr=Nr,M=64,bits=2,ns=1000,test_digest=tr.digest(*testdata),frozen_sha256=sha(d/'frozen.npz'),means={k:float(v.mean()) for k,v in values.items()},ao_mean_mse_history=history.tolist()));return dict(seed=seed,Nr=Nr)
def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['fit','freeze','test']);a.add_argument('--workers',type=int,default=7);x=a.parse_args()
 if x.phase=='freeze':freeze();return
 if x.phase=='test':verify()
 jobs=[(s,Nr) for s in P['environment_seeds'] for Nr in P['new_receive_antennas']];start=time.time();save(R/f'execution_{x.phase}.json',dict(start_unix=start,python=platform.python_version(),numpy=np.__version__,workers=x.workers,jobs=jobs,plan_sha256=sha(R/'analysis_plan.json'),driver_sha256=sha(Path(__file__))))
 with ProcessPoolExecutor(max_workers=x.workers) as ex:
  for i,f in enumerate(as_completed([ex.submit(fit if x.phase=='fit' else test,j) for j in jobs]),1):print(json.dumps(dict(completed=i,total=len(jobs),**f.result())),flush=True)
 print(json.dumps(dict(phase=x.phase,wall_seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
