"""RIS-size sensitivity with fixed L=16; fit, freeze, and test are separate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
import training as tr
P=json.loads((R/'analysis_plan.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2))
def dest(seed,M):return R/'results'/f'{seed}_m{M}'
def data(seed,stream,n,M):return tr.data(seed,stream,n,M,P['Nr'])
def nmse(book,d):return -tr.score_matrix(book,*d,'function').max(axis=1)*P['K']
def fit(job):
 seed,M=job;d=dest(*job);d.mkdir(parents=True,exist_ok=True)
 if (d/'frozen.json').exists():return dict(seed=seed,M=M,cached=True)
 start=time.perf_counter();full=data(seed,P['train_stream'],P['training_generation_count'],M)
 train=tuple(a[:P['training_states']] for a in full);val=data(seed,P['validation_stream'],P['validation_states'],M)
 alphabet=tr.c.legacy.phase_alphabet(P['bits']);tr.PASSES=P['outer_sweeps']
 pool,random=tr.make_pool(full[0],full[1],seed,alphabet)
 rec=dict(seed=seed,M=M,bits=P['bits'],ns=P['training_states'],plan_sha256=sha(R/'analysis_plan.json'),
   training_digest=tr.digest(*train),validation_digest=tr.digest(*val),pool_digest=tr.digest(pool),random_digest=tr.digest(random),methods={})
 arrays=dict(pool=pool,random=random)
 for kind in P['methods']:
  initial=pool[tr.greedy(tr.score_matrix(pool,*train,kind),P['L'])]
  cp,history,seconds=tr.refine(initial,*train,kind,alphabet)
  vals=np.stack([nmse(book,val) for book in cp]);means=vals.mean(1)
  arrays[kind+'_checkpoints']=np.array(cp);arrays[kind+'_validation_nmse']=vals
  assert len(cp)==4 and np.array(cp).shape==(4,16,M)
  assert max(np.min(abs(b[...,None]-alphabet),axis=-1).max() for b in cp)<1e-12
  rec['methods'][kind]=dict(history=history,validation_nmse=means.tolist(),selected3=int(means.argmin()),refinement_seconds=seconds,checkpoint_digests=[tr.digest(b) for b in cp])
 np.savez_compressed(d/'frozen.npz',**arrays);rec['fit_total_seconds']=time.perf_counter()-start;save(d/'frozen.json',rec)
 return dict(seed=seed,M=M,seconds=round(rec['fit_total_seconds'],1))
def freeze():
 out=dict(created_unix=time.time(),plan_sha256=sha(R/'analysis_plan.json'),driver_sha256=sha(Path(__file__)),files={})
 for s in P['environment_seeds']:
  for M in P['ris_sizes']:
   for name in ['frozen.npz','frozen.json']:
    p=dest(s,M)/name
    if not p.exists():raise RuntimeError(f'Missing fit: {p}')
    out['files'][str(p.relative_to(R))]=sha(p)
 assert not any((dest(s,M)/'test.json').exists() for s in P['environment_seeds'] for M in P['new_ris_sizes'])
 save(R/'pretest_frozen_manifest.json',out);print('All configurations frozen before held-out evaluation.',flush=True)
def verify_freeze():
 f=json.loads((R/'pretest_frozen_manifest.json').read_text())
 assert f['plan_sha256']==sha(R/'analysis_plan.json') and f['driver_sha256']==sha(Path(__file__))
 for p,h in f['files'].items():assert sha(R/p)==h,p
def evaluate(job):
 seed,M=job;d=dest(*job)
 if (d/'test.json').exists():return dict(seed=seed,M=M,cached=True)
 rec=json.loads((d/'frozen.json').read_text());f=np.load(d/'frozen.npz')
 test=data(seed,P['test_stream'],P['test_states'],M);hd,C,cov=test
 selected=[f[k+'_checkpoints'][rec['methods'][k]['selected3']] for k in P['methods']]
 values={k+'_3':nmse(book,test) for k,book in zip(P['methods'],selected)};values['random']=nmse(f['random'],test)
 initial=tr.c.bank(hd,C,np.concatenate(selected),cov,P['noise'])[0]
 theta,tx,rx,mse,history=tr.c.ao(hd,C,cov,P['noise'],initial,iterations=10,bits=P['bits'])
 phase=float(np.min(abs(theta[...,None]-tr.c.legacy.phase_alphabet(P['bits'])),axis=-1).max());assert phase<1e-12
 values['ao']=mse*P['K'];np.savez_compressed(d/'test_values.npz',**values)
 save(d/'test.json',dict(seed=seed,M=M,bits=P['bits'],ns=P['training_states'],test_digest=tr.digest(*test),frozen_sha256=sha(d/'frozen.npz'),means={k:float(v.mean()) for k,v in values.items()},ao_phase_residual=phase,ao_mean_mse_history=history.tolist()))
 return dict(seed=seed,M=M)
def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['fit','freeze','test']);a.add_argument('--workers',type=int,default=7);args=a.parse_args()
 if args.phase=='freeze':freeze();return
 if args.phase=='test':verify_freeze()
 jobs=[(s,M) for s in P['environment_seeds'] for M in P['new_ris_sizes']]
 start=time.time();save(R/f'execution_{args.phase}.json',dict(start_unix=start,python=platform.python_version(),numpy=np.__version__,workers=args.workers,jobs=jobs,plan_sha256=sha(R/'analysis_plan.json'),driver_sha256=sha(Path(__file__))))
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures=[ex.submit(fit if args.phase=='fit' else evaluate,j) for j in jobs]
  for i,f in enumerate(as_completed(futures),1):print(json.dumps(dict(completed=i,total=len(jobs),**f.result())),flush=True)
 print(json.dumps(dict(phase=args.phase,wall_seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
