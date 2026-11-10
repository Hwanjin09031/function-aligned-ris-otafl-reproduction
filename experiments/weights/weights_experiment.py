import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,json,hashlib,types
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
import core as c
from power_solver import solve
c.solve=solve
ALPHABET=np.array([1,1j,-1,-1j])
def utility(g,w,kind):
    return np.min(np.sum(abs(g)**2,axis=-1)/np.maximum(w*w,1e-300),axis=-1) if kind=='fairness' else c.metric(g,.01,kind)
# Reuse the audited coordinate-refit functions unchanged, with standalone imports.
s=(R/'source/refit_original.py').read_text();s=s[s.index('def score_matrix'):]
exec(s,globals())
old=types.SimpleNamespace(utility=utility)
def digest(*x):
    h=hashlib.sha256()
    for a in x:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
def data(seed,stream,n):
    env=c.legacy.create_clustered_environment(np.random.default_rng(seed),8,4,64,4,kappa_G=10.)
    ss=c.cc.generate_clustered_samples(np.random.default_rng(seed+stream*10000),env,n,8,4,64,kappa_hr=10**.3,csi_error_ratio=.05)
    hd,C,cv=c.moments(ss)
    return hd,C,cv,np.array([s.h_d_true for s in ss]),np.array([s.C_true for s in ss])
def weights(seed,stream,regime,n):
    if regime=='equal':return np.ones((n,8))/8
    if regime=='fixed_unequal':return np.broadcast_to(np.random.default_rng(seed+700000).permutation(np.arange(1,9))/36,(n,8)).copy()
    return np.random.default_rng(seed+800000+stream*10000).dirichlet(np.full(8,.5),n)
def make_pool(hd,C,w,seed):
    rng=np.random.default_rng(seed+200);ids=rng.choice(len(hd),64,replace=False)
    initial=rng.choice(ALPHABET,(64,64));chunks=[]
    for kind in ['rate','power','fairness']:
        theta=initial.copy()
        for _ in range(3):
            g=c.effective(hd[ids],C[ids],theta)
            for m in range(64):
                without=g-C[ids,:,:,m]*theta[:,m,None,None]
                cand=without[:,None]+C[ids,None,:,:,m]*ALPHABET[None,:,None,None]
                u=utility(cand.reshape(-1,8,4),np.repeat(w[ids],4,axis=0),kind).reshape(64,4)
                ix=u.argmax(1);theta[:,m]=ALPHABET[ix];g=cand[np.arange(64),ix]
        chunks.append(theta)
    chunks.append(rng.choice(ALPHABET,(32,64)));pool=np.concatenate(chunks)
    _,ids=np.unique(np.mod(np.rint(np.angle(pool)/(np.pi/2)).astype(int),4),axis=0,return_index=True)
    return pool[np.sort(ids)]
def evaluate(book,d,w):
    hd,C,cv,ht,Ct=d;theta,b,v,j=c.bank(hd,C,book,cv,.01,w)
    real=c.obj(c.effective(ht,Ct,theta),b,v,np.zeros_like(cv),.01,w)
    assert np.max(abs(b))<=1+1e-12
    e=np.sum(w*w,axis=1)
    return {'conditional':j,'realized':real,'energy':e}, {'conditional_nmse':float(j.mean()/e.mean()),'realized_nmse':float(real.mean()/e.mean()),'mean_state_nmse':float(np.mean(j/e))}
def run(seed,regime):
    t=time.time();out=R/'runs'/f'{seed}_{regime}';out.mkdir(exist_ok=True)
    train=data(seed,1,1000);val=data(seed,2,300);w=weights(seed,1,regime,1000);wv=weights(seed,2,regime,300)
    pool=make_pool(*train[:2],w,seed);np.savez_compressed(out/'pool.npz',pool=pool)
    books={};diags={}
    for kind in ['function','fairness','rate','power']:
        scores=score_matrix(pool,*train[:3],w,kind);ids=greedy(scores,16)
        checkpoints,d=refine(pool[ids],*train[:3],w,kind,passes=3)
        vs=[validation(b,val,wv) for b in checkpoints];ix=int(np.argmin(vs));books[kind]=checkpoints[ix]
        d.update({'validation_nmse':vs,'selected_pass':ix,'init_indices':ids.tolist()});diags[kind]=d
        print(seed,regime,kind,'frozen pass',ix,flush=True)
    np.savez_compressed(out/'books.npz',**books)
    meta={'seed':seed,'regime':regime,'train_hash':digest(*train[:3]),'validation_hash':digest(*val[:3]),'train_weights_hash':digest(w),'pool_count':len(pool),'diagnostics':diags,'books_sha256':hashlib.sha256((out/'books.npz').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest()}
    (out/'frozen.json').write_text(json.dumps(meta,indent=2))
    test=data(seed,3,1000);wt=weights(seed,3,regime,1000);rows=[];raw={}
    for kind,book in books.items():
        a,b=evaluate(book,test,wt);rows.append({'seed':seed,'regime':regime,'method':kind,**b})
        raw.update({kind+'_'+k:v for k,v in a.items()})
    np.savez_compressed(out/'test.npz',**raw)
    (out/'report.json').write_text(json.dumps({'rows':rows,'seconds':time.time()-t,'test_hash':digest(*test),'status':'complete'},indent=2))
    print(seed,regime,'complete',time.time()-t,rows,flush=True)
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
