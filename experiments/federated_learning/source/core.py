"""Conditional-CSI radio kernels. All vectors use E|CN(0,q)|^2=q."""
import numpy as np
import conditional_csi as cc
import radio_core as legacy

def moments(samples):
 return (np.array([s.h_d_hat for s in samples]),np.array([s.C_hat for s in samples]),
         np.array([s.csi.user_covariance(np.ones(s.C_hat.shape[-1])) for s in samples]))
def effective(hd,C,theta):return hd+np.einsum('bknm,bm->bkn',C,theta)
def obj(g,b,v,cov,noise,w):
 a=np.einsum('bkn,bn->bk',g,v.conj())
 proj=np.einsum('bn,bknm,bm->bk',v.conj(),cov,v).real
 return np.sum(abs(b*a-w)**2+abs(b)**2*proj,axis=1)+noise*np.sum(abs(v)**2,axis=1)
def solve(g,cov,noise,w=None,steps=20,v0=None):
 B,K,N=g.shape
 w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
 v=np.sum(g*w[...,None],axis=1) if v0 is None else v0.copy()
 if v0 is None:v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
 b=np.zeros((B,K),complex);active=np.ones(B,bool);prev=np.full(B,np.inf)
 for _ in range(steps):
  a=np.einsum('bkn,bn->bk',g,v.conj());p=np.einsum('bn,bknm,bm->bk',v.conj(),cov,v).real
  bn=w*a.conj()/np.maximum(abs(a)**2+p,1e-15);bn/=np.maximum(abs(bn),1)
  A=noise*np.eye(N)[None]+np.einsum('bk,bkn,bkm->bnm',abs(bn)**2,g,g.conj())+np.einsum('bk,bknm->bnm',abs(bn)**2,cov)
  z=np.einsum('bk,bk,bkn->bn',bn,w,g)
  vn=np.linalg.solve(A,z[...,None])[...,0]
  val=obj(g,bn,vn,cov,noise,w)
  b[active]=bn[active];v[active]=vn[active]
  done=np.isfinite(prev)&(abs(prev-val)<=1e-8*np.maximum(1,prev));active&=~done;prev=val
  if not active.any():break
 return b,v,obj(g,b,v,cov,noise,w)
def bank(hd,C,book,cov,noise,w=None,criterion='function'):
 B,K,N=hd.shape;L=len(book)
 w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
 g=hd[:,None]+np.einsum('bknm,lm->blkn',C,book)
 if criterion=='function':
  cv=np.broadcast_to(cov[:,None],(B,L,K,N,N)).reshape(-1,K,N,N)
  bb,vv,jj=solve(g.reshape(-1,K,N),cv,noise,np.repeat(w,L,axis=0))
  ix=jj.reshape(B,L).argmin(axis=1);rr=np.arange(B)
  return book[ix],bb.reshape(B,L,K)[rr,ix],vv.reshape(B,L,N)[rr,ix],jj.reshape(B,L)[rr,ix]
 score=metric(g.reshape(-1,K,N),noise,criterion).reshape(B,L)
 ix=score.argmax(axis=1);theta=book[ix];b,v,j=solve(g[np.arange(B),ix],cov,noise,w)
 return theta,b,v,j
def metric(g,noise,kind):
 if kind=='power':return np.sum(abs(g)**2,axis=(1,2))
 A=np.eye(g.shape[-1])[None]+np.einsum('bkn,bkm->bnm',g,g.conj())/noise
 return np.linalg.slogdet(A)[1]/np.log(2)
def phase_sweep(hd,C,theta,b,v,w,bits=None):
 theta=theta.copy();u=np.einsum('bknm,bn->bkm',C,v.conj())*b[:,:,None]
 residual=b* np.einsum('bkn,bn->bk',hd,v.conj())+np.einsum('bkm,bm->bk',u,theta)-w
 for m in range(theta.shape[1]):
  without=residual-u[:,:,m]*theta[:,m,None]
  z=-np.sum(u[:,:,m].conj()*without,axis=1)
  new=np.exp(1j*np.angle(z))
  if bits is not None:new=legacy.quantize_unit_modulus(new,bits)
  theta[:,m]=new;residual=without+u[:,:,m]*new[:,None]
 return theta
def ao(hd,C,cov,noise,initial,w=None,iterations=10,bits=None):
 B,K,N=hd.shape;w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
 theta=initial.copy();b,v,j=solve(effective(hd,C,theta),cov,noise,w)
 initial_j=j.copy();history=[j.mean()]
 for _ in range(iterations):
  tn=phase_sweep(hd,C,theta,b,v,w,bits)
  bn,vn,jn=solve(effective(hd,C,tn),cov,noise,w,v0=v)
  accept=jn<=j+1e-12
  theta[accept]=tn[accept];b[accept]=bn[accept];v[accept]=vn[accept];j[accept]=jn[accept]
  history.append(j.mean())
 assert np.all(j<=initial_j+1e-10)
 return theta,b,v,j,np.array(history)
def train_function(samples,rng,L=16,noise=.01):
 hd,C,cov=moments(samples);B,K,N=hd.shape;M=C.shape[-1];w=np.ones(K)/K
 initial=rng.choice(legacy.phase_alphabet(2),(L,M))
 representatives=rng.choice(B,L,replace=False)
 # Initialize on representative samples with transceiver-refitted phase
 # search; fixed-transceiver 2-bit moves can stall at high SNR.
 code=initial.copy();gr=effective(hd[representatives],C[representatives],code)
 for _ in range(2):
  for m in range(M):
   without=gr-C[representatives,:,:,m]*code[:,m,None,None]
   cand=without[:,None]+C[representatives,None,:,:,m]*legacy.phase_alphabet(2)[None,:,None,None]
   _,_,values=solve(cand.reshape(-1,K,N),np.broadcast_to(cov[representatives,None],(L,4,K,N,N)).reshape(L*4,K,N,N),noise)
   pick=values.reshape(L,4).argmin(1);code[:,m]=legacy.phase_alphabet(2)[pick]
   gr=cand[np.arange(L),pick]
 history=[]
 for step in range(6):
  previous_code=code.copy()
  previous_value=float(bank(hd,C,code,cov,noise)[-1].mean())
  g=hd[:,None]+np.einsum('bknm,lm->blkn',C,code)
  _,_,d=solve(g.reshape(-1,K,N),np.broadcast_to(cov[:,None],(B,L,K,N,N)).reshape(B*L,K,N,N),noise)
  groups=d.reshape(B,L).argmin(axis=1)
  for l in range(L):
   ids=np.where(groups==l)[0]
   if not len(ids):continue
   theta=code[l].copy()
   for _ in range(3):
    gl=hd[ids]+np.einsum('bknm,m->bkn',C[ids],theta)
    b,v,_=solve(gl,cov[ids],noise)
    u=np.einsum('bknm,bn->bkm',C[ids].conj(),v)
    direct=np.einsum('bn,bkn->bk',v.conj(),hd[ids])
    Q=np.einsum('bk,bkm,bkj->mj',abs(b)**2,u,u.conj())/len(ids)
    q=np.einsum('bk,bkm->m',b.conj()*(w-b*direct),u)/len(ids)
    # Continuous MM relaxation, followed by 2-bit projection and a
    # fixed-transceiver objective acceptance check.
    lam=float(np.linalg.eigvalsh(Q)[-1]);tn=theta.copy()
    for _ in range(30):
     z=(lam*np.eye(M)-Q)@tn+q
     new=np.exp(1j*np.angle(z))
     assert legacy.quadratic_value(new,Q,q)<=legacy.quadratic_value(tn,Q,q)+1e-9
     if np.array_equal(new,tn):break
     tn=new
    tn=legacy.quantize_unit_modulus(tn,2)
    if legacy.quadratic_value(tn,Q,q)>legacy.quadratic_value(theta,Q,q)+1e-12:tn=theta.copy()
    if np.array_equal(tn,theta):break
    theta=tn
   code[l]=theta
  value=float(bank(hd,C,code,cov,noise)[-1].mean())
  # Guard the finite-AO empirical objective; no global minimizer claim.
  if value>previous_value+1e-12:code=previous_code;value=previous_value
  history.append(value)
  print('function training',step+1,history[-1],flush=True)
 return code,history
def train_communication(samples,rng,kind,L=16,noise=.01):
 hd,C,cov=moments(samples);B,K,N=hd.shape;M=C.shape[-1]
 # Optimize 64 representative states with the named communication metric.
 ids=rng.choice(B,64,replace=False);theta=rng.choice(legacy.phase_alphabet(2),(64,M))
 for _ in range(3):
  g=effective(hd[ids],C[ids],theta)
  for m in range(M):
   without=g-C[ids,:,:,m]*theta[:,m,None,None]
   cand=without[:,None]+C[ids,None,:,:,m]*legacy.phase_alphabet(2)[None,:,None,None]
   score=metric(cand.reshape(-1,K,N),noise,kind).reshape(64,4)
   choice=score.argmax(axis=1);theta[:,m]=legacy.phase_alphabet(2)[choice]
   g=cand[np.arange(64),choice]
 pool=np.concatenate([theta,rng.choice(legacy.phase_alphabet(2),(32,M))])
 g=hd[:,None]+np.einsum('bknm,lm->blkn',C,pool)
 scores=metric(g.reshape(-1,K,N),noise,kind).reshape(B,len(pool))
 selected=[];best=np.full(B,-np.inf)
 for _ in range(L):
  val=np.maximum(best[:,None],scores).mean(axis=0);val[selected]=-np.inf
  pick=int(val.argmax());selected.append(pick);best=np.maximum(best,scores[:,pick])
 return pool[selected]
def normalize(delta,w):
 # Float32 mean/scale are the explicitly transmitted control fields.
 means=delta.mean(axis=1).astype(np.float32).astype(float)
 std=np.nextafter(np.sqrt(np.mean((delta-means[:,None])**2,axis=1)).astype(np.float32),np.float32(np.inf)).astype(float)
 std=np.maximum(std,1e-12);s0=std.max();u=(delta-means[:,None])/std[:,None]
 alpha=w*std/s0;offset=float(w@means)
 assert np.max(abs(offset+s0*(alpha@u)-w@delta))<1e-10
 return u,alpha,offset,s0
