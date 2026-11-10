from common import *
from pilot_study import component_ao,cn
from copy import deepcopy
import torch,json,time
PKINDS=['function','fairness','rate','ao_qDFT65']
def pcount(k):return 520 if k.startswith('ao_') else 128
def pilot_radio(delta,w,samples,books,slices,kind,z,pnoise,matrices):
    B=len(slices);alpha=np.empty((B,8));s0=np.empty(B);offset=np.empty(B);us=[]
    for l,(a,e) in enumerate(slices):
        u,alpha[l],offset[l],s0[l]=c.normalize(delta[:,a:e],w);us.append(u)
    # These true channels are used by the observation generator and evaluator.
    # Controller inputs below are only the generated pilot observations.
    hd=np.array([s.h_d_true for s in samples[:B]]);C=np.array([s.C_true for s in samples[:B]])
    q=NOISE/8
    if not kind.startswith('ao_'):
        repeats=1;book=books[kind];L=len(book)
        gt=hd[:,None]+np.einsum('bknm,lm->blkn',C,book)
        gh=gt+np.sqrt(q/repeats)*pnoise['finite'][:B]
        cv=np.broadcast_to(q/repeats*np.eye(4),(B*L,8,4,4))
        bb,vv,jj=solve(gh.reshape(-1,8,4),cv,NOISE,np.repeat(alpha,L,axis=0))
        ix=jj.reshape(B,L).argmin(1);rr=np.arange(B);theta=book[ix]
        b=bb.reshape(B,L,8)[rr,ix];v=vv.reshape(B,L,4)[rr,ix];j=jj.reshape(B,L)[rr,ix]
    else:
        A,inverse,W=matrices[kind]
        X=np.concatenate([hd[...,None],C],axis=-1)
        observed=np.einsum('sp,bknp->bkns',A,X)+np.sqrt(q)*pnoise['component'][:B]
        estimated=np.einsum('ps,bkns->bknp',inverse,observed)
        theta,b,v,j=component_ao(estimated[...,0],estimated[...,1:],books['union'],W,q,alpha)
    assert np.max(abs(theta-c.legacy.quantize_unit_modulus(theta,2)))<1e-10
    out=np.empty(D);actual=0.;maxpower=0.
    for l,(a,e) in enumerate(slices):
        u=us[l];g=hd[l]+np.einsum('knm,m->kn',C[l],theta[l]);coeff=b[l]*(g@v[l].conj())
        noisevar=NOISE/2*np.vdot(v[l],v[l]).real
        out[a:e]=offset[l]+s0[l]*(coeff.real@u+np.sqrt(noisevar)*z[a:e])
        R=u@u.T/(e-a);residual=coeff.real-alpha[l]
        actual+=(e-a)*s0[l]**2*(residual@R@residual+noisevar)
        maxpower=max(maxpower,float(np.max(abs(b[l])**2*np.mean(u*u,axis=1))))
    den=max(np.mean((w@delta)**2),1e-30)
    assert maxpower<=1+1e-10 and np.isfinite(out).all()
    return out,{'update_nmse':float(np.mean((out-w@delta)**2)/den),'expected_update_nmse':float(actual/D/den),
      'max_tx_power':maxpower,'surrogate_mse_mean':float(j.mean()),'update_input_sha256':digest(delta),'output_sha256':digest(out)}
