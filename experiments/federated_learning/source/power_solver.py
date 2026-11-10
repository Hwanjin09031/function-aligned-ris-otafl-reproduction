"""Original complex objective, with a feasible scale descent after every AO pair."""
import numpy as np

def solve(g,cov,noise,w=None,steps=20,v0=None):
    B,K,N=g.shape;w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
    v=np.sum(g*w[...,None],axis=1) if v0 is None else v0.copy()
    if v0 is None:v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    def obj(b,v):
        a=np.einsum('bkn,bn->bk',g,v.conj());p=np.einsum('bn,bknm,bm->bk',v.conj(),cov,v).real
        return np.sum(abs(b*a-w)**2+abs(b)**2*p,axis=1)+noise*np.sum(abs(v)**2,axis=1)
    prev=np.full(B,np.inf)
    for _ in range(steps):
        a=np.einsum('bkn,bn->bk',g,v.conj());p=np.einsum('bn,bknm,bm->bk',v.conj(),cov,v).real
        b=w*a.conj()/np.maximum(abs(a)**2+p,1e-15);b/=np.maximum(abs(b),1)
        A=noise*np.eye(N)[None]+np.einsum('bk,bkn,bkm->bnm',abs(b)**2,g,g.conj())+np.einsum('bk,bknm->bnm',abs(b)**2,cov)
        z=np.einsum('bk,bk,bkn->bn',b,w,g);v=np.linalg.solve(A,z[...,None])[...,0]
        before=obj(b,v);m=np.max(abs(b),axis=1);m=np.where(m>1e-15,m,1.)
        b/=m[:,None];v*=m[:,None]
        value=obj(b,v)
        assert np.max(value-before)<1e-10 and np.max(value-prev)<1e-10
        prev=value
    assert np.max(abs(b))<=1+1e-12
    return b,v,obj(b,v)
