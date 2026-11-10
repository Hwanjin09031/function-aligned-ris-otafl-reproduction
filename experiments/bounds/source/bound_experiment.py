"""Scalar global benchmark with an exact inner solve and phase interval bounds."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,csv
import numpy as np
from scipy.optimize import brentq,minimize_scalar
R=Path(__file__).resolve().parent
PLAN=json.loads((R/'plan.json').read_text())['bound']
NOISE=PLAN['noise'];CV=PLAN['covariance']
def magnitude_solve(a,w,cv=CV,noise=NOISE):
    a=np.atleast_2d(a);w=np.broadcast_to(w,a.shape);K=a.shape[1]
    A=a*a+cv;B=w*a;threshold=B/A
    ix=np.argsort(threshold,axis=1);aa=np.take_along_axis(A,ix,1);bb=np.take_along_axis(B,ix,1);th=np.take_along_axis(threshold,ix,1)
    sa=np.concatenate([np.cumsum(aa[:,::-1],axis=1)[:,::-1],np.zeros((len(a),1))],axis=1)
    sb=np.concatenate([np.cumsum(bb[:,::-1],axis=1)[:,::-1],np.zeros((len(a),1))],axis=1)
    lo=np.concatenate([np.zeros((len(a),1)),th],axis=1);hi=np.concatenate([th,np.full((len(a),1),np.inf)],axis=1)
    roots=sb/(noise+sa);t=np.minimum(np.maximum(roots,lo),hi)
    # Evaluate all K+1 intervals; this enumerates the exact convex scalar optimum.
    b=np.minimum(B[:,None,:]/np.maximum(t[:,:,None]*A[:,None,:],1e-300),1.)
    val=np.sum((t[:,:,None]*b*a[:,None,:]-w[:,None,:])**2+cv*(t[:,:,None]*b)**2,axis=-1)+noise*t*t
    j=val.argmin(1);rr=np.arange(len(a))
    return val[rr,j],t[rr,j],b[rr,j]
def at_phase(phi,d,h,w):
    theta=np.exp(1j*np.atleast_1d(phi));g=d[None,:]+theta[:,None]*h[None,:]
    val,t,bmag=magnitude_solve(abs(g),w)
    b=bmag*np.exp(-1j*np.angle(g))
    return val,t,b

def slope(phi,d,h,w):
    val,t,b=at_phase(phi,d,h,w);g=d+h*np.exp(1j*phi)
    return float(2*np.real(np.sum((b[0]*t[0]*g-w).conj()*b[0]*t[0]*1j*h*np.exp(1j*phi))))
def lower(lo,hi,d,h,w):
    mid=(lo+hi)/2;half=(hi-lo)/2;psi=np.angle(d.conj()*h)
    delta=np.abs(np.angle(np.exp(1j*(mid[:,None]+psi))))
    maxcos=np.cos(np.maximum(delta-half[:,None],0.))
    maxgain=np.sqrt(np.maximum(abs(d)**2+abs(h)**2+2*abs(d.conj()*h)*maxcos,0.))
    return magnitude_solve(maxgain,w)[0]
def local(phi,half,d,h,w):
    lo=phi-half;hi=phi+half
    if slope(lo,d,h,w)<0<slope(hi,d,h,w):
        x=brentq(lambda q:slope(q,d,h,w),lo,hi,xtol=1e-14)
    else:x=minimize_scalar(lambda q:at_phase(q,d,h,w)[0][0],bounds=(lo,hi),method='bounded',options={'xatol':1e-14}).x
    return x,float(at_phase(x,d,h,w)[0][0])
def global_phase(d,h,w,tol=1e-8):
    grid=np.arange(256)*2*np.pi/256;v=at_phase(grid,d,h,w)[0];i=int(v.argmin())
    theta,upper=local(grid[i],2*np.pi/256,d,h,w)
    lo=np.arange(64)*2*np.pi/64;hi=lo+2*np.pi/64
    pruned=np.inf;nodes=0;level=0;lbval=0.
    while len(lo):
        lbs=lower(lo,hi,d,h,w);nodes+=len(lo);mid=(lo+hi)/2
        vals=at_phase(mid,d,h,w)[0];j=int(vals.argmin())
        if vals[j]<upper:
            candidate,u=local(mid[j],(hi[j]-lo[j])/2,d,h,w)
            if u<upper:theta,upper=candidate,u
        drop=lbs>=upper-tol
        if np.any(drop):pruned=min(pruned,float(lbs[drop].min()))
        lo,hi=lo[~drop],hi[~drop]
        if not len(lo):break
        if level>=PLAN['max_levels'] or len(lo)>PLAN['max_active']:raise RuntimeError('Global bracket did not close')
        mid=(lo+hi)/2;lo,hi=np.concatenate([lo,mid]),np.concatenate([mid,hi]);level+=1
    lower_bound=min(pruned,upper)
    assert upper-lower_bound<=tol+1e-12
    return theta%(2*np.pi),lower_bound,upper,nodes,level

def run():
    rng=np.random.default_rng(PLAN['seed']);records=[];sens=[];global_rows=[];inputs=[];started=time.time()
    for regime in PLAN['regimes']:
      for idx in range(PLAN['states_per_regime']):
        cn=lambda shape:(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        d=.3*cn(3);h=cn(3)
        w=np.ones(3)/3 if regime=='equal' else (np.array([1,2,3])/6 if regime=='fixed_unequal' else rng.dirichlet(np.ones(3)*.5))
        inputs.append({'regime':regime,'index':idx,'d_real':d.real.tolist(),'d_imag':d.imag.tolist(),'h_real':h.real.tolist(),'h_imag':h.imag.tolist(),'weights':w.tolist()})
        phi,lb,ub,nodes,levels=global_phase(d,h,w,PLAN['global_tolerance']);th=np.exp(1j*phi)
        value,t,b=at_phase(phi,d,h,w);b=b[0];t=t[0]
        Q=float(np.sum(abs(b)**2*abs(h)**2)*t*t)
        q=np.sum(b.conj()*(w-b*t*d)*h.conj()*t)
        r=Q*th-q;alpha=float((th.conjugate()*r).real);kappa=abs(Q-alpha);res=abs(r-alpha*th)
        assert abs(value[0]-ub)<1e-12
        global_rows.append({'regime':regime,'index':idx,'phi':phi,'lower':lb,'upper':ub,'bracket':ub-lb,'nodes':nodes,'levels':levels,'kappa':kappa,'stationarity_residual':float(res)})
        for L in PLAN['L']:
            angles=np.arange(L)*2*np.pi/L;dist2=abs(np.exp(1j*angles)-th)**2
            jbook=float(at_phase(angles,d,h,w)[0].min());gap=jbook-ub
            idealbound=kappa*dist2.min()
            corrected=float(np.min(kappa*dist2+2*res*np.sqrt(dist2))+ub-lb)
            assert gap>=-1e-9 and jbook-lb<=corrected+1e-9
            records.append({'regime':regime,'index':idx,'L':L,'ideal_gap_upper':jbook-lb,'gap':gap,'distance2':float(dist2.min()),'kappa':kappa,'bound':float(idealbound),'residual_aware_bound':corrected,'violation':max(0.,gap-idealbound)})
        for offset in PLAN['fixed_offsets_rad']:
            angles=phi+np.array([-offset,offset]);dist2=2-2*np.cos(offset)
            gap=float(at_phase(angles,d,h,w)[0].min()-ub)
            sens.append({'regime':regime,'index':idx,'offset':offset,'distance2':dist2,'kappa':kappa,'gap':gap,'gap_per_distance2':gap/dist2,'bound':kappa*dist2})
        print(regime,idx,'bracket',ub-lb,'residual',res,'nodes',nodes,flush=True)
    for filename,rows in [('bound.csv',records),('sensitivity.csv',sens),('global_search.csv',global_rows)]:
        with (R/filename).open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    (R/'bound_inputs.json').write_text(json.dumps(inputs,indent=2))
    (R/'bound_status.json').write_text(json.dumps({'status':'complete','seconds':time.time()-started,'states':len(global_rows),'checks':len(records),'max_gap_violation':max(x['violation'] for x in records),'max_bracket':max(x['bracket'] for x in global_rows),'max_stationarity_residual':max(x['stationarity_residual'] for x in global_rows)},indent=2))
if __name__=='__main__':run()
