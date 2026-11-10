"""Discrete generalized Lloyd with achieved-objective transceiver refits.

All candidates use the same deterministic finite solver as online selection.
The old phase is feasible. Exact maximization over its four-point alphabet
therefore cannot reduce assigned-cluster utility. Reassignment cannot reduce
best-word utility. This proves empirical descent of the achieved surrogate,
not global transceiver or codebook optimality.
"""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
import experiment as old
import common_pool_control as cp
c=old.c
ALPHABET=old.ALPHABET

def score_matrix(book,hd,C,cov,w,kind):
    B,K,N=hd.shape;out=np.empty((B,len(book)))
    for start in range(0,len(book),8):
        sub=book[start:start+8];ell=len(sub)
        g=hd[:,None]+np.einsum('bknm,lm->blkn',C,sub)
        ww=np.repeat(w,ell,axis=0)
        if kind=='function':
            cv=np.broadcast_to(cov[:,None],(B,ell,K,N,N)).reshape(-1,K,N,N)
            out[:,start:start+ell]=-c.solve(g.reshape(-1,K,N),cv,.01,ww)[-1].reshape(B,ell)
        else:out[:,start:start+ell]=old.utility(g.reshape(-1,K,N),ww,kind).reshape(B,ell)
    return out

def greedy(scores,L):
    chosen=[];best=np.full(len(scores),-np.inf)
    for _ in range(L):
        utility=np.maximum(best[:,None],scores).mean(0);utility[chosen]=-np.inf
        index=int(utility.argmax());chosen.append(index);best=np.maximum(best,scores[:,index])
    return np.array(chosen)

def refine(initial,hd,C,cov,w,kind,passes=3,callback=None):
    started=time.time();B,K,N=hd.shape;L,M=initial.shape;book=initial.copy()
    scores=score_matrix(book,hd,C,cov,w,kind)
    history=[{'pass':0,'mean_best_utility':float(scores.max(1).mean()),'changed_phases':0,'accepted_coordinate_moves':0,'seconds':time.time()-started}]
    checkpoints=[book.copy()];solve_states=B*L if kind=='function' else 0
    for sweep in range(passes):
        oldbook=book.copy();groups=scores.argmax(1);counts=np.bincount(groups,minlength=L)
        g=c.effective(hd,C,book[groups]);order=range(M) if sweep%2==0 else range(M-1,-1,-1)
        moves=0;max_violation=0.;old_objective=history[-1]['mean_best_utility']
        cv=np.broadcast_to(cov[:,None],(B,4,K,N,N)).reshape(-1,K,N,N)
        ww=np.repeat(w,4,axis=0)
        for m in order:
            current=np.mod(np.rint(np.angle(book[:,m])/(np.pi/2)).astype(int),4)
            without=g-C[:,:,:,m]*book[groups,m,None,None]
            candidates=without[:,None]+C[:,None,:,:,m]*ALPHABET[None,:,None,None]
            if kind=='function':
                state_scores=-c.solve(candidates.reshape(-1,K,N),cv,.01,ww)[-1].reshape(B,4)
                solve_states+=B*4
            else:state_scores=old.utility(candidates.reshape(-1,K,N),ww,kind).reshape(B,4)
            totals=np.stack([np.bincount(groups,weights=state_scores[:,a],minlength=L) for a in range(4)],axis=1)
            pick=totals.argmax(1);gain=totals[np.arange(L),pick]-totals[np.arange(L),current]
            # Retain the incumbent on ties, including empty clusters.
            pick[(gain<=1e-12*np.maximum(counts,1))|(counts==0)]=current[(gain<=1e-12*np.maximum(counts,1))|(counts==0)]
            final_gain=totals[np.arange(L),pick]-totals[np.arange(L),current]
            assert np.min(final_gain)>=-1e-12
            max_violation=max(max_violation,float(max(0.,-np.min(final_gain))))
            moves+=int(np.count_nonzero(pick!=current))
            book[:,m]=ALPHABET[pick];g=candidates[np.arange(B),pick[groups]]
        scores_new=score_matrix(book,hd,C,cov,w,kind)
        if kind=='function':solve_states+=B*L
        value=float(scores_new.max(1).mean())
        assert value>=old_objective-1e-11,(kind,sweep,old_objective,value)
        scores=scores_new
        history.append({'pass':sweep+1,'mean_best_utility':value,
                        'changed_phases':int(np.count_nonzero(abs(book-oldbook)>1e-8)),
                        'accepted_coordinate_moves':moves,'max_coordinate_violation':max_violation,
                        'seconds':time.time()-started})
        checkpoints.append(book.copy())
        if callback:callback(kind,history[-1])
    return checkpoints,{'history':history,'training_seconds':time.time()-started,'transceiver_state_solves':solve_states,
                        'candidate_phase_evaluations':passes*M*B*4}

def validation(book,data,w):
    hd,C,cov=data[:3]
    values=-score_matrix(book,hd,C,cov,w,'function').max(1)
    return float(values.mean()/np.sum(w*w,axis=1).mean())
