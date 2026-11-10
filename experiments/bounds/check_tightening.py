"""One closed-form transmit update, optionally followed by one receive update.

Exploratory follow-up on ALL 60 existing scalar states and four codebook sizes.
No fitted coefficient, new channel selection, or iterative solve at a codeword.
The exact scalar codeword minima are used only as evaluation ground truth.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import csv,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
from bound_experiment import at_phase,CV,NOISE

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def scalar_checks():
    inputs=json.loads((R/'source/bound_inputs.json').read_text())
    refs=list(csv.DictReader((R/'source/global_search.csv').open()))
    rows=[];word_rows=[]
    for x,r in zip(inputs,refs):
        assert (x['regime'],x['index'])==(r['regime'],int(r['index']))
        d=np.array(x['d_real'])+1j*np.array(x['d_imag'])
        h=np.array(x['h_real'])+1j*np.array(x['h_imag']);w=np.array(x['weights'])
        phi=float(r['phi']);lb=float(r['lower']);ub=float(r['upper'])
        kappa=float(r['kappa']);res=float(r['stationarity_residual'])
        _,tt,bb=at_phase(phi,d,h,w);t=tt[0];b=bb[0]
        for L in [2,4,8,16]:
            angles=np.arange(L)*2*np.pi/L;g=d+np.exp(1j*angles[:,None])*h
            def J(bv,v):
                v=np.broadcast_to(v,(L,));bv=np.broadcast_to(bv,(L,3))
                return np.sum(abs(bv*v.conj()[:,None]*g-w)**2+CV*abs(bv*v[:,None])**2,axis=1)+NOISE*abs(v)**2
            def receive(bv):
                rr=np.sum(abs(bv)**2*(abs(g)**2+CV),axis=1)+NOISE
                return np.sum(bv*w*g,axis=1)/rr,rr
            a=t*g;den=abs(a)**2+CV*t*t;z=w*a.conj()/den
            b1=z/np.maximum(1,abs(z));v1,rr=receive(b1)
            bf=np.broadcast_to(b,(L,3));vr,_=receive(bf)
            jp=J(abs(b)*np.exp(-1j*np.angle(g)),t)
            jf=J(b,t);jt=J(b1,t);jtr=J(b1,v1);jr=J(bf,vr)
            gain_t=np.sum(den*(abs(b-z)**2-abs(b1-z)**2),axis=1)
            gain_r=rr*abs(t-v1)**2
            assert np.max(abs(gain_t-(jf-jt)))<1e-12
            assert np.max(abs(gain_r-(jt-jtr)))<1e-12
            assert np.min(gain_t)>-1e-12 and np.min(gain_r)>-1e-12
            exact=at_phase(angles,d,h,w)[0];jbook=exact.min()
            assert np.min(jtr-exact)>-1e-12
            dist=abs(np.exp(1j*angles)-np.exp(1j*phi))**2
            old_each=kappa*dist+2*res*np.sqrt(dist)+ub-lb
            assert np.min(old_each-(jf-lb))>-1e-12
            entry={'regime':x['regime'],'index':x['index'],'L':L,
                'gap_lower':float(jbook-ub),'gap_upper':float(jbook-lb),'global_bracket':ub-lb,
                'old_raw':float((kappa*dist).min()),'old_upper':float(old_each.min()),
                'fixed_upper':float(jf.min()-lb),'phase_upper':float(jp.min()-lb),
                'receive_only_upper':float(jr.min()-lb),'tx_upper':float(jt.min()-lb),
                'txrx_upper':float(jtr.min()-lb),
                'tx_corrected_norm_upper':float(np.min(old_each-gain_t)),
                'txrx_corrected_norm_upper':float(np.min(old_each-gain_t-gain_r))}
            assert entry['gap_lower']>0
            assert entry['gap_upper']<=entry['txrx_upper']+1e-12
            assert entry['txrx_upper']<=entry['tx_upper']+1e-12
            assert entry['tx_upper']<=entry['fixed_upper']+1e-12
            assert entry['fixed_upper']<=entry['old_upper']+1e-12
            rows.append(entry)
            for i in range(L):
                word_rows.append({'regime':x['regime'],'index':x['index'],'L':L,'word':i,
                    'J_fixed':float(jf[i]),'J_tx':float(jt[i]),'J_txrx':float(jtr[i]),
                    'J_global_inner':float(exact[i]),'gain_tx':float(gain_t[i]),'gain_rx':float(gain_r[i]),
                    'old_upper':float(old_each[i]),'distance_squared':float(dist[i])})
    write_csv(R/'bounds.csv',rows);write_csv(R/'per_codeword.csv',word_rows)
    return rows

def general_algebra_audit():
    """Check complex formulas with Nr=4, M=64 and unequal power caps.
    References are arbitrary feasible pairs, NOT claimed to be global optima.
    This audits algebra/descent only, not 64-element ideal-gap tightness.
    """
    rng=np.random.default_rng(18008);K,Nr,M=8,4,64
    errors={'quadratic_identity':0.,'tx_gain_identity':0.,'rx_gain_identity':0.,
        'descent_violation':0.,'power_violation':0.,'norm_envelope_violation':0.}
    def cn(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
    for trial in range(30):
        d=cn((K,Nr));H=cn((K,Nr,M))/np.sqrt(M)
        zc=cn((K,Nr,Nr));C=.02*np.einsum('kir,kjr->kij',zc,zc.conj())/Nr
        w=rng.dirichlet(np.ones(K));power=rng.uniform(.2,2,K);noise=.03
        theta=np.exp(1j*rng.uniform(-np.pi,np.pi,M));v=cn(Nr);v/=np.linalg.norm(v)
        b=np.sqrt(power)*rng.uniform(.1,1,K)*np.exp(1j*rng.uniform(-np.pi,np.pi,K))
        def J(G,bt,vt):
            aa=G@vt.conj();cc=np.einsum('i,kij,j->k',vt.conj(),C,vt).real
            return float(np.sum(abs(bt*aa-w)**2+cc*abs(bt)**2)+noise*np.vdot(vt,vt).real)
        G0=d+np.einsum('kim,m->ki',H,theta);j0=J(G0,b,v)
        u=np.einsum('kim,i->km',H.conj(),v);ds=d@v.conj()
        Q=np.einsum('k,ki,kj->ij',abs(b)**2,u,u.conj())
        q=np.sum((b.conj()*(w-b*ds))[:,None]*u,axis=0)
        rad=Q@theta-q;alpha=(theta.conj()*rad).real
        A=Q-np.diag(alpha);e=rad-alpha*theta
        kappa=np.max(abs(np.linalg.eigvalsh(A)))
        for scale in [.01,.1,.5,1.]:
            word=theta*np.exp(1j*scale*rng.normal(size=M));delta=word-theta
            G=d+np.einsum('kim,m->ki',H,word);jf=J(G,b,v)
            aa=G@v.conj();cc=np.einsum('i,kij,j->k',v.conj(),C,v).real
            den=abs(aa)**2+cc;z=w*aa.conj()/den
            b1=z*np.minimum(1,np.sqrt(power)/np.maximum(abs(z),1e-300))
            jt=J(G,b1,v);gt=np.sum(den*(abs(b-z)**2-abs(b1-z)**2))
            RR=np.einsum('k,ki,kj->ij',abs(b1)**2,G,G.conj())+np.einsum('k,kij->ij',abs(b1)**2,C)+noise*np.eye(Nr)
            p=np.sum((b1*w)[:,None]*G,axis=0);v1=np.linalg.solve(RR,p)
            jtr=J(G,b1,v1);gr=np.vdot(v-v1,RR@(v-v1)).real
            quad=np.vdot(delta,A@delta).real+2*np.vdot(delta,e).real
            envelope=kappa*np.vdot(delta,delta).real+2*np.linalg.norm(e)*np.linalg.norm(delta)
            errors['quadratic_identity']=max(errors['quadratic_identity'],abs(jf-j0-quad))
            errors['tx_gain_identity']=max(errors['tx_gain_identity'],abs(jf-jt-gt))
            errors['rx_gain_identity']=max(errors['rx_gain_identity'],abs(jt-jtr-gr))
            errors['descent_violation']=max(errors['descent_violation'],jt-jf,jtr-jt)
            errors['power_violation']=max(errors['power_violation'],float(np.max(abs(b1)**2-power)))
            errors['norm_envelope_violation']=max(errors['norm_envelope_violation'],jtr-j0-(envelope-gt-gr))
    assert max(errors.values())<1e-10
    return {'cases':120,'K':K,'Nr':Nr,'M':M,'purpose':'complex algebra and descent only; arbitrary feasible references',**errors}

def summary(rows):
    out={'scope':'exploratory tightening on all existing scalar cases; no fitted constants',
         'states':60,'checks':len(rows),'global_tolerance':1e-8,'methods':{},'by_L':{}}
    for key in ['old_upper','fixed_upper','phase_upper','receive_only_upper','tx_upper','txrx_upper']:
        upper_ratio=np.array([r[key]/r['gap_lower'] for r in rows])
        lower_ratio=np.array([r[key]/r['gap_upper'] for r in rows])
        out['methods'][key]={'median_ratio_upper':float(np.median(upper_ratio)),
            'median_ratio_lower':float(np.median(lower_ratio)),'max_ratio_upper':float(upper_ratio.max()),
            'mean_upper':float(np.mean([r[key] for r in rows])),
            'min_slack_to_gap_upper':float(min(r[key]-r['gap_upper'] for r in rows))}
    for L in [2,4,8,16]:
        rr=[r for r in rows if r['L']==L]
        out['by_L'][L]={'gap_mean':float(np.mean([r['gap_lower'] for r in rr])),
            **{k+'_mean':float(np.mean([r[k] for r in rr])) for k in ['old_upper','tx_upper','txrx_upper']},
            **{k+'_median_ratio':float(np.median([r[k]/r['gap_lower'] for r in rr])) for k in ['old_upper','tx_upper','txrx_upper']}}
    out['scalar_fixed_vs_original_max_difference']=float(max(abs(r['fixed_upper']-r['old_upper']) for r in rows))
    out['general_algebra']=general_algebra_audit()
    (R/'summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':summary(scalar_checks())
