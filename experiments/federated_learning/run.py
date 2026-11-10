"""Actual-pilot closed-loop FL. Frozen plan; no tuning on resulting accuracy."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from common import *
from pilot_radio import pilot_radio, PKINDS, pcount
from pilot_study import cn
from copy import deepcopy
import torch,json,time,subprocess,concurrent.futures,platform

PLAN=json.loads((ROOT/'plan.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(environment_seed,trajectory_seed,rounds=100):
    configure();start=time.time();fl_seed=trajectory_seed+5380;torch.manual_seed(fl_seed)
    assert torch.__version__=='2.6.0+cpu',torch.__version__
    assert sha(ROOT/'books'/str(environment_seed)/'books.npz')==PLAN['books_sha256'][str(environment_seed)]
    books=load_books(environment_seed)
    X,Y,TX,TY=base.dataset();ids=base.partition(Y.numpy(),fl_seed+11)
    initial=base.CNN();assert len(base.vec(initial))==D
    ideal=deepcopy(initial);models={(t,k):deepcopy(initial) for t in PLAN['tau'] for k in PKINDS}
    env=c.legacy.create_clustered_environment(np.random.default_rng(environment_seed),8,4,64,4,kappa_G=10.)
    i=np.arange(65);dft=np.exp(2j*np.pi*np.outer(i,i)/65)
    A=c.legacy.quantize_unit_modulus(dft,2);inverse=np.linalg.pinv(A)
    assert np.linalg.matrix_rank(A)==65
    matrices={'ao_qDFT65':(A,inverse,(inverse@inverse.conj().T).T)}
    outdir=ROOT/'pilot_runs'/str(trajectory_seed);outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'complete.json').exists():
        old=json.loads((outdir/'config.json').read_text())
        
        assert old['plan_sha256']==sha(ROOT/'plan.json'), 'Plan changed: choose a new work directory.'
        assert old['rounds']==rounds, 'Round count changed: choose a new work directory.'
        print('CACHED COMPLETE',trajectory_seed,flush=True);return
    config={'environment_seed':environment_seed,'trajectory_seed':trajectory_seed,'FL_seed':fl_seed,'rounds':rounds,
        'initial_sha256':digest(base.vec(initial).numpy()),'plan_sha256':sha(ROOT/'plan.json'),
        'qDFT_condition_number':float(np.linalg.cond(A)),
        'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
        'books_sha256':sha(ROOT/'books'/str(environment_seed)/'books.npz')}
    (outdir/'config.json').write_text(json.dumps(config,indent=2))
    np.savez(outdir/'client_partition.npz',**{f'client{k}':ix for k,ix in enumerate(ids)})
    torch.save(initial.state_dict(),outdir/'initial_model.pt')
    history=[];trace=[];probes=[];w=np.ones(8)/8;loss0,acc0=base.evaluate(initial,TX,TY)
    for tau in PLAN['tau']:
        for kind in PKINDS+['ideal']:
            history.append({'environment_seed':environment_seed,'trajectory_seed':trajectory_seed,'tau':tau,'method':kind,
                'round':0,'accuracy':acc0,'test_loss':loss0,'cumulative_channel_uses':0 if kind!='ideal' else float('nan'),
                'parameter_sha256':config['initial_sha256']})
    first_round=1
    checkpoint=outdir/'resume.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False)
        assert state['plan_sha256']==config['plan_sha256']
        for key,m in models.items():m.load_state_dict(state['models'][f'{key[0]}_{key[1]}'])
        ideal.load_state_dict(state['ideal']);history=state['history'];trace=state['trace'];probes=state['probes']
        first_round=state['round']+1
    for rnd in range(first_round,rounds+1):
        rng=np.random.default_rng(fl_seed*10000+rnd)
        plans=[[torch.tensor(rng.choice(ix,64,replace=True)) for _ in range(5)] for ix in ids]
        Bmax=len(layout(2000,520)[0])
        samples=c.cc.generate_clustered_samples(np.random.default_rng(np.random.SeedSequence([trajectory_seed,rnd,991])),
            env,Bmax,8,4,64,kappa_hr=10**.3,csi_error_ratio=.05)
        z=np.random.default_rng(np.random.SeedSequence([trajectory_seed,rnd,777])).normal(size=D)
        rngp=np.random.default_rng(np.random.SeedSequence([trajectory_seed,rnd,99173]))
        pnoise={'finite':cn(rngp,(Bmax,16,8,4),1.),'component':cn(rngp,(Bmax,8,4,65),1.)}
        accuracies={}
        for (tau,kind),model in models.items():
            delta=base.updates(model,X,Y,plans);slices,cost=layout(tau,pcount(kind))
            out,diag=pilot_radio(delta,w,samples,books,slices,kind,z,pnoise,matrices)
            new=base.vec(model)+torch.tensor(out,dtype=torch.float32)
            assert torch.isfinite(new).all()
            base.setvec(model,new)
            loss,acc=base.evaluate(model,TX,TY) if rnd%5==0 else (float('nan'),float('nan'))
            accuracies[f'{tau}_{kind}']=acc
            history.append({'environment_seed':environment_seed,'trajectory_seed':trajectory_seed,'tau':tau,'method':kind,
                'round':rnd,'accuracy':acc,'test_loss':loss,'cumulative_channel_uses':cost*rnd,
                'parameter_sha256':digest(base.vec(model).numpy()),**diag})
        delta=base.updates(ideal,X,Y,plans)
        # Read-only paired radio probe: these outputs never update a model.
        if rnd%5==0:
            for tau in PLAN['tau']:
                for kind in PKINDS:
                    slices,_=layout(tau,pcount(kind))
                    _,diag=pilot_radio(delta,w,samples,books,slices,kind,z,pnoise,matrices)
                    probes.append({'environment_seed':environment_seed,'trajectory_seed':trajectory_seed,'tau':tau,'method':kind,'round':rnd,**diag})
        base.setvec(ideal,base.vec(ideal)+torch.tensor(w@delta,dtype=torch.float32))
        if rnd%5==0:
            loss,acc=base.evaluate(ideal,TX,TY)
            for tau in PLAN['tau']:
                history.append({'environment_seed':environment_seed,'trajectory_seed':trajectory_seed,'tau':tau,'method':'ideal',
                    'round':rnd,'accuracy':acc,'test_loss':loss,'cumulative_channel_uses':float('nan'),
                    'parameter_sha256':digest(base.vec(ideal).numpy())})
            print(json.dumps({'trajectory':trajectory_seed,'round':rnd,'seconds':round(time.time()-start),'accuracies':accuracies}),flush=True)
        trace.append({'round':rnd,'noise_sha256':digest(z),'minibatch_sha256':digest(np.stack([[i.numpy() for i in p] for p in plans])),
            'first_true_channel_sha256':digest(samples[0].C_true),'finite_pilot_noise_sha256':digest(pnoise['finite']),
            'component_pilot_noise_sha256':digest(pnoise['component'])})
        if rnd%5==0 or rnd==rounds:
            save_csv(outdir/'history.csv',history);save_csv(outdir/'random_trace.csv',trace)
            if probes:save_csv(outdir/'common_update_probes.csv',probes)
        if rnd%20==0:
            state={'plan_sha256':config['plan_sha256'],'round':rnd,'models':{f'{t}_{k}':m.state_dict() for (t,k),m in models.items()},
                   'ideal':ideal.state_dict(),'history':history,'trace':trace,'probes':probes}
            torch.save(state,outdir/'resume.tmp');(outdir/'resume.tmp').replace(checkpoint)
    torch.save({**{f'{t}_{k}':m.state_dict() for (t,k),m in models.items()},'ideal':ideal.state_dict()},outdir/'final_models.pt')
    (outdir/'complete.json').write_text(json.dumps({'status':'complete','rounds':rounds,'seconds':time.time()-start,'history_rows':len(history),'probe_rows':len(probes)}))

if __name__=='__main__':run(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 100)
