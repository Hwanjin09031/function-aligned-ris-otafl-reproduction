"""Run manuscript experiments and plot their newly generated results."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def command(directory,*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=directory,check=True)

def parallel(directory,tasks,workers):
    logdir=directory/'logs';logdir.mkdir(exist_ok=True)
    def one(task):
        name='_'.join(map(str,task)).replace('.py','')
        with (logdir/(name+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,*map(str,task)],cwd=directory,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'Failed: {name}; see {logdir / (name+".log")}')
        print('Completed:',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,tasks))

def stage(work,name):
    src=ROOT/'experiments'/name;dest=work/name
    if not dest.exists():
        shutil.copytree(src,dest)
        for n in ['results','logs','runs','data']: (dest/n).mkdir(exist_ok=True)
        (dest/'SOURCE_VERSION').write_text(source_version())
    elif (dest/'SOURCE_VERSION').read_text()!=source_version():
        raise RuntimeError('Experiment code changed. Choose a new --work directory.')
    return dest

def source_version():
    h=hashlib.sha256()
    for p in sorted((ROOT/'experiments').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()

def phy(work,mode,workers,seeds=range(721,731)):
    d=stage(work,'phy');parallel(d,[('driver.py',mode,s) for s in seeds],workers);return d

def export_phy(d,out):
    import numpy as np
    (out/'phy').mkdir(exist_ok=True)
    for seed in range(721,731):
        arrays={}
        for name in ['phy.npz','sizes.npz']:
            p=d/f'results/seed_{seed}'/name
            if p.exists():
                with np.load(p) as z:arrays.update({k:z[k] for k in z.files if k.endswith('|conditional_nmse')})
        if arrays:np.savez_compressed(out/f'phy/{seed}.npz',**arrays)

def fl_setup(work,phy_dir):
    d=stage(work,'federated_learning');plan=json.loads((d/'plan.json').read_text())
    plan['books_sha256']={}
    for seed in range(721,731):
        dest=d/f'books/{seed}';dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(phy_dir/f'reference/training/seed_{seed}/L16/books.npz',dest/'books.npz')
        plan['books_sha256'][str(seed)]=sha(dest/'books.npz')
    shutil.copytree(ROOT/'inputs/MNIST',d/'data/MNIST',dirs_exist_ok=True)
    plan['data_sha256']={p.name:sha(p) for p in (d/'data/MNIST/raw').iterdir()}
    (d/'plan.json').write_text(json.dumps(plan,indent=2));return d,plan

def export_fl(d,plan,out):
    import numpy as np
    import pandas as pd
    h=pd.concat([pd.read_csv(d/f'pilot_runs/{s}/history.csv') for e,s in plan['pairs']],ignore_index=True)
    h[['environment_seed','trajectory_seed','tau','method','round','accuracy']].to_csv(out/'fl_rounds.csv',index=False)
    rows=[]
    for (e,s,tau,method),g in h[h.method!='ideal'].groupby(['environment_seed','trajectory_seed','tau','method']):
        for target in [.7,.75,.8]:
            hit=g[(g.accuracy>=target)&(g['round']>0)].sort_values('round')
            rows.append(dict(environment_seed=e,trajectory_seed=s,tau=tau,method=method,threshold=target,rule='first',reached=not hit.empty,channel_uses=float(hit.iloc[0].cumulative_channel_uses) if not hit.empty else np.nan))
    pd.DataFrame(rows).to_csv(out/'fl_target_cost.csv',index=False)

def sensitivity(work,out,workers):
    for name,folder,axis,values in [('ris_size','ris','m',[32,64,128]),('receive_array','receive','n',[2,4,8]),('phase_resolution','phase','b',[1,2,3]),('training_size','size','n',[250,500,1000,2000])]:
        d=stage(work,name);command(d,'study.py','fit','--workers',workers)
        # All fits are completed before any held-out evaluation.
        if name!='training_size' and not (d/'pretest_frozen_manifest.json').exists():command(d,'study.py','freeze')
        command(d,'study.py','test','--workers',workers)
        (out/folder).mkdir(exist_ok=True)
        for seed in range(1201,1211):
            for v in values:shutil.copy2(d/f'results/{seed}_{axis}{v}/test_values.npz',out/folder/f'{seed}_{axis}{v}.npz')

def supplementary(work,out,study,workers):
    if study=='gnn-training-size':
        dphy=phy(work,'phy',workers,seeds=[721,722,723]);d=stage(work,'gnn_training_size')
        shutil.copytree(dphy/'reference',d/'reference',dirs_exist_ok=True)
        parallel(d,[('experiment.py','prepare',e) for e in [721,722,723]],workers)
        jobs=[]
        for e in [721,722,723]:
            for off in [0,100000,200000]:
                jobs.extend([('experiment.py','control',e,800,e+off),('experiment.py','train',e,3200,e+off)])
        parallel(d,jobs,workers);command(d,'analyze_control.py')
        shutil.copy2(d/'matched_exposure_runs.csv',out/'gnn_training_size.csv')
    elif study=='weights':
        d=stage(work,'weights');p=json.loads((d/'plan.json').read_text())
        parallel(d,[('weights_experiment.py',s,r) for r in p['regimes'] for s in p['environments']],workers)
        import pandas as pd
        rows=[row for p in (d/'runs').glob('*/report.json') for row in json.loads(p.read_text())['rows']]
        pd.DataFrame(rows).to_csv(out/'weight_results.csv',index=False)
    else:
        # The timing protocol uses the Ns=1000 frozen model in one environment.
        base=stage(work,'training_size')
        command(base,'-c','import study; study.fit((1201,1000)); study.evaluate((1201,1000))')
        d=stage(work,'timing');(d/'reference').mkdir(exist_ok=True)
        for name in ['frozen.json','frozen.npz','test.json','test_values.npz']:shutil.copy2(base/'results/1201_n1000'/name,d/'reference'/name)
        p=json.loads((d/'analysis_plan.json').read_text())
        p['source_sha256']={x.name:sha(x) for x in (d/'source').glob('*.py')}
        p['reference_sha256']={x.name:sha(x) for x in (d/'reference').iterdir()}
        (d/'analysis_plan.json').write_text(json.dumps(p,indent=2));command(d,'benchmark.py')
        shutil.copytree(d/'results',out/'timing',dirs_exist_ok=True)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--figure',choices=['all']+[str(i) for i in range(1,8)],default='all')
    a.add_argument('--study',choices=['gnn-training-size','weights','timing'])
    a.add_argument('--mode',choices=['fresh','evaluate','replot'],default='fresh')
    a.add_argument('--workers',type=int,default=1)
    a.add_argument('--work',type=Path)
    x=a.parse_args()
    if x.workers<1:a.error('--workers must be positive')
    if x.mode=='replot':
        if x.study:a.error('--study requires fresh mode')
        command(ROOT,'plot_figures.py','--figure',x.figure);return
    if x.mode=='evaluate' and (x.study or x.figure not in ['2','3','4']):
        a.error('evaluate mode supports --figure 2, 3, or 4; it uses supplied PHY checkpoints on regenerated channels')
    work=(x.work or ROOT/('runs_evaluate' if x.mode=='evaluate' else 'runs')).resolve()
    work.mkdir(parents=True,exist_ok=True)
    marker=work/'EXECUTION_MODE.json'
    if marker.exists() and json.loads(marker.read_text())['mode']!=x.mode:
        a.error('Choose a separate --work directory for fresh training and checkpoint evaluation')
    marker.write_text(json.dumps({'mode':x.mode},indent=2))
    out=work/'plot_data';out.mkdir(exist_ok=True)
    if x.mode=='evaluate':
        d=stage(work,'phy')
        shutil.copytree(ROOT/'checkpoints/phy',d,dirs_exist_ok=True)
    if x.study:supplementary(work,out,x.study,x.workers);print('Results:',out);return
    numbers=list(range(1,8)) if x.figure=='all' else [int(x.figure)]
    dphy=None
    if any(n in numbers for n in [2,4]):dphy=phy(work,'phy',x.workers)
    if 3 in numbers:dphy=phy(work,'sizes',x.workers)
    if dphy:export_phy(dphy,out)
    if 5 in numbers:
        dphy=phy(work,'books',x.workers);d,p=fl_setup(work,dphy)
        parallel(d,[('run.py',e,s) for e,s in p['pairs']],x.workers);export_fl(d,p,out)
    if 6 in numbers:
        d=stage(work,'bounds');command(d,'source/bound_experiment.py');command(d,'check_tightening.py');shutil.copy2(d/'bounds.csv',out/'bounds.csv')
    if 7 in numbers:sensitivity(work,out,x.workers)
    for n in numbers:command(ROOT,'plot_figures.py','--figure',n,'--data-dir',out,'--output-dir',work/'figures')
    print('Fresh experiment figures:',work/'figures')

if __name__=='__main__':main()
