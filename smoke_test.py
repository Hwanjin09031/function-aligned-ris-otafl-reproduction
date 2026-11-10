"""Small execution check; does not reproduce the full manuscript experiments."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile

ROOT=Path(__file__).resolve().parent

def kernels(name):
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    sys.path.insert(0,str(ROOT/'experiments'/name/'source'))
    import numpy as np
    import training as tr
    d=tr.data(1201,1,4,64,4)
    alphabet=tr.c.legacy.phase_alphabet(2)
    book=np.random.default_rng(10).choice(alphabet,(2,64))
    old=tr.PASSES;tr.PASSES=1
    cp,history,_=tr.refine(book,*d,'function',alphabet);tr.PASSES=old
    values=-tr.score_matrix(cp[-1],*d,'function').max(axis=1)*8
    assert values.shape==(4,) and np.isfinite(values).all()
    if name=='phy':
        import torch
        from network import RISNet,features
        torch.set_num_threads(1);torch.manual_seed(1)
        x=torch.from_numpy(features(d[0],d[1]));net=RISNet(520);opt=torch.optim.Adam(net.parameters(),lr=.001)
        loss=net(x).square().mean();loss.backward();opt.step();net.eval()
        with torch.no_grad():
            a=net(x);b=net(x[:,torch.arange(7,-1,-1)])
        assert a.shape==(4,64,4) and torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    print(name,'channel / optimization / evaluation PASS')

def main():
    a=argparse.ArgumentParser();a.add_argument('--kernel');a.add_argument('--fl',action='store_true');x=a.parse_args()
    if x.kernel:kernels(x.kernel);return
    for name in ['phy','ris_size','receive_array','phase_resolution','training_size']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--kernel',name],check=True)
    if x.fl:
        with tempfile.TemporaryDirectory(prefix='ris_fl_smoke_') as tmp:
            d=Path(tmp)/'fl';shutil.copytree(ROOT/'experiments/federated_learning',d)
            shutil.copytree(ROOT/'inputs/MNIST',d/'data/MNIST')
            (d/'books/721').mkdir(parents=True)
            shutil.copy2(ROOT/'inputs/smoke/books_721.npz',d/'books/721/books.npz')
            plan=json.loads((d/'plan.json').read_text());plan['books_sha256']={'721':hashlib.sha256((d/'books/721/books.npz').read_bytes()).hexdigest()}
            (d/'plan.json').write_text(json.dumps(plan,indent=2))
            subprocess.run([sys.executable,'run.py','721','971','2'],cwd=d,check=True)
            assert json.loads((d/'pilot_runs/971/complete.json').read_text())['rounds']==2
        print('FL: two fresh rounds using a supplied smoke codebook PASS')
    print('Smoke checks complete; full training is a separate command.')

if __name__=='__main__':main()
