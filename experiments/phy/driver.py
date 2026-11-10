"""Fresh training and held-out evaluation using the original numerical routines."""
import sys,shutil,json
from pathlib import Path
import run
import train_books
R=Path(__file__).resolve().parent

def default_books(seed):
    dest=R/'reference/training'/f'seed_{seed}'
    if not (dest/'L16/books.npz').exists():
        train_books.prepare(seed)
        train_books.train_job(seed,16)
        shutil.copytree(R/'training'/f'seed_{seed}',dest,dirs_exist_ok=True)
    run.train(seed,16)

def main():
    mode,seed=sys.argv[1],int(sys.argv[2]);default_books(seed)
    if mode=='books':return
    if mode=='sizes':
        for L in [2,4,8,32]:
            if not (R/f'results/seed_{seed}/L{L}/books.npz').exists():run.train(seed,L)
        run.sizes(seed);return
    if not (R/f'reference/gnn_{seed}/gnn.pt').exists():
        import retrain
        (R/'retrain_check').mkdir(exist_ok=True)
        retrain.gnn(seed)
        shutil.copytree(R/f'retrain_check/gnn_{seed}',R/f'reference/gnn_{seed}',dirs_exist_ok=True)
    if not (R/f'results/seed_{seed}/phy.npz').exists():run.phy(seed)

if __name__=='__main__':main()
