"""Training-size study. Fit/validation and held-out evaluation are separate commands."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, hashlib, json, sys, time, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import training as tr
PLAN = json.loads((ROOT / 'analysis_plan.json').read_text())
ENERGY = 1 / PLAN['K']

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def path_for(seed, ns):
    return ROOT / 'results' / f'{seed}_n{ns}'

def dataset(seed, stream, count):
    return tr.data(seed, stream, count, PLAN['M'], PLAN['Nr'])

def nmse(book, data):
    return -tr.score_matrix(book, *data, 'function').max(axis=1) / ENERGY

def fit(job):
    seed, ns = job
    dest = path_for(seed, ns)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'frozen.json').exists():
        return {'phase': 'fit', 'seed': seed, 'ns': ns, 'cached': True}
    started = time.perf_counter()
    full = dataset(seed, PLAN['train_stream'], max(PLAN['training_sizes']))
    train = tuple(a[:ns] for a in full)
    val = dataset(seed, PLAN['validation_stream'], PLAN['validation_states'])
    alphabet = tr.c.legacy.phase_alphabet(PLAN['bits'])
    tr.PASSES = 3  # Pool construction is fixed, independently of refinement budget.
    pool, random = tr.make_pool(full[0], full[1], seed, alphabet)
    record = {'seed': seed, 'ns': ns, 'plan_sha256': hashlib.sha256((ROOT/'analysis_plan.json').read_bytes()).hexdigest(),
              'training_digest': tr.digest(*train), 'first250_digest': tr.digest(*(a[:250] for a in full)),
              'validation_digest': tr.digest(*val), 'pool_digest': tr.digest(pool),
              'random_digest': tr.digest(random), 'methods': {}}
    arrays = {'pool': pool, 'random': random}
    for kind in PLAN['methods']:
        scores = tr.score_matrix(pool, *train, kind)
        initial = pool[tr.greedy(scores, PLAN['L'])]
        tr.PASSES = PLAN['outer_sweeps'][kind]
        checkpoints, history, seconds = tr.refine(initial, *train, kind, alphabet)
        validation = np.stack([nmse(book, val) for book in checkpoints])
        means = validation.mean(axis=1)
        selected3 = int(means[:4].argmin())
        selected10 = int(means.argmin()) if kind == 'function' else None
        arrays[f'{kind}_checkpoints'] = np.array(checkpoints)
        arrays[f'{kind}_validation_nmse'] = validation
        assert all(np.max(np.min(abs(book[..., None]-alphabet), axis=-1)) < 1e-12 for book in checkpoints)
        record['methods'][kind] = {'history': history, 'validation_nmse': means.tolist(),
                                  'selected3': selected3, 'selected10': selected10,
                                  'refinement_seconds': seconds,
                                  'checkpoint_digests': [tr.digest(book) for book in checkpoints]}
    np.savez_compressed(dest/'frozen.npz', **arrays)
    record['fit_total_seconds'] = time.perf_counter() - started
    write_json(dest/'frozen.json', record)
    return {'phase': 'fit', 'seed': seed, 'ns': ns, 'seconds': round(record['fit_total_seconds'], 1)}

def evaluate(job):
    seed, ns = job
    dest = path_for(seed, ns)
    if (dest/'test.json').exists():
        return {'phase':'test', 'seed':seed, 'ns':ns, 'cached': True}
    record = json.loads((dest/'frozen.json').read_text())
    frozen = np.load(dest/'frozen.npz')
    test = dataset(seed, PLAN['test_stream'], PLAN['test_states'])
    hd, C, cov = test
    books, selected = {}, []
    for kind in PLAN['methods']:
        book = frozen[f'{kind}_checkpoints'][record['methods'][kind]['selected3']]
        books[f'{kind}_3'] = book
        selected.append(book)
    books['function_10'] = frozen['function_checkpoints'][record['methods']['function']['selected10']]
    for index in (0, 3, 10):
        books[f'function_fixed{index}'] = frozen['function_checkpoints'][index]
    books['random'] = frozen['random']
    cache, values = {}, {}
    for name, book in books.items():
        digest = tr.digest(book)
        if digest not in cache:
            cache[digest] = nmse(book, test)
        values[name] = cache[digest]
    union = np.concatenate(selected)
    initial = tr.c.bank(hd, C, union, cov, PLAN['noise'])[0]
    values['ao'] = tr.c.ao(hd, C, cov, PLAN['noise'], initial,
                           iterations=10, bits=PLAN['bits'])[3] / ENERGY
    np.savez_compressed(dest/'test_values.npz', **values)
    output = {'seed': seed, 'ns': ns, 'test_digest': tr.digest(*test),
              'frozen_sha256': hashlib.sha256((dest/'frozen.npz').read_bytes()).hexdigest(),
              'means': {key: float(value.mean()) for key, value in values.items()}}
    write_json(dest/'test.json', output)
    return {'phase': 'test', 'seed': seed, 'ns': ns}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['fit', 'test'])
    parser.add_argument('--workers', type=int, default=7)
    args = parser.parse_args()
    jobs = [(seed, ns) for seed in PLAN['environment_seeds'] for ns in PLAN['training_sizes']]
    if args.phase == 'test':
        missing = [job for job in jobs if not (path_for(*job)/'frozen.json').exists()]
        if missing:
            raise RuntimeError(f'All fits must be frozen before test evaluation: {missing}')
    started = time.time()
    write_json(ROOT/f'execution_{args.phase}.json', {'start_unix': started, 'python': platform.python_version(),
                'numpy': np.__version__, 'workers': args.workers, 'jobs': jobs,
                'plan_sha256': hashlib.sha256((ROOT/'analysis_plan.json').read_bytes()).hexdigest(),
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    function = fit if args.phase == 'fit' else evaluate
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            value = future.result()
            print(json.dumps({'completed': i, 'total': len(jobs), **value}), flush=True)
    print(json.dumps({'phase': args.phase, 'wall_seconds': time.time()-started}), flush=True)

if __name__ == '__main__':
    main()
