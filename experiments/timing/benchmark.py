"""CPU timing benchmark. CPU timings are measured; no performance tuning is applied."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import gc, hashlib, json, platform, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
affinity_before = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
if affinity_before:os.sched_setaffinity(0, {min(affinity_before)})
sys.path.insert(0, str(R / 'source'))
import numpy as np
import training as tr
from threadpoolctl import threadpool_info

P = json.loads((R / 'analysis_plan.json').read_text())
OUT = R / 'results'
OUT.mkdir(exist_ok=True)
def write(path, value): path.write_text(json.dumps(value, indent=2))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert not (OUT / 'summary.json').exists(), 'Use a fresh copy for a new benchmark; retain published evidence.'
for key, val in P['source_sha256'].items(): assert sha(R / 'source' / key) == val
for key, val in P['reference_sha256'].items(): assert sha(R / 'reference' / key) == val
runtime = {'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(),
    'cpu_model': platform.processor(),
    'affinity_before': affinity_before, 'affinity_during': sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],
    'thread_pools': threadpool_info(), 'plan_sha256': sha(R/'analysis_plan.json'),
    'benchmark_sha256': sha(Path(__file__)), 'load_average_at_start': os.getloadavg() if hasattr(os,'getloadavg') else None}
assert all(x['num_threads'] == 1 for x in runtime['thread_pools'])
write(OUT/'runtime.json', runtime)

reference = json.loads((R / 'reference' / 'frozen.json').read_text())
frozen = np.load(R / 'reference' / 'frozen.npz')
saved_test = np.load(R / 'reference' / 'test_values.npz')
full = tr.data(1201, 1, 2000, 64, 4)
train = tuple(a[:1000] for a in full)
val = tr.data(1201, 2, 300, 64, 4)
test = tr.data(1201, 3, 1000, 64, 4)
assert tr.digest(*train) == reference['training_digest']
assert tr.digest(*val) == reference['validation_digest']
assert tr.digest(*test) == json.loads((R/'reference'/'test.json').read_text())['test_digest']
alphabet = tr.c.legacy.phase_alphabet(2)
tr.PASSES = 3
offline, replay_checks = [], []
for repeat in range(P['offline_repeats']):
    row = {'repeat': repeat + 1, 'methods': {}}
    start = time.perf_counter_ns()
    pool, random = tr.make_pool(full[0], full[1], 1201, alphabet)
    row['pool_seconds'] = (time.perf_counter_ns() - start) / 1e9
    assert np.array_equal(pool, frozen['pool']) and np.array_equal(random, frozen['random'])
    checks = {'repeat': repeat+1, 'pool_exact': True, 'random_exact': True, 'methods': {}}
    arrays = {}
    for kind in tr.METHODS:
        start = time.perf_counter_ns()
        scores = tr.score_matrix(pool, *train, kind)
        initial = pool[tr.greedy(scores, 16)]
        greedy_end = time.perf_counter_ns()
        cp, history, _ = tr.refine(initial, *train, kind, alphabet)
        refine_end = time.perf_counter_ns()
        validation = np.stack([-tr.score_matrix(book, *val, 'function').max(axis=1)*8 for book in cp])
        selected = int(validation.mean(axis=1).argmin())
        end = time.perf_counter_ns()
        row['methods'][kind] = {'greedy_seconds': (greedy_end-start)/1e9,
            'refine_seconds': (refine_end-greedy_end)/1e9, 'validation_seconds': (end-refine_end)/1e9,
            'total_seconds': (end-start)/1e9, 'selected_checkpoint': selected,
            'moves': [x['moves'] for x in history], 'history': history}
        assert np.array_equal(np.array(cp), frozen[kind+'_checkpoints'][:4]), kind
        validation_error = float(np.max(abs(validation - frozen[kind+'_validation_nmse'][:4])))
        assert validation_error < 1e-12 and selected == reference['methods'][kind]['selected3']
        arrays[kind+'_checkpoints'] = np.array(cp)
        arrays[kind+'_validation_nmse'] = validation
        checks['methods'][kind] = {'all_four_checkpoints_exact': True,
            'validation_max_abs_difference': validation_error, 'selected_checkpoint_exact': True}
        print(json.dumps({'phase':'offline', 'repeat':repeat+1, 'method':kind,
            'seconds':row['methods'][kind]['total_seconds'], 'exact_checkpoints':True}), flush=True)
    np.savez_compressed(OUT/f'offline_replay_{repeat+1}.npz', **arrays)
    write(OUT/f'offline_replay_{repeat+1}.json', row)
    offline.append(row); replay_checks.append(checks)

books = {k: frozen[k+'_checkpoints'][reference['methods'][k]['selected3']] for k in tr.METHODS}
union = np.concatenate(list(books.values()))
assert union.shape == (64,64)
hd, C, cv = (x[:100] for x in test)
fixtures = []
for i in range(100):
    acquired = {}
    for kind, book in books.items():
        g = hd[i][None] + np.einsum('knm,lm->lkn', C[i], book)
        covariance = np.broadcast_to(cv[i], (16,8,4,4))
        acquired[kind] = (g, covariance)
    fixtures.append({'acquired': acquired, 'ao': (hd[i:i+1], C[i:i+1], cv[i:i+1])})
def online_book(args):
    g, covariance = args
    b,v,j = tr.solve(g, covariance, .01)
    ix = int(np.argmin(j))
    return b[ix],v[ix],float(j[ix])*8
def online_ao(args):
    h,c,cov = args
    initial = tr.c.bank(h,c,union,cov,.01)[0]
    return tr.c.ao(h,c,cov,.01,initial,iterations=10,bits=2)

method_names = list(tr.METHODS)+['ao']
for i in range(P['online_warmup_states']):
    for kind in tr.METHODS: online_book(fixtures[i]['acquired'][kind])
    online_ao(fixtures[i]['ao'])
online, errors = [], {k:0. for k in method_names}
gc_enabled = gc.isenabled(); gc.disable()
try:
    for rep in range(P['online_passes']):
        for i, fixture in enumerate(fixtures):
            shift = (rep+i)%len(method_names)
            order = method_names[shift:]+method_names[:shift]
            for kind in order:
                args = fixture['ao'] if kind=='ao' else fixture['acquired'][kind]
                fn = online_ao if kind=='ao' else online_book
                start = time.perf_counter_ns(); result = fn(args); end = time.perf_counter_ns()
                measured = float(result[3][0])*8 if kind=='ao' else result[2]
                expected = saved_test['ao' if kind=='ao' else kind+'_3'][i]
                errors[kind] = max(errors[kind],abs(measured-float(expected)))
                if kind=='ao':assert np.max(np.min(abs(result[0][...,None]-alphabet),axis=-1)) < 1e-12
                online.append({'pass':rep+1,'state_index':i,'method':kind,
                    'latency_ms':(end-start)/1e6,'nmse':measured})
        print(json.dumps({'phase':'online','pass':rep+1,'finished_states':100}), flush=True)
finally:
    if gc_enabled: gc.enable()
assert max(errors.values()) < 1e-12, errors
write(OUT/'online_raw.json', online)
def stats(values):
    a=np.array(values)
    return {'n':len(a),'median':float(np.median(a)), 'min':float(a.min()),
            'max':float(a.max()),'p10':float(np.percentile(a,10)), 'p90':float(np.percentile(a,90))}
summary={'offline_seconds':{'shared_pool':stats([r['pool_seconds'] for r in offline])},
         'online_ms':{},'reference_replay':replay_checks,'online_nmse_max_abs_difference':errors}
for kind in tr.METHODS:
    summary['offline_seconds'][kind]=stats([r['methods'][kind]['total_seconds'] for r in offline])
    summary['offline_seconds'][kind]['segments']={seg:stats([r['methods'][kind][seg+'_seconds'] for r in offline]) for seg in ('greedy','refine','validation')}
for kind in method_names:
    part=[r for r in online if r['method']==kind]
    summary['online_ms'][kind]=stats([r['latency_ms'] for r in part])
    summary['online_ms'][kind]['pass_medians']=[float(np.median([r['latency_ms'] for r in part if r['pass']==p])) for p in range(1,P['online_passes']+1)]
summary['ao_over_function_median_ratio']=summary['online_ms']['ao']['median']/summary['online_ms']['function']['median']
summary['scope']='One environment, one logical CPU, technical timing repeats; not an end-to-end latency or hardware-generalization claim.'
runtime['finish_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
runtime['load_average_at_end']=os.getloadavg() if hasattr(os,'getloadavg') else None
write(OUT/'runtime.json', runtime); write(OUT/'summary.json', summary)
print(json.dumps({'complete':True,'offline':{k:v['median'] for k,v in summary['offline_seconds'].items()},
                 'online':{k:v['median'] for k,v in summary['online_ms'].items()}}), flush=True)
