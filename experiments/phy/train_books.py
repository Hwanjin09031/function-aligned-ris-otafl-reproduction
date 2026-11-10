"""Frozen-protocol training and evaluation for the RIS codebook experiment."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys, json, time, hashlib, argparse, subprocess, concurrent.futures
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import training as tr
from power_solver import solve

c = tr.c
assert c.solve is solve
PLAN = json.loads((ROOT / "plan.json").read_text())
KINDS = ("function","fairness","rate","power")
SEEDS = PLAN["environment_seeds"]
LS = [16]

def digest(*xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def samples(seed, n, stream, rho=.05, K=8, N=4):
    env = c.legacy.create_clustered_environment(np.random.default_rng(seed), K, N, 64, 4, kappa_G=10.)
    return c.cc.generate_clustered_samples(np.random.default_rng(seed + stream * 10000), env, n, K, N, 64,
        kappa_hr=10**.3, csi_error_ratio=rho)

def prepare(seed):
    out = ROOT / "training" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    train = c.moments(samples(seed, 1000, 1))
    val = c.moments(samples(seed, 300, 4))
    pool, _ = tr.make_pool(*train[:2], seed, c.legacy.phase_alphabet(2))
    # Recover the 32 original random pool candidates, before deduplication.
    rng = np.random.default_rng(seed + 200)
    rng.choice(250, 64, replace=False)
    rng.choice(c.legacy.phase_alphabet(2), (64, 64))
    random_book = rng.choice(c.legacy.phase_alphabet(2), (32, 64))
    np.savez_compressed(out / "pool.npz", pool=pool, random=random_book)
    scores = {kind: tr.score_matrix(pool, *train, kind) for kind in KINDS}
    np.savez_compressed(out / "pool_scores.npz", **scores)
    write_json(out / "prepared.json", {"seed":seed, "train_hash":digest(*train), "validation_hash":digest(*val),
        "pool_hash":digest(pool), "pool_n":len(pool)})
    print("PREPARED", seed, "pool", len(pool), flush=True)

def train_job(seed, L):
    start = time.perf_counter()
    out = ROOT / "training" / f"seed_{seed}" / f"L{L}"
    out.mkdir(exist_ok=True)
    base = out.parent
    pool = np.load(base / "pool.npz")["pool"]
    scores = np.load(base / "pool_scores.npz")
    train = c.moments(samples(seed, 1000, 1))
    val = c.moments(samples(seed, 300, 4))
    prepared = json.loads((base / "prepared.json").read_text())
    assert digest(*train) == prepared["train_hash"]
    assert digest(*val) == prepared["validation_hash"]
    books, history, arrays = {}, {}, {}
    for kind in KINDS:
        ids = tr.greedy(scores[kind], L)
        checkpoints, hist, elapsed = tr.refine(pool[ids], *train, kind, c.legacy.phase_alphabet(2))
        vscore = [float(-tr.score_matrix(book, *val, "function").max(1).mean() * 8) for book in checkpoints]
        chosen = int(np.argmin(vscore))
        books[kind] = checkpoints[chosen]
        for i, book in enumerate(checkpoints):
            arrays[f"{kind}_pass{i}"] = book
        history[kind] = {"selected_pass":chosen, "validation_nmse":vscore, "training":hist,
            "greedy_indices":ids.tolist(), "refinement_seconds":elapsed, "book_hash":digest(books[kind])}
        print("TRAIN", seed, L, kind, "selected", chosen, "seconds", round(time.perf_counter()-start,1), flush=True)
    books["random_book"] = np.load(base / "pool.npz")["random"][:L]
    np.savez_compressed(out / "books.npz", **books)
    np.savez_compressed(out / "checkpoints.npz", **arrays)
    write_json(out / "frozen.json", {"seed":seed, "L":L, **prepared, "diagnostics":history,
        "elapsed_seconds":time.perf_counter()-start, "plan_sha256":hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest(),
        "books_sha256":hashlib.sha256((out / "books.npz").read_bytes()).hexdigest()})

if __name__=="__main__":
    seed=int(sys.argv[1]);prepare(seed);train_job(seed,16)
