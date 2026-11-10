# Function-Aligned RIS Codebook Design for Over-the-Air Federated Learning

Experiment code, input data, and plotting scripts for the manuscript.

## Install

Tested with Python 3.12 and CPU PyTorch 2.6.0. From an activated Python environment:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

## Run an experiment

```bash
python reproduce.py --figure 2
```

This trains the codebooks and GNN, evaluates generated held-out channels, and plots the new results. It does not use the saved manuscript arrays for simulation. New outputs are written to `runs/`; supplied results in `data/` remain unchanged.

| Figure | Command | Experiment |
|---|---|---|
| 1 | `python reproduce.py --figure 1` | Copy the original system schematic |
| 2 | `python reproduce.py --figure 2` | Train codebooks/GNN; SNR sweep |
| 3 | `python reproduce.py --figure 3` | Train separate books for each codebook size; evaluate |
| 4 | `python reproduce.py --figure 4` | Train codebooks/GNN; CSI uncertainty sweep |
| 5 | `python reproduce.py --figure 5` | Train books; run 30 actual-pilot FL trajectories, 100 rounds each |
| 6 | `python reproduce.py --figure 6` | Generate scalar problems; solve global brackets; evaluate bounds |
| 7 | `python reproduce.py --figure 7` | Train and evaluate RIS/receiver/phase/training-size sweeps |

Figures 2 and 4 share the same freshly trained PHY experiment. Figure 5 uses its default books. Completed stages in the same work directory are reused. To force an independent rerun, supply a new directory, e.g. `--work runs_second`. The fixed random seeds and experimental settings are in `experiments/*/plan.json` and `analysis_plan.json`. There is no automatic adjustment based on test results.

To run all figures:

```bash
python reproduce.py --figure all --workers 1
```

Training and FL trajectories are computationally expensive. The default uses one worker to limit memory use; `--workers 2` runs independent jobs concurrently. Per-job progress is saved under `runs/*/logs/`. A GPU is not required. CPU timings are hardware dependent, and retraining is not guaranteed to be bitwise identical across platforms.

## Plot the supplied results

To regenerate PHY channel samples and evaluate the exact trained models used by the manuscript, without retraining:

```bash
python reproduce.py --figure 2 --mode evaluate
python reproduce.py --figure 3 --mode evaluate
python reproduce.py --figure 4 --mode evaluate
```

This mode writes to `runs_evaluate/` and uses `checkpoints/phy/`. The fresh-training mode does not load these supplied checkpoints. Keep the two modes in separate work directories.

To only replot the stored arrays:

```bash
python reproduce.py --mode replot
```

This shorter command uses `data/` and writes to `figures/`. It does not train or simulate. Original manuscript PDFs are in `reference_figures/`; plot styling may differ slightly.

## Additional manuscript results

```bash
python reproduce.py --study gnn-training-size
python reproduce.py --study weights
python reproduce.py --study timing
```

The GNN command generates its baseline training inputs and runs the 18 matched-exposure fits (3 environments × 3 initializations × 2 training sizes). The unequal-weight command trains and evaluates all fixed configurations. The timing command measures the recorded computation protocol on the current machine; its milliseconds will differ from the paper's hardware.

## Quick execution check

```bash
python smoke_test.py
python smoke_test.py --fl
```

The first runs small channel, optimization, evaluation, and GNN-update checks. The second also runs two fresh FL rounds with a supplied smoke-test codebook. These are execution checks, not the full experiments. See `VALIDATION.md` for the checks completed on this distribution.

## Layout

- `experiments/`: original scientific routines with portable execution drivers.
- `inputs/MNIST/`: original MNIST input files used by the FL experiment.
- `inputs/smoke/`: one frozen codebook used only by the optional smoke check.
- `checkpoints/phy/`: original trained books/GNNs for channel-regeneration evaluation.
- `data/`: saved manuscript results for quick plotting and comparison.
- `runs/`: new training checkpoints, evaluation results, logs, and figures (created on execution).

PHY statistics first average states within each of ten environments. FL trajectories are averaged within environments. The GNN training-size comparison averages three initializations within each of three environments. Timing repeats are technical measurements.

Public release and the original-code license are pending author approval. MNIST is third-party data; see `THIRD_PARTY.md`.
