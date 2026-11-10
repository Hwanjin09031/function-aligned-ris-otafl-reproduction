# Execution checks

Checked with Python 3.12, NumPy 2.3.5, SciPy 1.17.0, and CPU PyTorch 2.6.0.

- All seven saved-result figure commands completed.
- Original-checkpoint evaluation, PHY environment 721: all 200 SNR/CSI output arrays matched archived results within absolute tolerance 1e-12 (maximum observed difference 0.0).
- Fig. 6: all 60 scalar problems / 240 bound records regenerated; numerical records matched the archived results exactly.
- Fig. 7(c): environment 1201, one-bit phases, full training and 1,000-state evaluation completed; all six method arrays matched the archived results exactly.
- PHY environment 721: default codebooks and GNN trained from generated channels; all SNR/CSI conditions evaluated. Codebook arrays matched the originals. GNN teacher arrays matched, but retrained GNN weights and outputs differed. At 20 dB, GNN mean conditional NMSE was 0.0536429647 versus archived 0.0532638283. No tuning or replacement of archived results was performed.
- Channel/optimization/GNN-update smoke checks passed. The FL smoke completed two actual-pilot rounds using the supplied smoke codebook. Unequal-weight channel/evaluation smoke passed.

The full ten-environment sweeps, 30 complete FL trajectories, and all auxiliary retraining runs were not rerun during packaging. Use fresh mode for independent retraining, evaluate mode for the manuscript's trained PHY models, and replot mode for stored results. None of these modes implies bitwise reproducibility across arbitrary hardware.
