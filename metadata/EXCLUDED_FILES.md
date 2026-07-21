# Files intentionally excluded from the upload package

The original project was not modified. The following files were excluded only
from this new upload package:

1. Full SAC training checkpoints, because each is larger than 250 MB:
   - `checkpoint_final_completo_reactivo_mejorado.pt`
   - `checkpoint_final_completo_sac_predictivo.pt`
   - `checkpoint_final_completo_sac_probabilistico.pt`

   The selected frozen actor checkpoints actually used in the article remain
   included.

2. Two generated text summaries with invalid global Cochran `nan` lines:
   - moderate uncertainty `resumen_comparacion.txt`
   - severe uncertainty `resumen_comparacion.txt`

   Their per-episode data and authoritative statistical CSV files remain
   included.

3. Historical experiments, previous training runs, ZIP duplicates, LaTeX
   template packages, caches, and diagnostic runs that were not used to produce
   the final article results.
