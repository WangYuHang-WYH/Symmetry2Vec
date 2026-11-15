# KG + CrabNet prediction demo

This demo predicts perovskite formation energy (eV/unit cell) using
`model_best.pth`, Mat2Vec element features, and the archived 200-dimensional
Symmetry2Vec-KG Wyckoff vectors. No training is performed.

## Files and environment

```text
demo/
  model_best.pth
  predict.py
  examples/
    perovskite_orbits.json
    CaTiO3.cif
  README.md
```

Use the environment described in the repository's `requirements.txt` (Python
3.9 was verified). Run the commands below from the **Symmetry2Vec repository
root**. The script also works from other directories because its default
resource paths are resolved relative to its own location.

```bash
python -m pip install -r requirements.txt
```

The demo uses `data/kg/entity_embeddings.tsv` and
`downstream/crabnet_runtime/data/element_properties/mat2vec.csv`. BERT weights
and the LM tokenizer are not needed.

The supplied checkpoint records `matbench_perovskites`, fold 0, seed 42,
best epoch 703, and best validation MAE 0.0449663 eV/atom. Its configuration
is input dimension 400 (200 Mat2Vec + 200 KG), model dimension 512, three
Transformer layers, four attention heads, and at most five orbit rows.
The script loads the saved target mean/standard deviation and model weights
strictly. It uses the archived stable perovskite CrabNet implementation and
the saved symmetry tolerances (`symprec=0.1`, `angle_tolerance=5.0`), with
the site-symmetry input channel disabled.

## 1. Default: official perovskites fold-0 test set

```bash
python demo/predict.py
```

With no sample input, the script loads the **official Matbench fold-0 test
split**, predicts every test structure, and reports test MAE. It does not
refit the model or the target scaler. 

Default outputs:

- `demo/outputs/perovskites_fold0_predictions.csv`: material ID, formula,
  reference formation energy, and predicted formation energy.
- `demo/outputs/perovskites_fold0_predictions.summary.json`: sample count,
  MAE, checkpoint path, device, and feature-resolution diagnostics.

```bash
python demo/predict.py --device cpu --batch-size 64 --output demo/outputs/test.csv
```

The default device is CUDA when available, otherwise CPU. Use `--device cuda:0`
to choose a specific visible GPU. `--cpu-threads` defaults to 4.

## 2. One sample: element, Wyckoff position, and fraction

Specify the international space-group number and repeat `--orbit`:

```bash
python demo/predict.py --space-group 221 \
  --orbit Ca 1a 0.2 \
  --orbit Ti 1b 0.2 \
  --orbit O 3c 0.6 \
  --output demo/outputs/single_orbits.json
```

Alternatively, use the included JSON example:

```bash
python demo/predict.py --orbits demo/examples/perovskite_orbits.json
```

```json
{
  "space_group": 221,
  "orbits": [
    {"element": "Ca", "wyckoff": "1a", "fraction": 0.2},
    {"element": "Ti", "wyckoff": "1b", "fraction": 0.2},
    {"element": "O", "wyckoff": "3c", "fraction": 0.6}
  ]
}
```

Each fraction is the **fraction of atoms assigned to that element-Wyckoff
pair**, not an individual site's occupancy. Positive atom-count ratios such
as 1, 1, 3 are also accepted and normalized to 0.2, 0.2, 0.6. Duplicate
element-Wyckoff rows are combined. All fractions must be finite and strictly
positive. A single sample must use one space group and no more than the
checkpoint's five orbit rows after combining duplicates.

Full KG identifiers are also accepted, so a separate space group is optional:

```bash
python demo/predict.py \
  --orbit Ca WP:221:1a 1 \
  --orbit Ti WP:221:1b 1 \
  --orbit O WP:221:3c 3
```

## 3. One sample: a complete CIF

```bash
python demo/predict.py --cif demo/examples/CaTiO3.cif \
  --output demo/outputs/single_cif.json
```

You can also supply the entire CIF text with `--cif-text`. For example, in Bash:

```bash
python demo/predict.py --cif-text "$(cat demo/examples/CaTiO3.cif)"
```

Resource paths can be overridden:

```bash
python demo/predict.py \
  --checkpoint demo/model_best.pth \
  --kg-vectors data/kg/entity_embeddings.tsv \
  --mat2vec downstream/crabnet_runtime/data/element_properties/mat2vec.csv
```

The demo requires a checkpoint carrying the supplied model/preprocessing
metadata and the perovskites fold-0 task identity. Keep the archived KG and
Mat2Vec tables paired with the trained checkpoint: matching vector dimensions
alone does not establish compatibility with newly retrained embeddings.

## Verified run

On the configured Linux environment with PyTorch 2.4.0 and an RTX 3080, the
supplied checkpoint predicted all **3,786 official fold-0 test samples** with
MAE **0.0462823 eV/atom**, without missing vectors or symmetry fallbacks.
The example JSON, repeated orbit arguments (including 1:1:3 atom counts),
CIF file, and inline CIF text produce the same single-sample prediction on
CPU. Run results and checkpoint SHA-256 are recorded in
`validation_results.json`. Small floating-point differences across devices
and library versions are possible.
