# Final Symmetry2Vec code and data bundle

This bundle contains the archived code and local data required to reproduce the
KG and LM WP vectors and to run the Final Matbench downstream
experiments for KG, LM, Wren, one-hot, and coordination
ABX one-hot.

All commands below assume Linux, CUDA, and that the current directory is the
relevant component directory. The original experiments used the `wyh_env`
Conda environment. Package versions recorded from the experiment machine are
listed in `requirements.txt`.

## Directory layout

```text
final_code/
  README.md
  requirements.txt
  kg/
    symmetry_kg/                 KG construction and standard TransE
    scripts/                     portable training and extraction scripts
    data/base_graph_source/      frozen source graph consumed by KG
    data/kg_reference/           frozen KG graph produced in the experiment
  lm/
    scripts/                     corpus, tokenizer, and LM training code
    configs/                     portable training and corpus configurations
    data/source_profiles/        canonical-Hall MP structure profiles
    data/corpus/                 exact LM train/validation JSONL corpus
    resources/bert-base-cased/   local BERT initialization weights/tokenizer
    resources/tokenizer/         exact extended one-word/one-token tokenizer
  downstream/
    scripts/                     Matbench and ABX launchers
    crabnet_runtime/             shared CrabNet model/runtime dependencies
    data/                        Wren, one-hot, and Mat2Vec data
    splits/                      ABX role cache and composition-grouped split
    vector_registry.json         portable representation paths

final_data/
  kg/                            final KG WP/full/relation vectors
  lm/                            final LM WP table and coverage
```

`manifest.sha256` records SHA-256 hashes for all files in `final_code` and
`final_data` except the manifest itself.

## 1. KG

KG starts from the frozen base graph and changes only the lattice-type
subgraph. The legacy PyXtal setting-dependent lattice IDs are replaced by a
setting-independent mapping from all 230 International space-group types to
the standard 14 Bravais lattice types. All non-lattice nodes and triples are
kept unchanged.

The graph contains 5,691 entities and 49,647 directed triples. Entity types
are 230 space groups, 1,731 Wyckoff positions, 78 site symmetries, 3,582
coordinate entities, 32 point groups, 17 multiplicities, 7 crystal systems,
and 14 Bravais lattice types.

Train KG with the archived configuration:

```bash
cd final_code/kg
DEVICE=cuda:0 bash scripts/run_kg_training.sh
```

The command performs:

1. base-graph to KG transformation;
2. standard TransE training;
3. extraction of the 1,731 canonical WP rows.

TransE settings are 200 dimensions, 1,000 epochs, batch size 4,096, four
filtered negatives per positive, learning rate 0.001, margin 1.0, L1 distance,
and seed 13. Generated files are placed under `kg/artifacts/`.

The original graph and vector outputs are retained in
`kg/data/kg_reference/` and `../final_data/kg/`.

## 2. LM

The packaged LM training set contains 154,377 Materials Project
structures: 152,806 training structures and 1,571 validation structures.
Structure profiles were standardized to conventional cells with spglib using
`symprec=0.1`, `angle_tolerance=5.0`, and a canonical Hall setting. The corpus
contains 1,606 empirically observed WP classes from the complete 1,731-row
canonical vocabulary.

The exact prebuilt corpus, tokenizer, and BERT initialization are included, so
LM training does not require network access or the original MP CIF archive:

```bash
cd final_code/lm
NPROC_PER_NODE=1 bash scripts/run_lm_training.sh
```

Set `NPROC_PER_NODE=2` for the original two-GPU distributed layout. This is the
single LM training path used to produce the Final Matbench representation.

LM uses 10 epochs, early-stopping patience 3, per-GPU batch size 32,
BERT learning rate `2e-5`, WP-table learning rate `5e-4`, FP16, seed 42, and a
15% WP mask rate. When a WP is selected, its WP token, multiplicity, and site
symmetry are all masked. Prediction is over the 1,606 WP classes observed in
MP. The tied trainable WP table is 1,731 x 200; the 125 unseen rows remain
inactive and are excluded by the downstream coverage filter.

The prebuilt corpus can be regenerated from the packaged source profiles:

```bash
python scripts/build_mp_wp_mlm_corpus.py \
  --config configs/build_mp_corpus.json
python scripts/prepare_mp_wp_mlm_tokenizer.py \
  --config configs/build_mp_corpus.json
```

`build_mp_fixed_text_corpus.py` and
`build_alexandria_fixed_text_corpus.py` are retained to document the upstream
spglib canonical-Hall profiling step. Re-running that first step requires the
original Materials Project CIF archive, which is not required for vector
training and is not duplicated here. The resulting exact profiles and source
manifest are included under `data/source_profiles/`.

## 3. Matbench vector runs

The unified runner exposes four WP representations:

| CLI name | Representation | Dimension | Coverage filter |
|---|---|---:|---|
| `kg` | Symmetry2Vec-KG | 200 | no |
| `lm` | Symmetry2Vec-LM | 200 | 1,606 active WP rows |
| `wren` | official Wren `bra-alg-off` descriptor | 444 | no |
| `onehot` | canonical WP identity matrix | 1,731 | no |

Run all five official folds of any method-task combination with one script:

```bash
cd final_code/downstream
python scripts/run_matbench.py \
  --method kg \
  --task mp_e_form
```

Available task names are `jdft2d`, `phonons`, `dielectric`, `log_gvrh`,
`log_kvrh`, `mp_e_form`, `mp_gap`, and `perovskites`. Add `--fold 0` through
`--fold 4` to run only one fold; without `--fold`, all five folds run
sequentially in five independent child processes, so the RNG is reset to seed
42 at the start of every fold.

Default settings are 1,000 epochs, patience 100, batch size 128, validation
fraction 0.1, fold/validation seed 42, no site-symmetry input channel, and a
0.48 per-process CUDA memory fraction. Element features are the packaged
Mat2Vec table. Space groups and WP assignments are recomputed from each
Matbench `Structure` with pymatgen `SpacegroupAnalyzer` using `symprec=0.1`
and `angle_tolerance=5.0`; pymatgen uses spglib underneath.

Use `--dry-run` to validate paths and print the exact child command without
starting training. Four workers cover all 8 tasks x 5 folds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_8tasks.py --worker 0 --method kg
CUDA_VISIBLE_DEVICES=0 python scripts/run_8tasks.py --worker 1 --method kg
CUDA_VISIBLE_DEVICES=1 python scripts/run_8tasks.py --worker 2 --method kg
CUDA_VISIBLE_DEVICES=1 python scripts/run_8tasks.py --worker 3 --method kg
```

`crabnet_runtime/` is not a reduced or altered CrabNet model. It is the exact
shared model code and imported support files needed by these runners, copied
without the unrelated experiment outputs and historical launch scripts from
the larger CrabNet repository.

## 4. Coordination ABX one-hot

ABX is specific to `matbench_perovskites`. It uses geometry-derived
crystallographic coordination roles: A, B, and X are encoded as a 3-dimensional
one-hot vector and concatenated to Mat2Vec element features. The exact role
cache is included.

Run one official fold:

```bash
cd final_code/downstream
python scripts/run_perovskites_coordination_abx.py \
  --split-kind official \
  --fold 0
```

The included composition-grouped files also allow
`--split-kind composition_grouped`. Rebuild the ABX role cache with:

```bash
python scripts/prepare_perovskites_coordination_abx_cache.py --force
```

## Data provenance

- KG graph source: frozen `kg/data/base_graph_source` nodes and triples.
- LM structure source: Materials Project current non-deprecated
  Summary collection, database version `2026.04.13`, as recorded in
  `lm/data/source_profiles_manifest.json`.
- Wren source: `downstream/sources/wren444/bra-alg-off.json`.
- Matbench structures and targets: supplied by `matbench==0.6` at runtime.
- Final vector tables: `../final_data/`.

The archived code retains the exact original data contracts. It does not
silently substitute a new symmetry tolerance, tokenizer, split, or vector row
order.
