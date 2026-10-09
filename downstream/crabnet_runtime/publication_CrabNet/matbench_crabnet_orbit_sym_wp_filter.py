import argparse
import copy
import csv
import gzip
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MATBENCH_TEST_ROOT = ROOT.parent
CRABNET_ROOT = MATBENCH_TEST_ROOT / "matbench_test" / "CrabNet-master"
CRABNET_IMPORT_ROOT = ROOT if (ROOT / "crabnet").is_dir() else CRABNET_ROOT
if CRABNET_IMPORT_ROOT.exists() and str(CRABNET_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(CRABNET_IMPORT_ROOT))

import numpy as np
import pandas as pd
import torch
from matbench.bench import MatbenchBenchmark
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset

from crabnet.kingcrab import FractionalEncoder, ResidualNetwork
from utils.get_compute_device import get_compute_device
from utils.optim import SWA
from utils.utils import Lamb, Lookahead, RobustL1, Scaler, count_parameters


ENTITY_EMBEDDINGS_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "kge_transe"
    / "entity_embeddings.tsv"
)
MAT2VEC_PATH = CRABNET_ROOT / "data" / "element_properties" / "mat2vec.csv"
TASK_NAME = "matbench_phonons"
WP_RE = re.compile(r"^(\d+)([A-Za-z]+)$")
WP_ENTITY_RE = re.compile(r"^WP:(\d{3}):(\d+)([A-Za-z]+)$")
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


def dataset_get(dataset, key, default=None):
    if isinstance(dataset, dict):
        return dataset.get(key, default)
    return getattr(dataset, key, default)


def normalize_site_symmetry(symbol):
    if symbol is None:
        return None
    normalized = str(symbol).strip().replace(" ", "")
    if not normalized:
        return None
    return normalized


def parse_wyckoff_multiplicity(symbol):
    if not symbol:
        return None
    match = WP_RE.match(str(symbol))
    return int(match.group(1)) if match else None


def load_entity_embeddings(path):
    embedding_path = Path(path)
    if embedding_path.is_dir():
        embedding_path = embedding_path / "entity_embeddings.tsv"
    entity_to_index = {}
    vectors = []
    with embedding_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        vector_columns = [name for name in reader.fieldnames or [] if name != "id"]
        if not vector_columns:
            raise ValueError(f"No vector columns found in {embedding_path}")
        for row in reader:
            entity_id = row["id"]
            entity_to_index[entity_id] = len(vectors)
            vectors.append([float(row[name]) for name in vector_columns])
    return entity_to_index, np.asarray(vectors, dtype=np.float32)


def load_active_wp_entities(path):
    coverage_path = Path(path)
    active_entities = set()
    total_rows = 0
    with coverage_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"corpus_occurrences", "updated_from_initialization"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"WP coverage table {coverage_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )
        for row in reader:
            total_rows += 1
            entity_id = row.get("entity_id") or row.get("id")
            if not entity_id:
                raise ValueError(
                    f"WP coverage table {coverage_path} has a row without entity_id"
                )
            occurrences = int(row["corpus_occurrences"])
            updated = str(row["updated_from_initialization"]).strip().lower()
            if occurrences > 0 and updated in {"1", "true", "yes"}:
                active_entities.add(entity_id)
    if not active_entities:
        raise ValueError(f"WP coverage table {coverage_path} has no active WP rows")
    return active_entities, total_rows


def load_mat2vec(path):
    df = pd.read_csv(path, index_col=0)
    return {
        str(element): row.values.astype(np.float32, copy=False)
        for element, row in df.iterrows()
    }


class OrbitFeaturizer:
    def __init__(
        self,
        mat2vec_path,
        entity_embeddings_path,
        symprec=0.1,
        angle_tolerance=5.0,
        include_site_symmetry=True,
    ):
        self.mat2vec = load_mat2vec(mat2vec_path)
        self.entity_to_index, self.sym_vectors = load_entity_embeddings(
            entity_embeddings_path
        )
        self.wp_entity_by_sg_letter = self.build_wp_entity_lookup()
        self.mat_dim = len(next(iter(self.mat2vec.values())))
        self.sym_dim = int(self.sym_vectors.shape[1])
        self.include_site_symmetry = bool(include_site_symmetry)
        self.input_dim = self.mat_dim + self.sym_dim * (
            2 if self.include_site_symmetry else 1
        )
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.missing_elements = Counter()
        self.missing_tokens = Counter()
        self.wp_token_fallbacks = Counter()
        self.missing_wp_sg_letters = Counter()
        self.failure_count = 0

    def build_wp_entity_lookup(self):
        candidates = defaultdict(list)
        for entity_id in self.entity_to_index:
            match = WP_ENTITY_RE.match(entity_id)
            if not match:
                continue
            sg_number, multiplicity, letter = match.groups()
            candidates[(int(sg_number), letter)].append(
                (int(multiplicity), entity_id)
            )

        lookup = {}
        for key, values in candidates.items():
            values.sort(reverse=True)
            lookup[key] = values[0][1]
        return lookup

    def element_vector(self, element):
        vector = self.mat2vec.get(str(element))
        if vector is None:
            self.missing_elements[str(element)] += 1
            return np.zeros(self.mat_dim, dtype=np.float32)
        return vector

    def sym_vector(self, token):
        if not token:
            return np.zeros(self.sym_dim, dtype=np.float32)
        idx = self.entity_to_index.get(token)
        if idx is None:
            self.missing_tokens[token] += 1
            return np.zeros(self.sym_dim, dtype=np.float32)
        return self.sym_vectors[idx]

    def wyckoff_token(self, sg_number, wyckoff):
        if not wyckoff:
            return None

        exact_token = f"WP:{int(sg_number):03d}:{wyckoff}"
        if exact_token in self.entity_to_index:
            return exact_token

        match = WP_RE.match(str(wyckoff))
        if not match:
            self.missing_wp_sg_letters[(int(sg_number), str(wyckoff))] += 1
            return exact_token

        letter = match.group(2)
        fallback_token = self.wp_entity_by_sg_letter.get((int(sg_number), letter))
        if fallback_token is None:
            self.missing_wp_sg_letters[(int(sg_number), letter)] += 1
            return exact_token

        self.wp_token_fallbacks[f"{exact_token}->{fallback_token}"] += 1
        return fallback_token

    def tokenize(self, structure):
        sg_number, wyckoffs, site_syms = self.site_annotations(structure)
        orbit_counts = defaultdict(float)

        for site, wyckoff, site_sym in zip(structure, wyckoffs, site_syms):
            if site.is_ordered:
                element = site.specie.symbol
                occu = 1.0
            else:
                specie, occu = max(site.species.items(), key=lambda item: item[1])
                element = specie.symbol
            wp_token = self.wyckoff_token(sg_number, wyckoff)
            normalized_site_sym = normalize_site_symmetry(site_sym)
            site_token = (
                f"SITE_SYMM:{normalized_site_sym}" if normalized_site_sym else None
            )
            orbit_counts[(element, wp_token, site_token)] += float(occu)

        if not orbit_counts:
            orbit_counts[("H", None, None)] = 1.0

        total_atoms = sum(orbit_counts.values())
        return [
            (element, wp_token, site_token, float(count) / float(total_atoms))
            for (element, wp_token, site_token), count in sorted(
                orbit_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0][0],
                    str(item[0][1]),
                    str(item[0][2]),
                ),
            )
        ]

    def vectorize(self, tokenized_orbits):
        orbit_features = []
        orbit_fracs = []
        for element, wp_token, site_token, fraction in tokenized_orbits:
            components = [
                self.element_vector(element),
                self.sym_vector(wp_token),
            ]
            if self.include_site_symmetry:
                components.append(self.sym_vector(site_token))
            feature = np.concatenate(components).astype(np.float32, copy=False)
            orbit_features.append(feature)
            orbit_fracs.append(float(fraction))

        return (
            np.vstack(orbit_features).astype(np.float32, copy=False),
            np.asarray(orbit_fracs, dtype=np.float32),
        )

    def featurize(self, structure):
        return self.vectorize(self.tokenize(structure))

    def site_annotations(self, structure):
        n_sites = len(structure)
        try:
            analyzer = SpacegroupAnalyzer(
                structure,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
            )
            dataset = analyzer.get_symmetry_dataset()
            sg_number = int(dataset_get(dataset, "number"))
            wyckoffs, site_syms = self._site_annotations(analyzer, dataset, n_sites)
        except Exception as exc:
            self.failure_count += 1
            sg_number = 1
            wyckoffs = ["1a"] * n_sites
            site_syms = ["1"] * n_sites
            print(
                "Warning: symmetry analysis failed; using P1 fallback: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return sg_number, wyckoffs, site_syms

    def _site_annotations(self, analyzer, dataset, n_sites):
        wyckoffs = [None] * n_sites
        site_syms = [None] * n_sites

        try:
            symm = analyzer.get_symmetrized_structure()
            equivalent_indices = list(getattr(symm, "equivalent_indices", []))
            self._assign_group_or_site_values(
                wyckoffs,
                list(getattr(symm, "wyckoff_symbols", [])),
                equivalent_indices,
                n_sites,
            )
            self._assign_group_or_site_values(
                site_syms,
                list(getattr(symm, "site_symmetry_symbols", [])),
                equivalent_indices,
                n_sites,
            )
        except Exception:
            equivalent_indices = []

        raw_wyckoffs = dataset_get(dataset, "wyckoffs", None)
        if any(value is None for value in wyckoffs) and raw_wyckoffs is not None:
            letters = [str(x) for x in raw_wyckoffs]
            equivalents = dataset_get(dataset, "equivalent_atoms", None)
            if equivalents is not None:
                group_counts = Counter(int(x) for x in equivalents)
                for idx, letter in enumerate(letters[:n_sites]):
                    if wyckoffs[idx] is None:
                        wyckoffs[idx] = f"{group_counts[int(equivalents[idx])]}{letter}"
            else:
                for idx, letter in enumerate(letters[:n_sites]):
                    if wyckoffs[idx] is None:
                        wyckoffs[idx] = letter

        raw_site_syms = dataset_get(dataset, "site_symmetry_symbols", None)
        if any(value is None for value in site_syms) and raw_site_syms is not None:
            values = [str(x) for x in raw_site_syms]
            self._assign_group_or_site_values(
                site_syms,
                values,
                equivalent_indices,
                n_sites,
            )

        wyckoffs = [value if value is not None else "1a" for value in wyckoffs]
        return wyckoffs, site_syms

    @staticmethod
    def _assign_group_or_site_values(output, values, equivalent_indices, n_sites):
        if not values:
            return
        if len(values) == n_sites:
            for idx, value in enumerate(values):
                if output[idx] is None:
                    output[idx] = str(value)
            return
        if equivalent_indices and len(values) == len(equivalent_indices):
            for group_indices, value in zip(equivalent_indices, values):
                for idx in group_indices:
                    if 0 <= idx < n_sites and output[idx] is None:
                        output[idx] = str(value)


class OrbitDataset(Dataset):
    def __init__(self, orbit_features, orbit_fracs, targets, max_orbits):
        self.targets = np.asarray(targets, dtype=np.float32)
        self.max_orbits = int(max_orbits)
        self.input_dim = int(orbit_features[0].shape[1])
        self.features = np.zeros(
            (len(orbit_features), self.max_orbits, self.input_dim),
            dtype=np.float32,
        )
        self.fracs = np.zeros((len(orbit_features), self.max_orbits), dtype=np.float32)
        for idx, (features, fracs) in enumerate(zip(orbit_features, orbit_fracs)):
            n_orbits = min(len(fracs), self.max_orbits)
            self.features[idx, :n_orbits, :] = features[:n_orbits]
            self.fracs[idx, :n_orbits] = fracs[:n_orbits]
            total = self.fracs[idx].sum()
            if total > 0:
                self.fracs[idx] /= total

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.features[idx], dtype=data_type_torch),
            torch.as_tensor(self.fracs[idx], dtype=data_type_torch),
            torch.as_tensor(self.targets[idx], dtype=data_type_torch),
        )


class OrbitEncoder(nn.Module):
    def __init__(self, input_dim, d_model, N, heads):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.embed = nn.Linear(input_dim, d_model)
        self.pe = FractionalEncoder(d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(d_model, resolution=5000, log10=True)
        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.0]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=heads,
            dim_feedforward=2048,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)

    def forward(self, orbit_features, frac):
        x = self.embed(orbit_features) * 2**self.emb_scaler
        src_key_padding_mask = frac <= 0

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe[:, :, : self.d_model // 2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model // 2 :] = self.ple(frac) * ple_scaler

        x_src = (x + pe + ple).transpose(0, 1)
        x = self.transformer_encoder(
            x_src,
            src_key_padding_mask=src_key_padding_mask,
        )
        x = x.transpose(0, 1)
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
        return x


class OrbitCrabNet(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dims=3,
        d_model=512,
        N=3,
        heads=4,
        compute_device=None,
    ):
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.encoder = OrbitEncoder(input_dim, d_model, N, heads)
        self.out_hidden = [1024, 512, 256, 128]
        self.output_nn = ResidualNetwork(d_model, out_dims, self.out_hidden)

    def forward(self, orbit_features, frac):
        output = self.encoder(orbit_features, frac)
        mask = (frac <= 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)
        output = output.masked_fill(mask, 0)
        output = output.sum(dim=1) / (~mask).sum(dim=1).clamp(min=1)
        output, logits = output.chunk(2, dim=-1)
        probability = torch.ones_like(output)
        probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
        return output * probability


class Trainer:
    def __init__(self, model, device, model_name, discard_n=100):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.discard_n = discard_n
        self.best_model_state = None
        self.best_val_mae = np.inf
        self.best_epoch = None

    def fit(self, train_loader, val_loader, epochs):
        train_targets = train_loader.dataset.targets
        self.scaler = Scaler(train_targets)
        criterion = RobustL1
        base_optim = Lamb(params=self.model.parameters())
        self.optimizer = SWA(Lookahead(base_optimizer=base_optim))
        lr_scheduler = CyclicLR(
            self.optimizer,
            base_lr=1e-4,
            max_lr=6e-3,
            cycle_momentum=False,
            step_size_up=len(train_loader),
        )
        bad_checks = 0
        for epoch in range(epochs):
            self.model.train()
            for orbit_features, frac, y in train_loader:
                orbit_features = orbit_features.to(self.device)
                frac = frac.to(self.device)
                frac = frac * (1 + torch.randn_like(frac) * 0.02)
                frac = torch.clamp(frac, 0, 1)
                frac[orbit_features.abs().sum(dim=-1) == 0] = 0
                frac_sum = frac.sum(dim=1, keepdim=True).clamp(min=1e-12)
                frac = frac / frac_sum
                y = self.scaler.scale(y).to(self.device, dtype=data_type_torch)

                output = self.model(orbit_features, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = criterion(
                    prediction.view(-1),
                    uncertainty.view(-1),
                    y.view(-1),
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr_scheduler.step()

            if epoch == 0 or (epoch + 1) % 2 == 0:
                train_mae = self.evaluate(train_loader)
                val_mae = self.evaluate(val_loader)
                if val_mae < self.best_val_mae:
                    self.best_val_mae = float(val_mae)
                    self.best_epoch = int(epoch)
                    self.best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    bad_checks = 0
                    print(
                        f"New best val checkpoint: epoch {epoch}, "
                        f"val mae {val_mae:0.6g}",
                        flush=True,
                    )
                else:
                    bad_checks += 1
                print(
                    f"Epoch: {epoch}/{epochs} --- train mae: {train_mae:0.3g} "
                    f"val mae: {val_mae:0.3g} bad_checks: {bad_checks}/{self.discard_n}",
                    flush=True,
                )
                if bad_checks >= self.discard_n:
                    print(f"early-stopping now at epoch {epoch}", flush=True)
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(
                f"Restored best val checkpoint: epoch {self.best_epoch}, "
                f"val mae {self.best_val_mae:0.6g}",
                flush=True,
            )

    def predict(self, loader):
        self.model.eval()
        actual = []
        pred = []
        with torch.no_grad():
            for orbit_features, frac, y in loader:
                orbit_features = orbit_features.to(self.device)
                frac = frac.to(self.device)
                output = self.model(orbit_features, frac)
                prediction, _ = output.chunk(2, dim=-1)
                prediction = self.scaler.unscale(prediction)
                actual.extend(y.cpu().numpy().reshape(-1).tolist())
                pred.extend(prediction.cpu().numpy().reshape(-1).tolist())
        return np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)

    def evaluate(self, loader):
        actual, pred = self.predict(loader)
        return float(mean_absolute_error(actual, pred))

    def save(self):
        os.makedirs("models/trained_models", exist_ok=True)
        path = f"models/trained_models/{self.model_name}.pth"
        torch.save(
            {
                "weights": self.model.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_mae": self.best_val_mae,
            },
            path,
        )
        print(f"Saved network ({self.model_name}) to {path}", flush=True)


def split_train_val(structures, targets, val_fraction, random_state):
    n_samples = len(structures)
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_samples * val_fraction)))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])
    structures_list = list(structures)
    targets_arr = np.asarray(targets, dtype=np.float32)
    train_structures = [structures_list[idx] for idx in train_idx]
    val_structures = [structures_list[idx] for idx in val_idx]
    return (
        train_structures,
        targets_arr[train_idx],
        val_structures,
        targets_arr[val_idx],
    )


def featurize_split(featurizer, structures):
    features = []
    fracs = []
    for structure in structures:
        feature, frac = featurizer.featurize(structure)
        features.append(feature)
        fracs.append(frac)
    return features, fracs


def _tokenize_split(featurizer, structures):
    return [featurizer.tokenize(structure) for structure in structures]


def _vectorize_split(featurizer, tokenized_structures):
    features = []
    fracs = []
    for tokenized_orbits in tokenized_structures:
        feature, frac = featurizer.vectorize(tokenized_orbits)
        features.append(feature)
        fracs.append(frac)
    return features, fracs


def _filter_tokenized_splits_by_wp_coverage(tokenized, active_wp_entities):
    filtered = {}
    keep_positions = {}
    audit = {}
    for split_name, samples in tokenized.items():
        kept_samples = []
        kept_positions = []
        missing_wp_sample_counts = Counter()
        removed_positions = []
        for position, sample in enumerate(samples):
            wp_tokens = {orbit[1] for orbit in sample if orbit[1]}
            uncovered = sorted(wp_tokens.difference(active_wp_entities))
            if not wp_tokens:
                uncovered = ["<NO_WP_TOKEN>"]
            if uncovered:
                removed_positions.append(position)
                missing_wp_sample_counts.update(uncovered)
                continue
            kept_samples.append(sample)
            kept_positions.append(position)

        if not kept_samples:
            raise ValueError(
                f"WP coverage filtering removed every sample from {split_name}"
            )
        filtered[split_name] = kept_samples
        keep_positions[split_name] = np.asarray(kept_positions, dtype=np.int64)
        audit[split_name] = {
            "original_samples": int(len(samples)),
            "kept_samples": int(len(kept_samples)),
            "removed_samples": int(len(removed_positions)),
            "removed_fraction": float(len(removed_positions) / max(1, len(samples))),
            "removed_position_preview": removed_positions[:100],
            "uncovered_wp_sample_counts": dict(
                sorted(
                    missing_wp_sample_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }
    return filtered, keep_positions, audit


def _token_cache_metadata(args, fold, split_lengths):
    return {
        "format_version": 1,
        "task_name": args.task_name,
        "fold": int(fold),
        "val_fraction": float(args.val_fraction),
        "random_state": int(args.random_state),
        "symprec": float(args.symprec),
        "angle_tolerance": float(args.angle_tolerance),
        "limit": args.limit,
        "split_lengths": dict(split_lengths),
    }


def _token_cache_path(args, fold):
    if args.token_cache_dir is None:
        return None
    return Path(args.token_cache_dir) / f"{args.task_name}_fold{int(fold)}_tokens.pkl.gz"


def featurize_splits_with_cache(
    featurizer,
    split_structures,
    args,
    fold,
    active_wp_entities=None,
    vectorize=True,
):
    split_lengths = {name: len(values) for name, values in split_structures.items()}
    metadata = _token_cache_metadata(args, fold, split_lengths)
    cache_path = _token_cache_path(args, fold)
    tokenized = None

    if cache_path is not None and cache_path.exists():
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata") != metadata:
            raise ValueError(
                f"Token cache metadata mismatch for {cache_path}; remove the stale cache"
            )
        tokenized = payload["tokenized"]
        print(f"Loaded orbit token cache: {cache_path}", flush=True)

    if tokenized is None:
        tokenized = {
            name: _tokenize_split(featurizer, structures)
            for name, structures in split_structures.items()
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with gzip.open(temporary_path, "wb", compresslevel=3) as handle:
                pickle.dump(
                    {"metadata": metadata, "tokenized": tokenized},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary_path.replace(cache_path)
            with (cache_path.with_suffix(cache_path.suffix + ".json")).open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
            print(f"Saved orbit token cache: {cache_path}", flush=True)

    keep_positions = {
        name: np.arange(len(samples), dtype=np.int64)
        for name, samples in tokenized.items()
    }
    coverage_audit = None
    if active_wp_entities is not None:
        tokenized, keep_positions, coverage_audit = (
            _filter_tokenized_splits_by_wp_coverage(
                tokenized,
                active_wp_entities,
            )
        )
        for name, split_audit in coverage_audit.items():
            print(
                f"Fold {fold} {name}: WP coverage kept "
                f"{split_audit['kept_samples']}/{split_audit['original_samples']} "
                f"samples; removed={split_audit['removed_samples']}",
                flush=True,
            )

    if not vectorize:
        return None, keep_positions, coverage_audit
    features = {
        name: _vectorize_split(featurizer, tokenized[name])
        for name in split_structures
    }
    return features, keep_positions, coverage_audit


def make_loader(features, fracs, targets, max_orbits, batch_size, shuffle):
    dataset = OrbitDataset(features, fracs, targets, max_orbits=max_orbits)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def run_fold(args, task, fold, device, featurizer, active_wp_entities=None):
    train_val_structures, train_val_targets = task.get_train_and_val_data(
        fold,
        as_type="tuple",
    )
    test_structures, test_targets = task.get_test_data(
        fold,
        as_type="tuple",
        include_target=True,
    )
    train_structures, y_train, val_structures, y_val = split_train_val(
        train_val_structures,
        train_val_targets,
        args.val_fraction,
        args.random_state + int(fold),
    )
    if args.limit:
        train_structures = train_structures[: args.limit]
        y_train = y_train[: args.limit]
        val_structures = val_structures[: max(1, args.limit // 5)]
        y_val = y_val[: max(1, args.limit // 5)]
        test_structures = list(test_structures)[: args.limit]
        test_targets = np.asarray(test_targets, dtype=np.float32)[: args.limit]

    print(
        f"Fold {fold}: featurizing train={len(train_structures)}, "
        f"val={len(val_structures)}, test={len(test_structures)}",
        flush=True,
    )
    split_features, keep_positions, coverage_audit = featurize_splits_with_cache(
        featurizer,
        {
            "train": train_structures,
            "val": val_structures,
            "test": list(test_structures),
        },
        args,
        fold,
        active_wp_entities=active_wp_entities,
        vectorize=not args.cache_only,
    )
    if coverage_audit is not None:
        audit_payload = {
            "coverage_path": str(Path(args.wp_training_coverage).resolve()),
            "active_wp_entities": int(len(active_wp_entities)),
            "filter_rule": (
                "Drop a sample when any of its WP entities has zero MP corpus "
                "occurrences or was not updated from initialization."
            ),
            "fold": int(fold),
            "splits": coverage_audit,
        }
        audit_dir = Path("artifacts") / args.run_name
        audit_dir.mkdir(parents=True, exist_ok=True)
        with (audit_dir / f"wp_coverage_filter_fold{fold}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(audit_payload, handle, indent=2, sort_keys=True)
    if args.cache_only:
        print(f"Fold {fold}: token cache complete; skipping model training", flush=True)
        return None
    target_splits = {
        "train": np.asarray(y_train, dtype=np.float32),
        "val": np.asarray(y_val, dtype=np.float32),
        "test": np.asarray(test_targets, dtype=np.float32),
    }
    target_splits = {
        name: targets[keep_positions[name]]
        for name, targets in target_splits.items()
    }
    y_train = target_splits["train"]
    y_val = target_splits["val"]
    test_targets = target_splits["test"]
    x_train, f_train = split_features["train"]
    x_val, f_val = split_features["val"]
    x_test, f_test = split_features["test"]
    max_orbits = max(
        max(len(frac) for frac in f_train),
        max(len(frac) for frac in f_val),
        max(len(frac) for frac in f_test),
    )
    print(f"Fold {fold}: max_orbits={max_orbits}", flush=True)

    train_loader = make_loader(
        x_train,
        f_train,
        y_train,
        max_orbits,
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        x_val,
        f_val,
        y_val,
        max_orbits,
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        x_test,
        f_test,
        test_targets,
        max_orbits,
        args.batch_size,
        shuffle=False,
    )

    model = OrbitCrabNet(
        input_dim=featurizer.input_dim,
        d_model=args.d_model,
        N=args.layers,
        heads=args.heads,
        compute_device=device,
    ).to(device)
    print(f"Model size: {count_parameters(model)} parameters", flush=True)
    trainer = Trainer(
        model,
        device,
        model_name=f"{args.target_name}_{args.run_name}{fold}",
        discard_n=args.patience,
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    trainer.save()
    test_mae = trainer.evaluate(test_loader)
    val_mae = trainer.evaluate(val_loader)
    actual, pred = trainer.predict(test_loader)
    prediction_dir = Path("publication_predictions") / f"{args.run_name}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "fold": fold,
            "test_split_position": keep_positions["test"],
            "actual": actual,
            "predicted": pred,
        }
    ).to_csv(prediction_dir / f"{args.target_name}_test_cv{fold}.csv", index=False)
    print(
        f"Fold {fold}: best_epoch={trainer.best_epoch}, "
        f"val_mae={val_mae:.6f}, test_mae={test_mae:.6f}",
        flush=True,
    )
    result = {
        "fold": fold,
        "best_epoch": trainer.best_epoch,
        "val_mae": val_mae,
        "test_mae": test_mae,
    }
    if coverage_audit is not None:
        for name, split_audit in coverage_audit.items():
            result[f"{name}_samples_before_filter"] = split_audit[
                "original_samples"
            ]
            result[f"{name}_samples_after_filter"] = split_audit["kept_samples"]
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="CrabNet with mat2vec+KG symmetry entity embeddings for Matbench."
    )
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument(
        "--target-name",
        default=None,
        help="Short name used in checkpoint/prediction filenames. "
        "Defaults to task-name with leading 'matbench_' removed.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="First Matbench fold index to run. Use with --folds for sharding.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--token-cache-dir",
        type=Path,
        default=None,
        help="Optional cache for embedding-independent orbit tokens.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build or validate the orbit-token cache, then skip model training.",
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="Optional per-process CUDA allocator limit in the interval (0, 1].",
    )
    parser.add_argument(
        "--entity-embeddings",
        type=Path,
        default=ENTITY_EMBEDDINGS_PATH,
        help="Path to KG entity_embeddings.tsv, or its containing directory.",
    )
    parser.add_argument(
        "--disable-site-symmetry",
        action="store_true",
        help="Use element and Wyckoff vectors only; omit the site-symmetry channel.",
    )
    parser.add_argument(
        "--wp-training-coverage",
        type=Path,
        default=None,
        help=(
            "TSV describing which WP rows occurred during embedding pretraining; "
            "used with --drop-uncovered-wp-samples."
        ),
    )
    parser.add_argument(
        "--drop-uncovered-wp-samples",
        action="store_true",
        help=(
            "Remove each train/validation/test sample containing any WP that was "
            "not observed and updated during embedding pretraining."
        ),
    )
    parser.add_argument(
        "--sym2vec-dir",
        type=Path,
        dest="entity_embeddings",
        help="Deprecated alias for --entity-embeddings.",
    )
    parser.add_argument(
        "--mat2vec-path",
        type=Path,
        default=MAT2VEC_PATH,
        help="Path to CrabNet mat2vec.csv element embeddings.",
    )
    parser.add_argument(
        "--run-name",
        default="orbit_sym",
        help="Name used for output artifacts, predictions, and checkpoint prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda_memory_fraction is not None and not (
        0.0 < args.cuda_memory_fraction <= 1.0
    ):
        raise ValueError("--cuda-memory-fraction must be in (0, 1]")
    if args.target_name is None:
        args.target_name = str(args.task_name)
        if args.target_name.startswith("matbench_"):
            args.target_name = args.target_name[len("matbench_") :]
    if args.drop_uncovered_wp_samples and args.wp_training_coverage is None:
        raise ValueError(
            "--drop-uncovered-wp-samples requires --wp-training-coverage"
        )
    device = get_compute_device(prefer_last=False)
    if args.cuda_memory_fraction is not None and torch.device(device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=device,
        )
    print(f"Running on compute device: {device}", flush=True)
    print(f"Using task: {args.task_name}", flush=True)
    print(f"Using entity embeddings: {args.entity_embeddings}", flush=True)
    print(f"Using mat2vec path: {args.mat2vec_path}", flush=True)
    featurizer = OrbitFeaturizer(
        mat2vec_path=args.mat2vec_path,
        entity_embeddings_path=args.entity_embeddings,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        include_site_symmetry=not args.disable_site_symmetry,
    )
    print(f"Include site symmetry: {featurizer.include_site_symmetry}", flush=True)
    print(f"Orbit feature dim: {featurizer.input_dim}", flush=True)
    active_wp_entities = None
    if args.drop_uncovered_wp_samples:
        active_wp_entities, coverage_rows = load_active_wp_entities(
            args.wp_training_coverage
        )
        unavailable_vectors = active_wp_entities.difference(featurizer.entity_to_index)
        if unavailable_vectors:
            raise ValueError(
                f"{len(unavailable_vectors)} active WP entities have no embedding; "
                f"examples: {sorted(unavailable_vectors)[:10]}"
            )
        print(
            f"WP coverage filter: active={len(active_wp_entities)}/"
            f"{coverage_rows} canonical rows",
            flush=True,
        )
    benchmark = MatbenchBenchmark(autoload=False, subset=[args.task_name])
    task = benchmark.tasks_map[args.task_name]
    task.load()
    rows = []
    selected_folds = list(task.folds)[args.start_fold : args.start_fold + args.folds]
    print(f"Running folds: {selected_folds}", flush=True)
    for fold in selected_folds:
        result = run_fold(
            args,
            task,
            fold,
            device,
            featurizer,
            active_wp_entities=active_wp_entities,
        )
        if result is not None:
            rows.append(result)
    if args.cache_only:
        print("Orbit token cache-only run complete", flush=True)
        return
    df = pd.DataFrame(rows)
    out_dir = Path("artifacts") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    score_name = (
        "fold_scores.csv"
        if args.start_fold == 0 and args.folds >= 5
        else f"fold_scores_start{args.start_fold}_n{args.folds}.csv"
    )
    df.to_csv(out_dir / score_name, index=False)
    print(f"Average test mae: {df['test_mae'].mean()}", flush=True)
    print(f"Average val mae: {df['val_mae'].mean()}", flush=True)
    print(
        f"Missing elements: {featurizer.missing_elements.most_common(20)}",
        flush=True,
    )
    print(
        f"Missing symmetry tokens: {featurizer.missing_tokens.most_common(20)}",
        flush=True,
    )
    print(
        f"Wyckoff token fallbacks: {featurizer.wp_token_fallbacks.most_common(20)}",
        flush=True,
    )
    print(
        f"Missing Wyckoff SG letters: {featurizer.missing_wp_sg_letters.most_common(20)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
