import argparse
import copy
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
MATBENCH_TEST_ROOT = ROOT.parent
SYM2VEC_DIR = MATBENCH_TEST_ROOT / "sym2vec_v2_200"
MAT2VEC_PATH = ROOT / "data" / "element_properties" / "mat2vec.csv"
TASK_NAME = "matbench_perovskites"
WP_RE = re.compile(r"^(\d+)([A-Za-z]+)$")
WP_TOKEN_RE = re.compile(r"^WP\|SG_(\d+)\|(\d+)([A-Za-z]+)$")
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


def safe_pow2(exponent, min_exp=-8.0, max_exp=8.0):
    exponent = torch.clamp(exponent, min=min_exp, max=max_exp)
    base = torch.as_tensor(2.0, device=exponent.device, dtype=exponent.dtype)
    return torch.pow(base, exponent)


def StableRobustL1(output, log_std, target):
    log_std = torch.clamp(log_std, min=-8.0, max=8.0)
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


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
    return normalized.replace("/", "_")


def parse_wyckoff_multiplicity(symbol):
    if not symbol:
        return None
    match = WP_RE.match(str(symbol))
    return int(match.group(1)) if match else None


def load_sym2vec_embeddings(embedding_dir):
    token_path = Path(embedding_dir) / "tokens.tsv"
    vector_path = Path(embedding_dir) / "vectors.npy"
    token_to_index = {}
    token_counts = {}
    with token_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            token = row["token"]
            token_to_index[token] = int(row["index"])
            token_counts[token] = int(row.get("count") or 0)
    vectors = np.load(vector_path).astype(np.float32, copy=False)
    return token_to_index, token_counts, vectors


def load_mat2vec(path):
    df = pd.read_csv(path, index_col=0)
    return {
        str(element): row.values.astype(np.float32, copy=False)
        for element, row in df.iterrows()
    }


class OrbitFeaturizer:
    """Perovskite-only ABX site featurizer.

    Matbench perovskites structures contain five sites. The dataset ordering is
    treated as B site at index 0, A site at index 1, and X sites at indices 2-4.
    Tokens are grouped by (element, ABX site type), so the three X sites collapse
    only when they have the same element. The fraction vector is the grouped
    occupancy divided by the total site occupancy.
    """

    ABX_ORDER = ("A", "B", "X")
    ABX_ONEHOT = {
        "A": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "B": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "X": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def __init__(self, mat2vec_path, sym2vec_dir=None, symprec=0.1, angle_tolerance=5.0):
        self.mat2vec = load_mat2vec(mat2vec_path)
        self.mat_dim = len(next(iter(self.mat2vec.values())))
        self.abx_dim = 3
        self.input_dim = self.mat_dim + self.abx_dim
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.missing_elements = Counter()
        self.missing_tokens = Counter()
        self.wp_token_fallbacks = Counter()
        self.missing_wp_sg_letters = Counter()
        self.failure_count = 0
        self.non_five_site_count = 0

    def element_vector(self, element):
        vector = self.mat2vec.get(str(element))
        if vector is None:
            self.missing_elements[str(element)] += 1
            return np.zeros(self.mat_dim, dtype=np.float32)
        return vector

    @staticmethod
    def abx_label(site_index):
        if site_index == 1:
            return "A"
        if site_index == 0:
            return "B"
        return "X"

    @staticmethod
    def site_element_and_occupancy(site):
        if site.is_ordered:
            return site.specie.symbol, 1.0
        specie, occu = max(site.species.items(), key=lambda item: item[1])
        return specie.symbol, float(occu)

    def featurize(self, structure):
        if len(structure) != 5:
            self.non_five_site_count += 1
        orbit_counts = defaultdict(float)
        for site_index, site in enumerate(structure):
            element, occu = self.site_element_and_occupancy(site)
            abx = self.abx_label(site_index)
            orbit_counts[(element, abx)] += float(occu)

        if not orbit_counts:
            orbit_counts[("H", "X")] = 1.0

        orbit_features = []
        orbit_fracs = []
        total_atoms = sum(orbit_counts.values())
        for (element, abx), count in sorted(
            orbit_counts.items(),
            key=lambda item: (
                self.ABX_ORDER.index(item[0][1]),
                item[0][0],
            ),
        ):
            feature = np.concatenate(
                [
                    self.element_vector(element),
                    self.ABX_ONEHOT[abx],
                ]
            ).astype(np.float32, copy=False)
            orbit_features.append(feature)
            orbit_fracs.append(float(count) / float(total_atoms))

        return (
            np.vstack(orbit_features).astype(np.float32, copy=False),
            np.asarray(orbit_fracs, dtype=np.float32),
        )
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
        x = self.embed(orbit_features) * safe_pow2(self.emb_scaler, -4.0, 4.0)
        src_key_padding_mask = frac <= 0

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = safe_pow2((1 - self.pos_scaler) ** 2, 0.0, 8.0)
        ple_scaler = safe_pow2((1 - self.pos_scaler_log) ** 2, 0.0, 8.0)
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
    def __init__(self, model, device, model_name, discard_n=500):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.discard_n = discard_n
        self.best_model_state = None
        self.best_val_mae = np.inf
        self.best_epoch = None
        self.gradient_clip_norm = 1.0
        self.nan_fallback_epoch = 500

    def model_is_finite(self):
        return all(
            torch.isfinite(parameter).all().item()
            for parameter in self.model.parameters()
        )

    def handle_nonfinite(self, epoch, reason):
        if epoch > self.nan_fallback_epoch and self.best_model_state is not None:
            print(
                f"{reason} at epoch {epoch}; epoch > {self.nan_fallback_epoch}, "
                f"using best checkpoint from epoch {self.best_epoch} for test.",
                flush=True,
            )
            self.model.load_state_dict(self.best_model_state)
            return True
        raise FloatingPointError(f"{reason} at epoch {epoch}")

    def fit(self, train_loader, val_loader, epochs):
        train_targets = train_loader.dataset.targets
        self.scaler = Scaler(train_targets)
        criterion = StableRobustL1
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
        stop_training = False
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
                if not torch.isfinite(loss).all().item():
                    self.optimizer.zero_grad()
                    stop_training = self.handle_nonfinite(epoch, "Non-finite loss")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not self.model_is_finite():
                    stop_training = self.handle_nonfinite(
                        epoch,
                        "Non-finite model parameters after optimizer step",
                    )
                    break
                lr_scheduler.step()

            if stop_training:
                break

            if epoch == 0 or (epoch + 1) % 2 == 0:
                try:
                    train_mae = self.evaluate(train_loader)
                    val_mae = self.evaluate(val_loader)
                except (FloatingPointError, ValueError) as exc:
                    stop_training = self.handle_nonfinite(
                        epoch,
                        f"Non-finite prediction during evaluate ({exc})",
                    )
                    break
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
                if not torch.isfinite(prediction).all().item():
                    raise FloatingPointError("prediction contains NaN or Inf")
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


def make_loader(features, fracs, targets, max_orbits, batch_size, shuffle):
    dataset = OrbitDataset(features, fracs, targets, max_orbits=max_orbits)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def run_fold(args, task, fold, device, featurizer):
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
    x_train, f_train = featurize_split(featurizer, train_structures)
    x_val, f_val = featurize_split(featurizer, val_structures)
    x_test, f_test = featurize_split(featurizer, list(test_structures))
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
            "actual": actual,
            "predicted": pred,
        }
    ).to_csv(prediction_dir / f"{args.target_name}_test_cv{fold}.csv", index=False)
    print(
        f"Fold {fold}: best_epoch={trainer.best_epoch}, "
        f"val_mae={val_mae:.6f}, test_mae={test_mae:.6f}",
        flush=True,
    )
    return {
        "fold": fold,
        "best_epoch": trainer.best_epoch,
        "val_mae": val_mae,
        "test_mae": test_mae,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perovskites-only CrabNet with mat2vec+3D ABX one-hot site tokens."
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
        "--sym2vec-dir",
        type=Path,
        default=SYM2VEC_DIR,
        help="Directory containing symmetry2vec tokens.tsv and vectors.npy.",
    )
    parser.add_argument(
        "--mat2vec-path",
        type=Path,
        default=MAT2VEC_PATH,
        help="Path to CrabNet mat2vec.csv element embeddings.",
    )
    parser.add_argument(
        "--run-name",
        default="perovskites_abx_onehot_stable",
        help="Name used for output artifacts, predictions, and checkpoint prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_name is None:
        args.target_name = str(args.task_name)
        if args.target_name.startswith("matbench_"):
            args.target_name = args.target_name[len("matbench_") :]
    device = get_compute_device(prefer_last=False)
    print(f"Running on compute device: {device}", flush=True)
    print(f"Using task: {args.task_name}", flush=True)
    print(f"Using symmetry2vec dir: {args.sym2vec_dir}", flush=True)
    print(f"Using mat2vec path: {args.mat2vec_path}", flush=True)
    featurizer = OrbitFeaturizer(
        mat2vec_path=args.mat2vec_path,
        sym2vec_dir=args.sym2vec_dir,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )
    print(f"ABX feature dim: {featurizer.input_dim} = mat2vec {featurizer.mat_dim} + ABX one-hot {featurizer.abx_dim}", flush=True)
    print("Grouping key: element + ABX site type; ABX one-hot order is A,B,X", flush=True)
    if args.task_name != TASK_NAME:
        raise ValueError(f"This script is perovskites-only; got {args.task_name}")
    benchmark = MatbenchBenchmark(autoload=False, subset=[args.task_name])
    task = benchmark.tasks_map[args.task_name]
    task.load()
    rows = []
    selected_folds = list(task.folds)[args.start_fold : args.start_fold + args.folds]
    print(f"Running folds: {selected_folds}", flush=True)
    for fold in selected_folds:
        rows.append(run_fold(args, task, fold, device, featurizer))
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
        f"Missing ABX/symmetry tokens (unused): {featurizer.missing_tokens.most_common(20)}",
        flush=True,
    )
    print(
        f"Wyckoff token fallbacks (unused): {featurizer.wp_token_fallbacks.most_common(20)}",
        flush=True,
    )
    print(
        f"Missing Wyckoff SG letters (unused): {featurizer.missing_wp_sg_letters.most_common(20)}",
        flush=True,
    )
    print(
        f"Non-5-site perovskite structures: {featurizer.non_five_site_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()




