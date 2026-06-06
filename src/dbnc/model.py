"""The DBNC model: a differentiable Bayesian network classifier.

Reference implementation of the ``DBNC`` model described in the paper. The model
couples a *rank-parameterised soft DAG* (acyclic by construction) with an
*attention-based neural conditional probability model*, and is trained
end-to-end under a regularised hybrid generative--discriminative objective.

The class depends only on PyTorch, NumPy, and a couple of scikit-learn metrics,
so it can be reused independently of the experiment pipeline in
:mod:`dbnc._pipeline`.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dbnc._common import (
    ExperimentError,
    GRAPH_THRESHOLD,
    is_acyclic,
    validation_checkpoint_improved,
)

__all__ = ["DBNC"]


class DBNC(nn.Module):
    """Differentiable BNC with rank-acyclic learned or supplied adjacency."""

    def __init__(
        self,
        cards: Sequence[int],
        n_classes: int,
        *,
        d: int = 32,
        n_heads: int = 4,
        epochs: int = 100,
        lr: float = 0.003,
        batch_size: int = 1024,
        lambda_sparse: float = 0.001,
        lambda_degree: float = 0.001,
        max_parents: float = 2.0,
        hybrid_weight: float = 0.5,
        device: str = "cpu",
        dropout: float = 0.10,
        cpd_type: str = "attention",
        fixed_adjacency: np.ndarray | None = None,
        early_stopping_rounds: int = 10,
        min_epochs_before_checkpoint: int = 1,
        min_epochs_before_stopping: int = 1,
        structure_warmup_epochs: int = 0,
        edge_logit_mean: float = -2.0,
        rank_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.cards = tuple(int(card) for card in cards)
        self.n_classes = int(n_classes)
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_degree = float(lambda_degree)
        self.max_parents = float(max_parents)
        self.hybrid_weight = float(hybrid_weight)
        self.dropout_rate = float(dropout)
        self.cpd_type = str(cpd_type)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.min_epochs_before_checkpoint = int(min_epochs_before_checkpoint)
        self.min_epochs_before_stopping = int(min_epochs_before_stopping)
        self.structure_warmup_epochs = int(structure_warmup_epochs)
        self.edge_logit_mean = float(edge_logit_mean)
        self.rank_init_std = float(rank_init_std)
        self.m = len(self.cards)
        self.epochs_trained = 0
        self.best_epoch = 0
        self.best_validation_accuracy: float | None = None
        self.best_validation_log_loss: float | None = None

        if self.d % self.n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        if not 0.0 <= self.hybrid_weight <= 1.0:
            raise ValueError("hybrid_weight must lie between 0 and 1 inclusive")
        if self.cpd_type not in {"attention", "mlp"}:
            raise ValueError(f"Unsupported CPD type: {self.cpd_type}")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if self.min_epochs_before_checkpoint < 1:
            raise ValueError("min_epochs_before_checkpoint must be positive")
        if self.min_epochs_before_stopping < 1:
            raise ValueError("min_epochs_before_stopping must be positive")
        if self.structure_warmup_epochs < 0:
            raise ValueError("structure_warmup_epochs cannot be negative")
        if self.rank_init_std <= 0.0:
            raise ValueError("rank_init_std must be positive")

        self.cls_emb = nn.Embedding(self.n_classes, self.d)
        self.par_emb = nn.ModuleList(nn.Embedding(card, self.d) for card in self.cards)
        self.base_logit = nn.ParameterList(
            nn.Parameter(torch.zeros(self.n_classes, card)) for card in self.cards
        )
        self.var_q = nn.Embedding(self.m, self.d)
        self.cls_logit = nn.Parameter(torch.zeros(self.n_classes))
        self.edge_logit = nn.Parameter(torch.empty(self.m, self.m))
        self.rank = nn.Parameter(torch.empty(self.m))
        self.dropout = nn.Dropout(self.dropout_rate)
        if self.cpd_type == "attention":
            self.out_emb = nn.ModuleList(nn.Embedding(card, self.d) for card in self.cards)
            self.Wq: nn.Linear | None = nn.Linear(self.d, self.d, bias=False)
            self.Wk: nn.Linear | None = nn.Linear(self.d, self.d, bias=False)
            self.Wv: nn.Linear | None = nn.Linear(self.d, self.d, bias=False)
            self.mlp_cpd = nn.ModuleList()
        else:
            self.out_emb = nn.ModuleList()
            self.Wq = None
            self.Wk = None
            self.Wv = None
            self.mlp_cpd = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(3 * self.d, self.d),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.d, card),
                )
                for card in self.cards
            )
        if fixed_adjacency is None:
            fixed = torch.empty(0, dtype=torch.float32)
        else:
            fixed = torch.as_tensor(fixed_adjacency, dtype=torch.float32)
            if fixed.shape != (self.m, self.m):
                raise ValueError("Fixed adjacency has the wrong shape")
        self.register_buffer("_fixed_adjacency", fixed)
        self.reset_parameters()
        self.to(torch.device(device))

    @property
    def structure_type(self) -> str:
        if self._fixed_adjacency.numel() == 0:
            return "learned_rank_acyclic"
        if torch.count_nonzero(self._fixed_adjacency) == 0:
            return "no_edges"
        return "fixed_tan"

    def reset_parameters(self) -> None:
        for layer in (self.Wq, self.Wk, self.Wv):
            if layer is not None:
                layer.reset_parameters()
        for embedding in [self.cls_emb, *self.par_emb, *self.out_emb, self.var_q]:
            nn.init.normal_(embedding.weight, std=self.d ** -0.5)
        for parameter in self.base_logit:
            nn.init.zeros_(parameter)
        nn.init.zeros_(self.cls_logit)
        nn.init.normal_(self.edge_logit, mean=self.edge_logit_mean, std=0.1)
        nn.init.normal_(self.rank, mean=0.0, std=self.rank_init_std)
        for network in self.mlp_cpd:
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def learned_adjacency(self) -> torch.Tensor:
        gap = F.relu(self.rank[None] - self.rank[:, None])
        return torch.sigmoid(self.edge_logit) * gap / (1.0 + gap)

    def adjacency(self) -> torch.Tensor:
        return self.learned_adjacency() if self._fixed_adjacency.numel() == 0 else self._fixed_adjacency

    def _attention_log_joint(self, X: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        assert self.Wq is not None and self.Wk is not None and self.Wv is not None
        rows = torch.arange(X.shape[0], device=X.device)
        heads, head_dim = self.n_heads, self.d // self.n_heads
        logp = F.log_softmax(self.cls_logit, 0).expand(X.shape[0], -1)
        parents = torch.stack(
            [embedding(X[:, index]) for index, embedding in enumerate(self.par_emb)], 1
        )
        parents = self.dropout(parents)
        keys = self.Wk(parents).view(X.shape[0], self.m, heads, head_dim)
        values = self.Wv(parents).view(X.shape[0], self.m, heads, head_dim)
        queries = self.Wq(
            self.var_q.weight[None] + self.cls_emb.weight[:, None]
        ).view(self.n_classes, self.m, heads, head_dim)
        score = torch.einsum("yihd,nmhd->nyimh", queries, keys) / (head_dim ** 0.5)
        parent_weight = adjacency.T
        mask = parent_weight > 0
        score = score + torch.log(parent_weight.clamp_min(1e-9))[None, None, :, :, None]
        score = score.masked_fill(
            ~mask[None, None, :, :, None],
            torch.finfo(score.dtype).min,
        )
        attention = score.softmax(dim=3) * mask.any(dim=1)[None, None, :, None, None]
        attention = self.dropout(attention * parent_weight[None, None, :, :, None])
        context = torch.einsum("nyimh,nmhd->nyihd", attention, values).reshape(
            X.shape[0], self.n_classes, self.m, self.d
        )
        for feature, embedding in enumerate(self.out_emb):
            logits = (
                context[:, :, feature, :] @ embedding.weight.T
                + self.base_logit[feature][None, :, :]
            )
            logp = logp + logits.log_softmax(-1)[rows, :, X[:, feature]]
        return logp

    def _mlp_log_joint(self, X: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(X.shape[0], device=X.device)
        logp = F.log_softmax(self.cls_logit, 0).expand(X.shape[0], -1)
        parent_embeddings = torch.stack(
            [embedding(X[:, index]) for index, embedding in enumerate(self.par_emb)], 1
        )
        for feature, network in enumerate(self.mlp_cpd):
            weights = adjacency[:, feature]
            context = (parent_embeddings * weights[None, :, None]).sum(dim=1)
            context = context / weights.sum().clamp_min(1.0)
            context = context[:, None, :].expand(-1, self.n_classes, -1)
            classes = self.cls_emb.weight[None, :, :].expand(X.shape[0], -1, -1)
            variable = self.var_q.weight[feature][None, None, :].expand(
                X.shape[0], self.n_classes, -1
            )
            logits = network(torch.cat([context, classes, variable], dim=2))
            logits = logits + self.base_logit[feature][None, :, :]
            logp = logp + logits.log_softmax(-1)[rows, :, X[:, feature]]
        return logp

    def forward(self, X: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        adjacency = self.adjacency() if adjacency is None else adjacency
        if self.cpd_type == "attention":
            return self._attention_log_joint(X, adjacency)
        return self._mlp_log_joint(X, adjacency)

    def loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        penalty_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rows = torch.arange(X.shape[0], device=X.device)
        adjacency = self.adjacency()
        log_joint = self(X, adjacency)
        joint = -log_joint[rows, y].mean()
        conditional = -log_joint.log_softmax(1)[rows, y].mean()
        sparse = adjacency.sum()
        degree = F.relu(adjacency.sum(0) - self.max_parents).square().sum()
        objective = (
            self.hybrid_weight * joint
            + (1.0 - self.hybrid_weight) * conditional
            + penalty_scale * self.lambda_sparse * sparse
            + penalty_scale * self.lambda_degree * degree
        )
        return objective, {
            "joint_nll": float(joint.detach().cpu()),
            "conditional_nll": float(conditional.detach().cpu()),
            "sparse_penalty": float(sparse.detach().cpu()),
            "degree_penalty": float(degree.detach().cpu()),
            "loss": float(objective.detach().cpu()),
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_validation: np.ndarray | None = None,
        y_validation: np.ndarray | None = None,
    ) -> "DBNC":
        device = self.cls_logit.device
        features = torch.as_tensor(X, dtype=torch.long)
        labels = torch.as_tensor(y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(features, labels),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        best_accuracy = -float("inf")
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(1, self.epochs + 1):
            self.train()
            penalty_scale = 0.0 if epoch <= self.structure_warmup_epochs else 1.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                objective, _ = self.loss(xb, yb, penalty_scale=penalty_scale)
                if not torch.isfinite(objective):
                    raise ExperimentError("DBNC encountered a non-finite training loss")
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
            self.epochs_trained = epoch
            if X_validation is None or y_validation is None:
                self.best_epoch = epoch
                continue
            probabilities = self.predict_proba(X_validation)
            validation_accuracy = float(
                accuracy_score(y_validation, probabilities.argmax(axis=1))
            )
            validation_loss = float(
                log_loss(y_validation, probabilities, labels=np.arange(self.n_classes))
            )
            can_checkpoint = epoch >= min(self.min_epochs_before_checkpoint, self.epochs)
            if not can_checkpoint:
                stale = 0
                continue
            if validation_checkpoint_improved(
                validation_accuracy,
                validation_loss,
                best_accuracy,
                best_loss,
            ):
                best_accuracy = validation_accuracy
                best_loss = validation_loss
                best_state = copy.deepcopy(self.state_dict())
                self.best_epoch = epoch
                self.best_validation_accuracy = validation_accuracy
                self.best_validation_log_loss = validation_loss
                stale = 0
            else:
                stale += 1
                can_stop = epoch >= min(self.min_epochs_before_stopping, self.epochs)
                if stale >= self.early_stopping_rounds and can_stop:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        training = self.training
        self.eval()
        try:
            device = self.cls_logit.device
            loader = DataLoader(
                TensorDataset(torch.as_tensor(X, dtype=torch.long)),
                batch_size=self.batch_size,
                shuffle=False,
            )
            output: list[np.ndarray] = []
            for (features,) in loader:
                probabilities = self(features.to(device)).softmax(dim=1)
                output.append(probabilities.detach().cpu().numpy())
            return np.concatenate(output, axis=0)
        finally:
            self.train(training)

    def graph_diagnostics(self) -> dict[str, Any]:
        adjacency = self.adjacency().detach().cpu().numpy()
        thresholded = adjacency > GRAPH_THRESHOLD
        incoming = thresholded.sum(axis=0)
        return {
            "edge_mass": float(adjacency.sum()),
            "incoming_edge_mass": adjacency.sum(axis=0).tolist(),
            "threshold": GRAPH_THRESHOLD,
            "thresholded_edge_density": float(thresholded.sum() / max(1, self.m * (self.m - 1))),
            "parent_limit_violation_rate": float(np.mean(incoming > self.max_parents)),
            "acyclic": bool(is_acyclic(thresholded)),
        }
