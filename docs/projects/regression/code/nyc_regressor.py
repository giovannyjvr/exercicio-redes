"""MLP de regressão para o preço de diárias em NYC — modelo, treino e métricas.

Módulo **importável**, escrito em PyTorch. A rubrica exige rede própria: o
`MLPRegressor` do scikit-learn só vale como *baseline*, e é nesse papel que ele
**não** aparece aqui — o baseline linear desta entrega é uma `Ridge`.

Uso típico::

    from nyc_regressor import MLP, TrainConfig, train_model, regression_metrics

    model = MLP(n_features=106, hidden=(256, 128), dropout=0.2, batch_norm=True)
    history, best_epoch = train_model(model, X_tr, y_tr, X_val, y_val, TrainConfig())
    metrics = regression_metrics(y_true_dollars, predict_dollars(model, X_test))

Convenção do alvo: o modelo é treinado sobre ``log1p(price)`` e **todas as
métricas voltam para dólares** com ``expm1`` antes de serem reportadas. O EDA
mostrou por quê: 2,14 % dos anúncios respondem por 86,1 % da soma dos desvios
quadráticos na escala original.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42


def set_seed(seed: int) -> None:
    """Fixa as sementes de NumPy e PyTorch antes de construir qualquer modelo."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- o modelo


class MLP(nn.Module):
    """MLP para regressão: camadas densas + ReLU, saída linear de 1 unidade.

    Parameters
    ----------
    n_features:
        Largura da entrada — 106 colunas saem da pipeline do EDA.
    hidden:
        Larguras das camadas escondidas, em ordem. ``(256, 128)`` são duas.
    dropout:
        Probabilidade de dropout **depois** de cada ativação. 0 desliga.
    batch_norm:
        Insere ``BatchNorm1d`` entre a camada linear e a ativação.

    A saída é **uma unidade linear, sem ativação**: o alvo é contínuo e pode
    assumir qualquer valor real na escala de log. Uma sigmoide ou ReLU na saída
    limitaria a imagem e truncaria as previsões altas.

    A inicialização é de **He (Kaiming) normal** nas camadas escondidas, que é a
    que corresponde à ReLU: ela mantém a variância da ativação estável ao longo
    da profundidade, porque compensa o fato de a ReLU zerar metade das entradas.
    A camada de saída usa Xavier, apropriada para uma saída linear, e todos os
    vieses começam em zero.
    """

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[int] = (256, 128),
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)

        layers: list[nn.Module] = []
        in_features = self.n_features
        for width in self.hidden:
            layers.append(nn.Linear(in_features, width))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for index, module in enumerate(linears):
            if index < len(linears) - 1:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def architecture(self) -> str:
        return "–".join(str(w) for w in (self.n_features, *self.hidden, 1))


# ------------------------------------------------------------------- o treino


@dataclass
class TrainConfig:
    """Hiperparâmetros do laço de treino.

    ``patience`` conta épocas **sem melhora na validação**; ao estourar, o laço
    para e os pesos da melhor época são restaurados.
    """

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    weight_decay: float = 0.0
    loss: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    optimizer: str = "adamw"  # "adamw" | "adam" | "sgd"
    momentum: float = 0.9  # só para sgd
    seed: int = RANDOM_STATE

    def describe(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "seed": self.seed,
        }


def _make_loss(config: TrainConfig) -> nn.Module:
    if config.loss == "mse":
        return nn.MSELoss()
    if config.loss == "huber":
        return nn.HuberLoss(delta=config.huber_delta)
    if config.loss == "mae":
        return nn.L1Loss()
    raise ValueError(f"perda desconhecida: {config.loss!r}")


def _make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate,
            momentum=config.momentum, weight_decay=config.weight_decay,
        )
    raise ValueError(f"otimizador desconhecido: {config.optimizer!r}")


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


@torch.no_grad()
def predict_log(model: MLP, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Previsão na escala de treino (log1p), em lotes para não estourar memória."""
    model.eval()
    tensor = _as_tensor(X)
    chunks = [model(tensor[i:i + batch_size]).cpu().numpy()
              for i in range(0, len(tensor), batch_size)]
    return np.concatenate(chunks) if chunks else np.empty(0)


def predict_dollars(model: MLP, X: np.ndarray) -> np.ndarray:
    """Previsão de volta na escala original, em dólares."""
    return np.expm1(predict_log(model, X))


def train_model(
    model: MLP,
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    config: TrainConfig,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Treina com *early stopping* na validação, restaurando os melhores pesos.

    Devolve o histórico por época e o estado do modelo já com os pesos da melhor
    época carregados — nunca os da última, que costuma ser pior.
    """
    set_seed(config.seed)

    loader = DataLoader(
        TensorDataset(_as_tensor(X_train), _as_tensor(y_train_log)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    loss_fn = _make_loss(config)
    optimizer = _make_optimizer(model, config)

    X_val_t, y_val_t = _as_tensor(X_val), _as_tensor(y_val_log)
    X_train_t, y_train_t = _as_tensor(X_train), _as_tensor(y_train_log)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae_log": []}
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())
            val_mae_log = float(torch.abs(val_pred - y_val_t).mean().item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_log"].append(val_mae_log)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, wait = val_loss, epoch, 0
            # Cópia em memória: mais rápido e mais limpo que gravar em disco, e
            # o efeito é o mesmo que o torch.save/torch.load do esqueleto.
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= config.patience:
                break

        if verbose and epoch % 10 == 0:
            print(f"      época {epoch:>3} | treino {train_loss:.5f} | val {val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        model.eval()
        final_train_loss = float(loss_fn(model(X_train_t), y_train_t).item())

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history["val_loss"]),
        "stopped_early": len(history["val_loss"]) < config.max_epochs,
        "train_loss_at_best": final_train_loss,
        "config": config.describe(),
        "n_parameters": model.n_parameters(),
        "architecture": model.architecture(),
    }


# ------------------------------------------------------------------ métricas


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R² e MAPE **na escala original**, em dólares.

    O MAPE é reportado porque o alvo não chega perto de zero (mínimo US$ 10
    depois de remover os `price = 0` no EDA), mas ele ainda pesa mais os
    anúncios baratos — a leitura fica junto no relatório.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
        "mape": float(np.mean(np.abs(error / y_true)) * 100.0),
        "median_ae": float(np.median(np.abs(error))),
        "n": int(len(y_true)),
    }


def log_scale_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """As mesmas medidas na escala em que a rede de fato treinou."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    error = y_true_log - y_pred_log
    ss_tot = float(np.sum((y_true_log - y_true_log.mean()) ** 2))
    return {
        "mae_log": float(np.mean(np.abs(error))),
        "rmse_log": float(np.sqrt(np.mean(error ** 2))),
        "r2_log": float(1.0 - float(np.sum(error ** 2)) / ss_tot),
    }


def permutation_importance(
    model: MLP,
    X: np.ndarray,
    y_true_dollars: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_repeats: int = 5,
    seed: int = RANDOM_STATE,
    groups: dict[str, Iterable[int]] | None = None,
) -> list[dict[str, Any]]:
    """Importância por permutação: quanto o MAE piora ao embaralhar uma coluna.

    ``groups`` permite permutar **blocos** de colunas de uma vez — necessário
    para o one-hot, em que embaralhar uma única coluna de bairro quase não muda
    nada, mas embaralhar as 89 juntas mede o efeito real da variável.
    """
    rng = np.random.default_rng(seed)
    baseline = regression_metrics(y_true_dollars, predict_dollars(model, X))["mae"]

    blocks = groups if groups is not None else {
        name: [i] for i, name in enumerate(feature_names)
    }

    rows: list[dict[str, Any]] = []
    for name, columns in blocks.items():
        columns = list(columns)
        deltas = []
        for _ in range(n_repeats):
            X_shuffled = np.array(X, copy=True)
            order = rng.permutation(len(X_shuffled))
            X_shuffled[:, columns] = X_shuffled[order][:, columns]
            mae = regression_metrics(y_true_dollars, predict_dollars(model, X_shuffled))["mae"]
            deltas.append(mae - baseline)
        rows.append({
            "feature": name,
            "n_columns": len(columns),
            "delta_mae_mean": float(np.mean(deltas)),
            "delta_mae_std": float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0,
        })

    rows.sort(key=lambda r: r["delta_mae_mean"], reverse=True)
    return rows


__all__ = [
    "RANDOM_STATE", "MLP", "TrainConfig", "set_seed", "train_model",
    "predict_log", "predict_dollars", "regression_metrics", "log_scale_metrics",
    "permutation_importance",
]
