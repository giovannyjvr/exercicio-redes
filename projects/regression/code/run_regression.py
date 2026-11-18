"""Entrega 2 do projeto — regressão com rede neural. Ponto de entrada único.

Executa as etapas 0 a 7 da rubrica, grava as figuras numeradas em ``figures/`` e
**todos** os números citados no relatório em ``results/metrics.json``.

Uso, a partir da raiz do repositório::

    python docs/projects/regression/code/run_regression.py

A pipeline de pré-processamento **não é reescrita aqui**: ela é importada de
``docs/projects/eda/code/nyc_preprocessing.py``, que é a entrega do EDA.

Sementes: 42 em tudo que sorteia. A avaliação final repete com 42, 43 e 44.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EDA_CODE = BASE.parent / "eda" / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EDA_CODE))

# A pipeline do EDA, importada — não copiada.
from nyc_preprocessing import (  # noqa: E402
    CAT_HIGH, CAT_LOW, FLAG_COLUMNS, NUM_HEAVY, NUM_PLAIN,
    build_preprocessor, drop_invalid_target, feature_names, load_dataset, verify,
)
from nyc_regressor import (  # noqa: E402
    MLP, RANDOM_STATE, TrainConfig, log_scale_metrics, permutation_importance,
    predict_dollars, regression_metrics, set_seed, train_model,
)

FIGURES = BASE / "figures"
RESULTS = BASE / "results"
DPI = 140

TEST_SIZE = 0.20          # igual ao EDA
VAL_SIZE = 0.20           # a validação sai de dentro do treino
FINAL_SEEDS = (42, 43, 44)

#: Orçamento de épocas da busca e do retreino final. Constantes para que o
#: teste de fumaça possa reduzi-las sem tocar na lógica.
SEARCH_MAX_EPOCHS = 150
SEARCH_PATIENCE = 12

#: Ponto de corte do Huber, na escala em que a rede treina (log1p).
#: O resíduo típico em log fica perto de 0,4; um delta de 1,0 deixaria quase
#: tudo na região quadrática e o Huber viraria um MSE disfarçado. Com 0,5 a
#: parte linear passa a valer para erros acima de um fator e^0,5 ~ 1,65x.
HUBER_DELTA = 0.5

#: Critério de sucesso, escrito ANTES de treinar (etapa 0 da rubrica).
SUCCESS_CRITERION = {
    "main_metric": "MAE em dólares no conjunto de teste",
    "rule": "MAE ao menos 15 % menor que o do melhor baseline",
    "threshold_ratio": 0.85,
}

EXPECTED_FIGURES = [
    "figure_01_learning_curves.png",
    "figure_02_regularization.png",
    "figure_03_hyperparameter_search.png",
    "figure_04_predicted_vs_actual.png",
    "figure_05_residuals.png",
    "figure_06_error_by_subgroup.png",
    "figure_07_permutation_importance.png",
]


def save_figure(fig: plt.Figure, filename: str, *, tight: bool = True) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    figura gravada: {filename}")
    return filename


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ====================================================== Etapa 0 — ponto de partida


def etapa_0() -> dict[str, Any]:
    """Dados, split idêntico ao do EDA, pipeline importada e critério de sucesso."""
    section("Etapa 0 — ponto de partida")

    print(f"  critério de sucesso (fixado ANTES do treino): {SUCCESS_CRITERION['rule']}")
    print(f"  métrica principal: {SUCCESS_CRITERION['main_metric']}")

    df, target_filter = drop_invalid_target(load_dataset())
    features = df.drop(columns=["price", "id", "host_id", "name", "host_name"])
    target = df["price"]

    # Split externo: EXATAMENTE o do EDA (mesma proporção, mesma semente).
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Validação sai de dentro do treino, como a rubrica exige.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # A pipeline aprende SÓ no sub-treino. Ajustá-la nos 39 107 do treino
    # cheio deixaria a validação participar da média e do desvio da escala —
    # vazamento pequeno, mas que contamina justamente o conjunto usado para
    # escolher hiperparâmetros.
    prep = build_preprocessor(X_train)
    Z_train = prep.fit_transform(X_train)
    Z_val = prep.transform(X_val)
    Z_test = prep.transform(X_test)
    checks = verify(prep, Z_train, Z_val)
    verify(prep, Z_train, Z_test)
    names = [str(n) for n in feature_names(prep)]

    # O alvo vai para log1p: o EDA mediu que 2,14 % dos anúncios respondem por
    # 86,1 % da soma dos desvios quadráticos na escala original.
    y_train_log = np.log1p(y_train.to_numpy(dtype=float))
    y_val_log = np.log1p(y_val.to_numpy(dtype=float))
    y_test_log = np.log1p(y_test.to_numpy(dtype=float))

    print(f"  treino {Z_train.shape} | validação {Z_val.shape} | teste {Z_test.shape}")
    print(f"  NaN: {checks['nan_train']} / {checks['nan_test']}")
    print(f"  |média| das colunas escaladas — treino {checks['max_abs_mean_train_scaled']:.2e},"
          f" validação {checks['max_abs_mean_test_scaled']:.4f}"
          f"  <- validação != 0 prova que a escala não a viu")
    print(f"  price treino: média {y_train.mean():.2f}, mediana {y_train.median():.2f}")

    return {
        "target_filter": target_filter,
        "split": {
            "test_size": TEST_SIZE, "val_size": VAL_SIZE, "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
            "n_train_full": int(len(X_train_full)),
            "train_price_mean": float(y_train.mean()),
            "train_price_median": float(y_train.median()),
            "val_price_mean": float(y_val.mean()),
            "test_price_mean": float(y_test.mean()),
        },
        "pipeline_checks": checks,
        "success_criterion": SUCCESS_CRITERION,
        "n_features": int(Z_train.shape[1]),
        "_data": {
            "Z_train": Z_train, "Z_val": Z_val, "Z_test": Z_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "y_train_log": y_train_log, "y_val_log": y_val_log, "y_test_log": y_test_log,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "names": names, "prep": prep,
        },
    }


# ============================================================ Etapa 1 — baselines


def etapa_1(data: dict[str, Any]) -> dict[str, Any]:
    """Baselines: trivial, por grupo, linear e um de árvores (opcional)."""
    section("Etapa 1 — baselines (avaliados na validação)")

    Z_train, Z_val = data["Z_train"], data["Z_val"]
    y_train, y_val = data["y_train"], data["y_val"]
    y_train_log = data["y_train_log"]
    X_train, X_val = data["X_train"], data["X_val"]
    y_val_dollars = y_val.to_numpy(dtype=float)

    results: dict[str, dict[str, Any]] = {}

    # (a) trivial — média do treino, em dólares.
    dummy_mean = DummyRegressor(strategy="mean").fit(Z_train, y_train)
    results["Dummy (média)"] = regression_metrics(y_val_dollars, dummy_mean.predict(Z_val))

    # (b) trivial — mediana do treino. Com alvo assimétrico costuma ganhar da média.
    dummy_median = DummyRegressor(strategy="median").fit(Z_train, y_train)
    results["Dummy (mediana)"] = regression_metrics(y_val_dollars, dummy_median.predict(Z_val))

    # (c) o baseline que o próprio EDA propôs: mediana por room_type x borough.
    key = ["room_type", "neighbourhood_group"]
    lookup = y_train.groupby([X_train[k] for k in key]).median()
    global_median = float(y_train.median())
    pairs = pd.MultiIndex.from_arrays([X_val[k] for k in key])
    group_pred = pd.Series(pairs.map(lookup)).astype(float).fillna(global_median).to_numpy()
    results["Mediana por room_type × borough"] = regression_metrics(y_val_dollars, group_pred)

    # (d) linear — Ridge sobre log1p, revertido com expm1.
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(Z_train, y_train_log)
    results["Ridge (log1p)"] = regression_metrics(y_val_dollars, np.expm1(ridge.predict(Z_val)))

    # (e) opcional — boosting. A rubrica avisa que costuma empatar com a rede.
    boost = HistGradientBoostingRegressor(random_state=RANDOM_STATE).fit(Z_train, y_train_log)
    results["HistGradientBoosting (log1p)"] = regression_metrics(
        y_val_dollars, np.expm1(boost.predict(Z_val))
    )

    for name, m in sorted(results.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {name:<34} MAE {m['mae']:>7.2f} | RMSE {m['rmse']:>7.2f} | R² {m['r2']:>7.4f}")

    # O melhor baseline é escolhido ENTRE OS NÃO-REDE, na validação.
    best_name = min(results, key=lambda k: results[k]["mae"])
    best_mae = results[best_name]["mae"]
    target_mae = best_mae * SUCCESS_CRITERION["threshold_ratio"]
    print(f"\n  melhor baseline: {best_name} (MAE {best_mae:.2f})")
    print(f"  alvo da rede para cumprir o critério: MAE <= {target_mae:.2f}")

    return {
        "results": results,
        "best_name": best_name,
        "best_mae_val": best_mae,
        "target_mae": target_mae,
        "_models": {"ridge": ridge, "boost": boost, "dummy_mean": dummy_mean,
                    "dummy_median": dummy_median, "group_lookup": lookup,
                    "global_median": global_median},
    }


# ======================================================== Etapas 2 e 3 — a rede


def etapa_2_3(data: dict[str, Any]) -> dict[str, Any]:
    """A rede de referência e o laço de treino com early stopping."""
    section("Etapas 2 e 3 — a rede e o treino")

    Z_train, Z_val = data["Z_train"], data["Z_val"]
    y_train_log, y_val_log = data["y_train_log"], data["y_val_log"]
    y_val_dollars = data["y_val"].to_numpy(dtype=float)

    set_seed(RANDOM_STATE)
    model = MLP(Z_train.shape[1], hidden=(256, 128))
    config = TrainConfig(learning_rate=1e-3, batch_size=256, max_epochs=200, patience=15)

    print(f"  arquitetura {model.architecture()} | {model.n_parameters()} parâmetros"
          f" para {len(Z_train)} linhas de treino"
          f" (razão {model.n_parameters() / len(Z_train):.2f})")

    start = time.perf_counter()
    run = train_model(model, Z_train, y_train_log, Z_val, y_val_log, config, verbose=True)
    elapsed = time.perf_counter() - start

    val_metrics = regression_metrics(y_val_dollars, predict_dollars(model, Z_val))
    print(f"  parou na época {run['epochs_run']}, melhor época {run['best_epoch']}"
          f" ({elapsed:.1f} s)")
    print(f"  validação: MAE {val_metrics['mae']:.2f} | RMSE {val_metrics['rmse']:.2f}"
          f" | R² {val_metrics['r2']:.4f}")

    # --- Figura 1: curvas de treino e validação ------------------------------
    history = run["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes[0].plot(epochs, history["train_loss"], color="#1f77b4", lw=1.8, label="Treino")
    axes[0].plot(epochs, history["val_loss"], color="#d62728", lw=1.8, label="Validação")
    axes[0].axvline(run["best_epoch"], color="grey", ls="--", lw=1.2,
                    label=f"melhor época = {run['best_epoch']}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda (MSE sobre log1p(price))")
    axes[0].set_title("Perda por época")
    axes[0].legend()
    axes[0].grid(alpha=0.25, linestyle=":")

    axes[1].plot(epochs, history["val_mae_log"], color="#2ca02c", lw=1.8,
                 label="MAE na validação (escala log)")
    axes[1].axvline(run["best_epoch"], color="grey", ls="--", lw=1.2)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE em log1p(price)")
    axes[1].set_title("Erro absoluto na validação")
    axes[1].legend()
    axes[1].grid(alpha=0.25, linestyle=":")
    fig.suptitle(f"Figura 1 — Treino da rede de referência {model.architecture()}"
                 f" (early stopping, paciência {config.patience})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig1 = save_figure(fig, EXPECTED_FIGURES[0])

    gap = history["val_loss"][run["best_epoch"] - 1] - history["train_loss"][run["best_epoch"] - 1]

    return {
        "architecture": model.architecture(),
        "hidden": list(model.hidden),
        "n_parameters": model.n_parameters(),
        "params_per_row": float(model.n_parameters() / len(Z_train)),
        "config": run["config"],
        "best_epoch": run["best_epoch"],
        "epochs_run": run["epochs_run"],
        "stopped_early": run["stopped_early"],
        "best_val_loss": run["best_val_loss"],
        "train_loss_at_best": history["train_loss"][run["best_epoch"] - 1],
        "val_minus_train_loss_at_best": float(gap),
        "val_metrics": val_metrics,
        "history": history,
        "seconds": elapsed,
        "figures": [fig1],
    }


# ============================ Etapa 4 — regularização e busca de hiperparâmetros


def _fit_and_score(
    data: dict[str, Any], hidden: tuple[int, ...], *, dropout: float, batch_norm: bool,
    weight_decay: float, learning_rate: float, loss: str = "mse",
    huber_delta: float = HUBER_DELTA, seed: int = RANDOM_STATE,
    max_epochs: int | None = None, patience: int | None = None,
) -> dict[str, Any]:
    """Treina uma configuração e devolve o desempenho **na validação**."""
    set_seed(seed)
    model = MLP(data["Z_train"].shape[1], hidden=hidden, dropout=dropout, batch_norm=batch_norm)
    config = TrainConfig(
        learning_rate=learning_rate, weight_decay=weight_decay, loss=loss,
        huber_delta=huber_delta,
        max_epochs=SEARCH_MAX_EPOCHS if max_epochs is None else max_epochs,
        patience=SEARCH_PATIENCE if patience is None else patience,
        seed=seed,
    )
    start = time.perf_counter()
    run = train_model(model, data["Z_train"], data["y_train_log"],
                      data["Z_val"], data["y_val_log"], config)
    elapsed = time.perf_counter() - start
    val = regression_metrics(data["y_val"].to_numpy(dtype=float),
                             predict_dollars(model, data["Z_val"]))
    train = regression_metrics(data["y_train"].to_numpy(dtype=float),
                               predict_dollars(model, data["Z_train"]))
    return {
        "hidden": list(hidden), "dropout": dropout, "batch_norm": batch_norm,
        "weight_decay": weight_decay, "learning_rate": learning_rate, "loss": loss,
        "huber_delta": huber_delta if loss == "huber" else None,
        "n_parameters": model.n_parameters(), "best_epoch": run["best_epoch"],
        "epochs_run": run["epochs_run"],
        "val_mae": val["mae"], "val_rmse": val["rmse"], "val_r2": val["r2"],
        "train_mae": train["mae"],
        "overfit_gap_mae": float(val["mae"] - train["mae"]),
        "seconds": elapsed,
    }


def etapa_4(data: dict[str, Any]) -> dict[str, Any]:
    """Regularização comparada e busca de 12 configurações, decididas na validação."""
    section("Etapa 4 — regularização e hiperparâmetros (tudo na validação)")

    base_hidden = (256, 128)

    # --- (a) a função de perda ------------------------------------------------
    # A rubrica pede a perda "justificada pelos outliers que o EDA encontrou".
    # O EDA mediu que 2,14 % dos anúncios respondem por 86,1 % da soma dos
    # desvios quadráticos; o log1p já absorve a maior parte disso, mas quanto
    # sobra é pergunta empírica — então as três perdas são comparadas, em vez
    # de uma ser escolhida no texto.
    print("  (a) função de perda: MSE contra Huber e MAE (na validação)")
    losses = []
    for loss_name in ("mse", "huber", "mae"):
        row = _fit_and_score(data, base_hidden, dropout=0.0, batch_norm=False,
                             weight_decay=0.0, learning_rate=1e-3, loss=loss_name)
        row["name"] = loss_name
        losses.append(row)
        delta = f" (delta {HUBER_DELTA})" if loss_name == "huber" else ""
        print(f"    {loss_name + delta:<20} MAE val {row['val_mae']:>7.2f}"
              f" | RMSE {row['val_rmse']:>7.2f} | R² {row['val_r2']:>6.4f}"
              f" | época {row['best_epoch']:>3} | {row['seconds']:.0f} s")

    chosen_loss = min(losses, key=lambda r: r["val_mae"])["name"]
    print(f"    perda escolhida na validação: {chosen_loss}")

    print("\n  (b) regularização: três técnicas contra a rede sem nenhuma")
    reg_specs = [
        ("sem regularização", dict(dropout=0.0, batch_norm=False, weight_decay=0.0)),
        ("dropout 0.2", dict(dropout=0.2, batch_norm=False, weight_decay=0.0)),
        ("weight decay 1e-4", dict(dropout=0.0, batch_norm=False, weight_decay=1e-4)),
        ("batch norm", dict(dropout=0.0, batch_norm=True, weight_decay=0.0)),
        ("dropout 0.2 + wd 1e-4", dict(dropout=0.2, batch_norm=False, weight_decay=1e-4)),
    ]
    regularization = []
    for name, kwargs in reg_specs:
        row = _fit_and_score(data, base_hidden, learning_rate=1e-3,
                             loss=chosen_loss, **kwargs)
        row["name"] = name
        regularization.append(row)
        print(f"    {name:<24} MAE val {row['val_mae']:>7.2f} | treino {row['train_mae']:>7.2f}"
              f" | distância {row['overfit_gap_mae']:>6.2f} | época {row['best_epoch']:>3}"
              f" | {row['seconds']:.0f} s")

    print("\n  (c) busca de 12 configurações")
    grid = []
    for hidden in [(128,), (256, 128), (512, 256, 128)]:
        for learning_rate in [1e-3, 3e-4]:
            for reg_name, reg in [("sem reg.", dict(dropout=0.0, weight_decay=0.0)),
                                  ("dropout+wd", dict(dropout=0.2, weight_decay=1e-4))]:
                grid.append((hidden, learning_rate, reg_name, reg))

    search = []
    for hidden, learning_rate, reg_name, reg in grid:
        row = _fit_and_score(data, hidden, batch_norm=False,
                             learning_rate=learning_rate, loss=chosen_loss, **reg)
        row["label"] = (f"{'–'.join(map(str, hidden))} | lr {learning_rate:g} | {reg_name}")
        search.append(row)
        print(f"    {row['label']:<38} MAE val {row['val_mae']:>7.2f}"
              f" | R² {row['val_r2']:>6.4f} | época {row['best_epoch']:>3}"
              f" | {row['seconds']:.0f} s")

    search.sort(key=lambda r: r["val_mae"])
    best = search[0]
    print(f"\n  melhor configuração na validação: {best['label']} (MAE {best['val_mae']:.2f})")

    # --- Figura 2: regularização ---------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    names = [r["name"] for r in regularization]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["train_mae"] for r in regularization], 0.4,
           label="MAE no treino", color="#1f77b4")
    ax.bar(x + 0.2, [r["val_mae"] for r in regularization], 0.4,
           label="MAE na validação", color="#d62728")
    for i, r in enumerate(regularization):
        ax.text(i + 0.2, r["val_mae"], f"{r['val_mae']:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("MAE (US$)")
    ax.set_title(f"Figura 2 — Regularização na rede {'–'.join(map(str, base_hidden))}\n"
                 "(a distância entre as barras é o overfitting)")
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":", axis="y")
    fig2 = save_figure(fig, EXPECTED_FIGURES[1])

    # --- Figura 3: busca -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    labels = [r["label"] for r in search][::-1]
    values = [r["val_mae"] for r in search][::-1]
    colors = ["#2ca02c" if i == len(values) - 1 else "#7f7f7f" for i in range(len(values))]
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=9)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("MAE na validação (US$) — menor é melhor")
    ax.set_title("Figura 3 — As 12 configurações, ordenadas pela validação\n"
                 "(verde: a escolhida)")
    ax.grid(alpha=0.25, linestyle=":", axis="x")
    fig3 = save_figure(fig, EXPECTED_FIGURES[2])

    return {
        "losses": losses,
        "chosen_loss": chosen_loss,
        "huber_delta": HUBER_DELTA,
        "regularization": regularization,
        "search": search,
        "n_configurations": len(search),
        "best": best,
        "figures": [fig2, fig3],
    }


# ================================================== Etapa 5 — avaliação no teste


def etapa_5(data: dict[str, Any], best: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    """A configuração escolhida, retreinada com 3 sementes e avaliada UMA vez."""
    section("Etapa 5 — avaliação no teste (o teste é tocado aqui, uma vez)")

    y_test_dollars = data["y_test"].to_numpy(dtype=float)
    y_val_dollars = data["y_val"].to_numpy(dtype=float)
    hidden = tuple(best["hidden"])

    runs, models = [], []
    for seed in FINAL_SEEDS:
        set_seed(seed)
        model = MLP(data["Z_train"].shape[1], hidden=hidden,
                    dropout=best["dropout"], batch_norm=best["batch_norm"])
        config = TrainConfig(
            learning_rate=best["learning_rate"], weight_decay=best["weight_decay"],
            # A perda é a escolhida na etapa 4; sem isto o retreino final
            # voltaria silenciosamente ao MSE padrão do TrainConfig.
            loss=best["loss"], huber_delta=best["huber_delta"] or HUBER_DELTA,
            max_epochs=SEARCH_MAX_EPOCHS, patience=SEARCH_PATIENCE, seed=seed,
        )
        run = train_model(model, data["Z_train"], data["y_train_log"],
                          data["Z_val"], data["y_val_log"], config)
        test_metrics = regression_metrics(y_test_dollars, predict_dollars(model, data["Z_test"]))
        test_log = log_scale_metrics(data["y_test_log"],
                                     np.log1p(np.maximum(predict_dollars(model, data["Z_test"]), 0)))
        train_metrics = regression_metrics(data["y_train"].to_numpy(dtype=float),
                                           predict_dollars(model, data["Z_train"]))
        val_metrics = regression_metrics(y_val_dollars, predict_dollars(model, data["Z_val"]))
        runs.append({
            "seed": seed, "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
            "test": test_metrics, "test_log": test_log,
            "train": train_metrics, "val": val_metrics,
        })
        models.append(model)
        print(f"  semente {seed}: MAE teste {test_metrics['mae']:.2f}"
              f" | RMSE {test_metrics['rmse']:.2f} | R² {test_metrics['r2']:.4f}"
              f" | parou na época {run['epochs_run']}")

    def agg(path: str) -> dict[str, float]:
        values = [r["test"][path] for r in runs]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    network = {k: agg(k) for k in ("mae", "rmse", "r2", "mape", "median_ae")}
    print(f"  rede (3 sementes): MAE {network['mae']['mean']:.2f} ± {network['mae']['std']:.2f}"
          f" | RMSE {network['rmse']['mean']:.2f} | R² {network['r2']['mean']:.4f}")

    # --- os baselines, agora no mesmo conjunto de teste ----------------------
    m = baselines["_models"]
    Z_test = data["Z_test"]
    X_test = data["X_test"]
    pairs = pd.MultiIndex.from_arrays([X_test["room_type"], X_test["neighbourhood_group"]])
    group_pred = pd.Series(pairs.map(m["group_lookup"])).astype(float)
    group_pred = group_pred.fillna(m["global_median"]).to_numpy()

    baseline_test = {
        "Dummy (média)": regression_metrics(y_test_dollars, m["dummy_mean"].predict(Z_test)),
        "Dummy (mediana)": regression_metrics(y_test_dollars, m["dummy_median"].predict(Z_test)),
        "Mediana por room_type × borough": regression_metrics(y_test_dollars, group_pred),
        "Ridge (log1p)": regression_metrics(y_test_dollars, np.expm1(m["ridge"].predict(Z_test))),
        "HistGradientBoosting (log1p)": regression_metrics(
            y_test_dollars, np.expm1(m["boost"].predict(Z_test))),
    }
    for name, met in sorted(baseline_test.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {name:<34} MAE teste {met['mae']:>7.2f} | R² {met['r2']:>7.4f}")

    # O baseline de referência do critério é o que venceu NA VALIDAÇÃO (etapa 1).
    # Escolhê-lo pelo teste seria usar o teste para decidir algo, que é
    # exatamente o que a regra da rubrica proíbe.
    best_baseline_name = baselines["best_name"]
    best_baseline_mae = baseline_test[best_baseline_name]["mae"]
    improvement = (best_baseline_mae - network["mae"]["mean"]) / best_baseline_mae
    gap = best_baseline_mae - network["mae"]["mean"]
    criterion_met = bool(improvement >= 1 - SUCCESS_CRITERION["threshold_ratio"])

    # Comparação secundária: contra o melhor dos baselines que a rubrica EXIGE
    # (trivial e linear). O boosting entrou como opcional.
    required = {k: v for k, v in baseline_test.items() if "Boosting" not in k}
    best_required_name = min(required, key=lambda k: required[k]["mae"])
    best_required_mae = required[best_required_name]["mae"]
    improvement_required = (best_required_mae - network["mae"]["mean"]) / best_required_mae

    print(f"\n  baseline de referência (escolhido na validação): {best_baseline_name}")
    print(f"    MAE dele no teste: {best_baseline_mae:.2f}")
    print(f"  rede: {network['mae']['mean']:.2f} -> redução de {improvement * 100:.2f} %")
    print(f"  diferença {gap:.2f} contra desvio entre sementes {network['mae']['std']:.2f}"
          f" ({gap / max(network['mae']['std'], 1e-9):.1f}x)")
    print(f"  CRITÉRIO (>= 15 %): {'CUMPRIDO' if criterion_met else 'NÃO CUMPRIDO'}")
    print(f"  contra o melhor baseline EXIGIDO ({best_required_name}, MAE {best_required_mae:.2f}):"
          f" redução de {improvement_required * 100:.2f} %")

    # Modelo representativo: o da semente 42, usado nas figuras e na etapa 6.
    model = models[0]
    y_pred = predict_dollars(model, Z_test)

    # --- Figura 4: previsto x real -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    limit = float(np.percentile(y_test_dollars, 99))
    axes[0].scatter(y_test_dollars, y_pred, s=4, alpha=0.18, color="#1f77b4", edgecolors="none")
    axes[0].plot([0, limit], [0, limit], color="#d62728", lw=1.6, ls="--", label="previsão perfeita")
    axes[0].set_xlim(0, limit); axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("price real (US$)")
    axes[0].set_ylabel("price previsto (US$)")
    axes[0].set_title(f"Escala original (eixos até o percentil 99 = {limit:.0f})")
    axes[0].legend(); axes[0].grid(alpha=0.2, linestyle=":")

    axes[1].scatter(np.log1p(y_test_dollars), np.log1p(np.maximum(y_pred, 0)),
                    s=4, alpha=0.18, color="#2ca02c", edgecolors="none")
    lo, hi = np.log1p(y_test_dollars).min(), np.log1p(y_test_dollars).max()
    axes[1].plot([lo, hi], [lo, hi], color="#d62728", lw=1.6, ls="--")
    axes[1].set_xlabel("log(1 + price real)")
    axes[1].set_ylabel("log(1 + price previsto)")
    axes[1].set_title("Escala de treino (log1p)")
    axes[1].grid(alpha=0.2, linestyle=":")
    fig.suptitle(f"Figura 4 — Previsto contra real no teste (semente {FINAL_SEEDS[0]},"
                 f" {len(y_test_dollars)} anúncios)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig4 = save_figure(fig, EXPECTED_FIGURES[3])

    # --- Figura 5: resíduos ----------------------------------------------------
    residual = y_test_dollars - y_pred
    residual_log = data["y_test_log"] - np.log1p(np.maximum(y_pred, 0))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    axes[0].scatter(y_pred, residual, s=4, alpha=0.18, color="#1f77b4", edgecolors="none")
    axes[0].axhline(0, color="#d62728", lw=1.4, ls="--")
    axes[0].set_xlim(0, float(np.percentile(y_pred, 99.5)))
    axes[0].set_xlabel("price previsto (US$)")
    axes[0].set_ylabel("resíduo  y − ŷ  (US$)")
    axes[0].set_title("Resíduo contra previsto — escala original")
    axes[0].grid(alpha=0.2, linestyle=":")

    axes[1].scatter(np.log1p(np.maximum(y_pred, 0)), residual_log,
                    s=4, alpha=0.18, color="#9467bd", edgecolors="none")
    axes[1].axhline(0, color="#d62728", lw=1.4, ls="--")
    axes[1].set_xlabel("log(1 + price previsto)")
    axes[1].set_ylabel("resíduo em log")
    axes[1].set_title("Resíduo contra previsto — escala de treino")
    axes[1].grid(alpha=0.2, linestyle=":")
    fig.suptitle("Figura 5 — Resíduos no conjunto de teste", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig5 = save_figure(fig, EXPECTED_FIGURES[4])

    return {
        "seeds": list(FINAL_SEEDS),
        "runs": runs,
        "network_test": network,
        "baselines_test": baseline_test,
        "best_baseline_name": best_baseline_name,
        "best_baseline_mae": best_baseline_mae,
        "best_required_baseline_name": best_required_name,
        "best_required_baseline_mae": best_required_mae,
        "improvement_vs_required": float(improvement_required),
        "improvement_ratio": float(improvement),
        "gap_vs_std": float(gap / max(network["mae"]["std"], 1e-9)),
        "criterion_met": criterion_met,
        "chosen_config": {k: best[k] for k in
                          ("hidden", "dropout", "batch_norm", "weight_decay",
                           "learning_rate", "loss", "huber_delta",
                           "n_parameters", "label")},
        "figures": [fig4, fig5],
        "_model": model,
        "_y_pred": y_pred,
    }


# =================================================== Etapa 6 — análise de erro


def etapa_6(data: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """Erro por subgrupo, piores exemplos e importância por permutação."""
    section("Etapa 6 — análise de erro")

    y_true = data["y_test"].to_numpy(dtype=float)
    y_pred = final["_y_pred"]
    model = final["_model"]
    X_test = data["X_test"].reset_index(drop=True)
    error = np.abs(y_true - y_pred)

    frame = X_test.copy()
    frame["price"] = y_true
    frame["pred"] = y_pred
    frame["abs_error"] = error

    subgroups: dict[str, Any] = {}
    for col in ["room_type", "neighbourhood_group"]:
        grouped = frame.groupby(col)
        table = pd.DataFrame({
            "n": grouped.size(),
            "mae": grouped["abs_error"].mean(),
            "mediana_price": grouped["price"].median(),
        }).sort_values("mae", ascending=False)
        subgroups[col] = json.loads(table.round(4).to_json(orient="index"))
        print(f"\n  MAE por {col}:")
        print(table.round(2).to_string())

    # Faixas do alvo — quartis do teste, para não inventar cortes.
    edges = [0, 69, 106, 175, np.inf]
    labels = ["até 69", "69–106", "106–175", "acima de 175"]
    frame["faixa"] = pd.cut(frame["price"], bins=edges, labels=labels, right=True)
    grouped = frame.groupby("faixa", observed=True)
    band = pd.DataFrame({
        "n": grouped.size(),
        "mae": grouped["abs_error"].mean(),
        "mape": grouped.apply(
            lambda g: float(np.mean(np.abs((g["price"] - g["pred"]) / g["price"])) * 100),
            include_groups=False),
        "vies_medio": grouped.apply(
            lambda g: float(np.mean(g["pred"] - g["price"])), include_groups=False),
    })
    subgroups["faixa_de_preco"] = json.loads(band.round(4).to_json(orient="index"))
    print("\n  MAE por faixa de price (quartis do treino):")
    print(band.round(2).to_string())

    worst = frame.nlargest(10, "abs_error")[
        ["price", "pred", "abs_error", "room_type", "neighbourhood_group", "neighbourhood",
         "minimum_nights", "availability_365"]
    ]
    print("\n  10 maiores erros absolutos:")
    print(worst.round(2).to_string())

    # --- Figura 6: erro por subgrupo ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (key, title) in zip(axes, [
        ("room_type", "MAE por room_type"),
        ("neighbourhood_group", "MAE por borough"),
        ("faixa_de_preco", "MAE por faixa de price"),
    ]):
        table = subgroups[key]
        keys = list(table.keys())
        values = [table[k]["mae"] for k in keys]
        counts = [table[k]["n"] for k in keys]
        bars = ax.bar(range(len(keys)), values, color="#1f77b4")
        for bar, v, c in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}\n(n={int(c)})", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(keys)), keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("MAE (US$)")
        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle=":", axis="y")
    fig.suptitle("Figura 6 — Erro absoluto médio por subgrupo, no conjunto de teste",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig6 = save_figure(fig, EXPECTED_FIGURES[5])

    # --- importância por permutação, na VALIDAÇÃO ----------------------------
    # Na validação de propósito: o teste já foi usado na etapa 5 e não deve
    # alimentar mais nenhuma análise que possa virar decisão.
    names = data["names"]
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        block, _, rest = str(name).partition("__")
        if block == "hicard":
            key = "neighbourhood"
        elif block == "cat":
            # rest é "<coluna>_<categoria>"; partir em "_" quebraria
            # "neighbourhood_group" ao meio, então casa pelo prefixo conhecido.
            key = next((c for c in CAT_LOW if rest.startswith(c + "_")), rest)
        else:
            key = rest
        groups.setdefault(key, []).append(i)

    # O rótulo diz quantas colunas cada bloco tem — é o que justifica permutar
    # o bloco inteiro em vez de uma coluna de cada vez.
    groups = {
        (f"{k} ({len(v)} colunas)" if len(v) > 1 else k): v
        for k, v in groups.items()
    }

    importance = permutation_importance(
        model, data["Z_val"], data["y_val"].to_numpy(dtype=float), names,
        n_repeats=5, seed=RANDOM_STATE, groups=groups,
    )
    print("\n  importância por permutação (validação, piora do MAE em US$):")
    for row in importance[:10]:
        print(f"    {row['feature']:<28} +{row['delta_mae_mean']:>6.2f}"
              f" ± {row['delta_mae_std']:.2f}")

    # --- Figura 7 --------------------------------------------------------------
    top = importance[:12][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(range(len(top)), [r["delta_mae_mean"] for r in top],
            xerr=[r["delta_mae_std"] for r in top], color="#ff7f0e",
            error_kw={"ecolor": "#555", "capsize": 3})
    ax.set_yticks(range(len(top)), [r["feature"] for r in top], fontsize=9)
    ax.set_xlabel("Piora do MAE ao embaralhar a feature (US$)")
    ax.set_title("Figura 7 — Importância por permutação na validação\n"
                 "(5 repetições; blocos one-hot embaralhados juntos)")
    ax.grid(alpha=0.25, linestyle=":", axis="x")
    fig7 = save_figure(fig, EXPECTED_FIGURES[6])

    return {
        "subgroups": subgroups,
        "worst_examples": json.loads(worst.round(4).to_json(orient="records")),
        "permutation_importance": importance,
        "worst_subgroup": max(
            ((k, v["mae"]) for k, v in subgroups["room_type"].items()), key=lambda kv: kv[1]
        ),
        "figures": [fig6, fig7],
    }


# ========================================================================= main


def main() -> int:
    t0 = time.perf_counter()

    metrics_0 = etapa_0()
    data = metrics_0.pop("_data")

    metrics_1 = etapa_1(data)
    metrics_2_3 = etapa_2_3(data)
    metrics_4 = etapa_4(data)
    metrics_5 = etapa_5(data, metrics_4["best"], metrics_1)
    metrics_6 = etapa_6(data, metrics_5)

    baseline_models = metrics_1.pop("_models")
    metrics_5.pop("_model"); metrics_5.pop("_y_pred")

    all_figures = (metrics_2_3["figures"] + metrics_4["figures"]
                   + metrics_5["figures"] + metrics_6["figures"])
    missing = [f for f in EXPECTED_FIGURES if not (FIGURES / f).exists()]
    if missing:
        for name in missing:
            print(f"  [FALHA] figura não gerada: {name}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    payload = {
        "random_state": RANDOM_STATE,
        "generated_in_seconds": round(elapsed, 1),
        "etapa_0": metrics_0,
        "etapa_1_baselines": metrics_1,
        "etapa_2_3_network": metrics_2_3,
        "etapa_4_tuning": metrics_4,
        "etapa_5_test": metrics_5,
        "etapa_6_errors": metrics_6,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    section("Concluído")
    print(f"  {len(all_figures)} figuras em {FIGURES}")
    print(f"  métricas em {RESULTS / 'metrics.json'}")
    print(f"  tempo total: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
