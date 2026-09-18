"""Exercise 3 — Preparing Real-World Data for a Neural Network.

Prepara o ``train.csv`` rotulado do Spaceship Titanic para alimentar uma rede
neural com ativação ``tanh``. Nenhum modelo é treinado: o scikit-learn entra
apenas como biblioteca de *split* e pré-processamento.

A ordem das operações é a parte que importa e está deliberadamente explícita em
:func:`preprocess`:

1. ``train_test_split`` estratificado — **antes** de qualquer estatística;
2. descarte de ``Cabin``, ``Name`` e ``PassengerId``;
3. imputação (mediana nas numéricas, moda nas categóricas), ajustada só no treino;
4. ``TotalSpend`` somado **depois** da imputação dos cinco gastos;
5. ``log1p`` nos cinco gastos e no ``TotalSpend``;
6. one-hot das categóricas, com categorias aprendidas só no treino;
7. ``MinMaxScaler(feature_range=(-1, 1))`` nas numéricas, ajustado só no treino.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from common import BINARY_COLORS, DATA, save_figure

TARGET = "Transported"
DROP_COLUMNS = ["Cabin", "Name", "PassengerId"]
SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_COLUMNS = ["Age", *SPEND_COLUMNS]
CATEGORICAL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
ENGINEERED_COLUMN = "TotalSpend"

TEST_SIZE = 0.20

# As APIs do scikit-learn usadas aqui não aceitam um ``Generator`` do NumPy:
# ``train_test_split`` só entende ``random_state`` inteiro (ou um ``RandomState``
# legado). Por isso a semente 42 reaparece como inteiro neste módulo — é o mesmo
# número do ``default_rng(42)``, e nenhuma outra fonte de aleatoriedade é usada.
SPLIT_RANDOM_STATE = 42

DATASET_PATH = DATA / "train.csv"
DATASET_URL = "https://www.kaggle.com/competitions/spaceship-titanic/data"


class DatasetNotFound(FileNotFoundError):
    """Levantada quando o ``train.csv`` do Spaceship Titanic não está no repositório."""


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carrega o ``train.csv`` e confirma, pelas colunas, que é o Spaceship Titanic."""
    if not path.exists():
        raise DatasetNotFound(
            f"train.csv do Spaceship Titanic não encontrado em {path}.\n"
            f"Baixe-o em {DATASET_URL} e salve-o exatamente nesse caminho."
        )
    df = pd.read_csv(path)
    expected = {
        "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name", "Transported",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} não parece ser o train.csv do Spaceship Titanic — faltam as colunas {sorted(missing)}."
        )
    return df


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Item A — balanceamento das classes, tipos de feature, ausentes e gastos."""
    counts = df[TARGET].value_counts()
    positive = int(counts.get(True, 0))
    negative = int(counts.get(False, 0))
    total = len(df)

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean() * 100.0).to_numpy(),
        }
    ).sort_values("missing", ascending=False)

    spend_stats = {
        column: {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "max": float(df[column].max()),
        }
        for column in SPEND_COLUMNS
    }

    return {
        "n_rows": total,
        "n_columns": int(df.shape[1]),
        "class_counts": {"True": positive, "False": negative},
        "class_share": {
            "True": positive / total,
            "False": negative / total,
        },
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "identifier_features": DROP_COLUMNS,
        "missing_table": missing.to_dict(orient="records"),
        "spend_stats": spend_stats,
    }


def split(df: pd.DataFrame) -> dict[str, Any]:
    """Item B — split 80/20 estratificado pelo alvo, feito sobre os dados **brutos**."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SPLIT_RANDOM_STATE,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "share_full": float(y.mean()),
        "share_train": float(y_train.mean()),
        "share_test": float(y_test.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def preprocess(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> dict[str, Any]:
    """Item C — imputa, engenheira, transforma e escala. Tudo ajustado só no treino."""
    X_train = X_train_raw.drop(columns=DROP_COLUMNS).copy()
    X_test = X_test_raw.drop(columns=DROP_COLUMNS).copy()

    # --- 3. Imputação -----------------------------------------------------
    # Mediana nas numéricas (robusta às caudas pesadas dos gastos) e moda nas
    # categóricas. Os dois imputadores veem apenas o treino.
    numeric_imputer = SimpleImputer(strategy="median")
    X_train[NUMERIC_COLUMNS] = numeric_imputer.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = numeric_imputer.transform(X_test[NUMERIC_COLUMNS])

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    X_train[CATEGORICAL_COLUMNS] = categorical_imputer.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test[CATEGORICAL_COLUMNS] = categorical_imputer.transform(X_test[CATEGORICAL_COLUMNS])
    # Após a imputação não há mais NaN nessas colunas; virar string deixa as
    # categorias booleanas (CryoSleep, VIP) explícitas para o one-hot.
    for column in CATEGORICAL_COLUMNS:
        X_train[column] = X_train[column].astype(str)
        X_test[column] = X_test[column].astype(str)

    # --- 4. Feature engineering ------------------------------------------
    # A soma vem DEPOIS da imputação: se fosse feita antes, um passageiro com um
    # único gasto ausente produziria um TotalSpend menor do que o real (ou NaN),
    # e o erro entraria disfarçado de valor válido.
    X_train[ENGINEERED_COLUMN] = X_train[SPEND_COLUMNS].sum(axis=1)
    X_test[ENGINEERED_COLUMN] = X_test[SPEND_COLUMNS].sum(axis=1)

    # --- 5. Caudas pesadas ------------------------------------------------
    log_columns = [*SPEND_COLUMNS, ENGINEERED_COLUMN]
    for frame in (X_train, X_test):
        negatives = (frame[log_columns] < 0).to_numpy().sum()
        assert negatives == 0, "log1p exige valores >= 0; há gastos negativos nos dados"
    foodcourt_before = X_train["FoodCourt"].to_numpy(dtype=float).copy()
    for frame in (X_train, X_test):
        frame[log_columns] = np.log1p(frame[log_columns].to_numpy(dtype=float))
    foodcourt_after = X_train["FoodCourt"].to_numpy(dtype=float).copy()

    # --- 6. One-hot encoding ---------------------------------------------
    # handle_unknown="ignore": uma categoria que só apareça no teste vira uma
    # linha de zeros em vez de quebrar o transform.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    train_dummies = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    test_dummies = encoder.transform(X_test[CATEGORICAL_COLUMNS])
    dummy_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))

    # --- 7. Escalonamento -------------------------------------------------
    # MinMaxScaler para [-1, 1]: a faixa de saída coincide com a imagem da tanh,
    # então nenhuma feature nasce numa região de saturação da ativação. As
    # colunas one-hot ficam de fora e permanecem em 0/1.
    scaled_columns = [*NUMERIC_COLUMNS, ENGINEERED_COLUMN]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(X_train[scaled_columns])
    test_scaled = scaler.transform(X_test[scaled_columns])

    feature_names = [*scaled_columns, *dummy_names]
    X_train_final = pd.DataFrame(
        np.hstack([train_scaled, train_dummies]), columns=feature_names, index=X_train.index
    )
    X_test_final = pd.DataFrame(
        np.hstack([test_scaled, test_dummies]), columns=feature_names, index=X_test.index
    )

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "feature_names": feature_names,
        "scaled_columns": scaled_columns,
        "dummy_names": dummy_names,
        "numeric_medians": dict(zip(NUMERIC_COLUMNS, numeric_imputer.statistics_.tolist())),
        "categorical_modes": dict(zip(CATEGORICAL_COLUMNS, categorical_imputer.statistics_.tolist())),
        "foodcourt_before": foodcourt_before,
        "foodcourt_after": foodcourt_after,
    }


def final_checks(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict[str, Any]:
    """Item D — as verificações que precisam passar antes de alimentar a rede."""
    train_values = X_train.to_numpy(dtype=float)
    test_values = X_test.to_numpy(dtype=float)

    checks = {
        "shape_train": list(X_train.shape),
        "shape_test": list(X_test.shape),
        "n_features": int(X_train.shape[1]),
        "nan_train": int(np.isnan(train_values).sum()),
        "nan_test": int(np.isnan(test_values).sum()),
        "inf_train": int(np.isinf(train_values).sum()),
        "inf_test": int(np.isinf(test_values).sum()),
        "min_train": float(train_values.min()),
        "max_train": float(train_values.max()),
        "min_test": float(test_values.min()),
        "max_test": float(test_values.max()),
        "all_numeric_train": bool(all(np.issubdtype(t, np.number) for t in X_train.dtypes)),
        "all_numeric_test": bool(all(np.issubdtype(t, np.number) for t in X_test.dtypes)),
    }
    assert checks["nan_train"] == 0 and checks["nan_test"] == 0, "sobrou NaN após o pré-processamento"
    assert checks["inf_train"] == 0 and checks["inf_test"] == 0, "há valores infinitos nas matrizes finais"
    assert checks["all_numeric_train"] and checks["all_numeric_test"], "há colunas não numéricas"
    assert checks["shape_train"][1] == checks["shape_test"][1], "treino e teste têm larguras diferentes"
    return checks


def figure_06(before: np.ndarray, after: np.ndarray, y_train: pd.Series) -> str:
    """Figura 6 — ``FoodCourt`` antes e depois do ``log1p``, apenas no conjunto de treino."""
    labels = y_train.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    panels = [
        (axes[0], before, "Antes — `FoodCourt` bruto (unidades monetárias)",
         "Antes da transformação\n(valores originais, sem log e sem escalonamento)"),
        (axes[1], after, r"Depois — $\log(1 + \mathrm{FoodCourt})$",
         "Depois do log1p\n(antes do escalonamento MinMax)"),
    ]
    for ax, values, xlabel, title in panels:
        bins = np.histogram_bin_edges(values, bins=50)
        for label, name, color in zip((0, 1), ("Transported = False", "Transported = True"), BINARY_COLORS):
            ax.hist(
                values[labels == label],
                bins=bins,
                alpha=0.6,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=name,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequência (nº de passageiros no treino)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2, linestyle=":")

    fig.suptitle(
        "Figura 6 — `FoodCourt` no conjunto de treino: a cauda pesada antes e depois de log1p\n"
        "(eixo y em escala logarítmica para que a cauda continue visível)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return save_figure(fig, "figure_06_foodcourt_before_after.png").name


def run() -> dict[str, Any]:
    """Executa o Exercise 3 inteiro e devolve todas as métricas citadas no relatório.

    Não recebe ``rng``: a única fonte de aleatoriedade é o ``train_test_split``,
    fixado por ``random_state=42``.
    """
    df = load_dataset()
    overview = describe_dataset(df)

    print(f"Spaceship Titanic — {overview['n_rows']} linhas × {overview['n_columns']} colunas")
    print(
        f"  Transported=True : {overview['class_counts']['True']}"
        f" ({overview['class_share']['True'] * 100:.2f}%)"
    )
    print(
        f"  Transported=False: {overview['class_counts']['False']}"
        f" ({overview['class_share']['False'] * 100:.2f}%)"
    )
    print("\n  Valores ausentes por coluna:")
    for row in overview["missing_table"]:
        print(f"    {row['column']:<14} {int(row['missing']):>5}  ({row['missing_pct']:.2f}%)")
    print("\n  Gastos (dataset completo): média / mediana / máximo")
    for column, stats in overview["spend_stats"].items():
        print(f"    {column:<14} {stats['mean']:>10.2f} {stats['median']:>10.2f} {stats['max']:>12.2f}")

    parts = split(df)
    print(
        f"\nSplit 80/20 estratificado: treino = {parts['n_train']}, teste = {parts['n_test']}"
    )
    print(
        f"  proporção de Transported=True — completo {parts['share_full'] * 100:.4f}% |"
        f" treino {parts['share_train'] * 100:.4f}% | teste {parts['share_test'] * 100:.4f}%"
    )

    # Média e mediana de FoodCourt no TREINO, antes de qualquer transformação —
    # é a linha 11 da tabela de resultados.
    foodcourt_train_raw = parts["X_train"]["FoodCourt"]
    foodcourt_train = {
        "mean": float(foodcourt_train_raw.mean()),
        "median": float(foodcourt_train_raw.median()),
        "max": float(foodcourt_train_raw.max()),
        "missing": int(foodcourt_train_raw.isna().sum()),
    }
    print(
        f"  FoodCourt no treino (bruto): média = {foodcourt_train['mean']:.4f},"
        f" mediana = {foodcourt_train['median']:.4f}"
    )

    prepared = preprocess(parts["X_train"], parts["X_test"])
    checks = final_checks(prepared["X_train"], prepared["X_test"])
    fig6 = figure_06(prepared["foodcourt_before"], prepared["foodcourt_after"], parts["y_train"])

    print(f"\nMatriz final — treino {tuple(checks['shape_train'])}, teste {tuple(checks['shape_test'])}")
    print(f"  {checks['n_features']} features: {', '.join(prepared['feature_names'])}")
    print(f"  NaN: treino {checks['nan_train']}, teste {checks['nan_test']}")
    print(f"  Infinitos: treino {checks['inf_train']}, teste {checks['inf_test']}")
    print(f"  Faixa treino: [{checks['min_train']:.4f}, {checks['max_train']:.4f}]")
    print(f"  Faixa teste : [{checks['min_test']:.4f}, {checks['max_test']:.4f}]")

    return {
        "overview": overview,
        "split": {k: v for k, v in parts.items() if k not in {"X_train", "X_test", "y_train", "y_test"}},
        "foodcourt_train_raw": foodcourt_train,
        "numeric_medians": prepared["numeric_medians"],
        "categorical_modes": prepared["categorical_modes"],
        "feature_names": prepared["feature_names"],
        "checks": checks,
        "figures": [fig6],
    }
