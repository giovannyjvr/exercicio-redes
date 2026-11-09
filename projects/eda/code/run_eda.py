"""Análise exploratória completa do *NYC Airbnb Open Data* — ponto de entrada único.

Executa as etapas 0 a 5 da rubrica, grava as figuras numeradas em ``figures/`` e
escreve **todos** os números citados no relatório em ``results/metrics.json``.

Uso, a partir da raiz do repositório::

    python docs/projects/eda/code/run_eda.py

A semente é 42 em tudo que sorteia: split, amostragem para os mapas não lineares,
PCA, t-SNE e UMAP.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sem display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE, trustworthiness  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nyc_preprocessing import (  # noqa: E402
    CAT_HIGH, CAT_LOW, DROPPED_COLUMNS, FLAG_COLUMNS, NUM_HEAVY, NUM_PLAIN,
    RANDOM_STATE, TARGET, TEST_SIZE, build_preprocessor, drop_invalid_target,
    feature_names, load_dataset, split_data, verify,
)

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
RESULTS = BASE / "results"
DPI = 140

#: Amostra reproduzível do treino para t-SNE/UMAP. t-SNE é O(n²) em memória;
#: 5 000 pontos rodam em ~1 min por configuração nesta máquina, 39 107 não rodam.
EMBED_SAMPLE = 5000
PRICE_CMAP = "viridis"

DATASET_URL = "https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data"


def save_figure(fig: plt.Figure, filename: str, *, tight: bool = True) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # tight_layout briga com uma colorbar que rouba espaço de vários eixos de
    # uma vez; essas figuras usam constrained layout e passam tight=False.
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


# ============================================================ Etapa 0 e 1A/1B/1C


def etapa_0_1(df: pd.DataFrame) -> dict[str, Any]:
    """Adequação do dataset, dicionário de dados, qualidade e o alvo."""
    section("Etapa 0 e 1 — adequação, dicionário, qualidade e alvo")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # --- 0. adequação -------------------------------------------------------
    adequacy = {
        "url": DATASET_URL,
        "file": Path(df.attrs["source_file"]).name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": TARGET,
        "target_in_file": TARGET in df.columns,
        "n_numeric_columns": len(numeric_cols),
        "n_categorical_columns": len(categorical_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "at_least_1000_rows": bool(df.shape[0] >= 1000),
    }
    print(f"  {adequacy['n_rows']} linhas x {adequacy['n_cols']} colunas | alvo: {TARGET}")
    print(f"  numéricas: {len(numeric_cols)} | categóricas/texto: {len(categorical_cols)}")

    # --- 1B. qualidade ------------------------------------------------------
    missing = pd.DataFrame({
        "ausentes": df.isna().sum(),
        "pct": (df.isna().mean() * 100).round(4),
    }).sort_values("ausentes", ascending=False)
    worst = missing.index[0]
    print(f"  coluna com mais ausentes: {worst} "
          f"({int(missing.loc[worst, 'ausentes'])} = {missing.loc[worst, 'pct']:.2f} %)")

    # A ausência é estrutural? Confere a hipótese em vez de supô-la.
    never = df["number_of_reviews"] == 0
    structural = {
        "reviews_per_month_missing": int(df["reviews_per_month"].isna().sum()),
        "last_review_missing": int(df["last_review"].isna().sum()),
        "number_of_reviews_zero": int(never.sum()),
        "missing_iff_zero_reviews": bool(
            (df["reviews_per_month"].isna() == never).all()
            and (df["last_review"].isna() == never).all()
        ),
    }
    print(f"  ausência estrutural confirmada: {structural['missing_iff_zero_reviews']} "
          f"({structural['number_of_reviews_zero']} anúncios sem nenhuma avaliação)")

    # Duplicatas: linha inteira, id, e "mesmo anúncio" por chave de negócio.
    business_key = ["host_id", "name", "latitude", "longitude", "room_type"]
    duplicates = {
        "full_rows": int(df.duplicated().sum()),
        "id": int(df["id"].duplicated().sum()),
        "business_key": int(df.duplicated(subset=business_key).sum()),
        "business_key_columns": business_key,
    }
    print(f"  duplicatas: linha inteira {duplicates['full_rows']}, "
          f"id {duplicates['id']}, chave de negócio {duplicates['business_key']}")

    # Valores impossíveis ou suspeitos, cada um com a regra explícita.
    rules = {
        "price <= 0": int((df[TARGET] <= 0).sum()),
        "price > 1000": int((df[TARGET] > 1000).sum()),
        "minimum_nights > 365": int((df["minimum_nights"] > 365).sum()),
        "availability_365 == 0": int((df["availability_365"] == 0).sum()),
        "reviews_per_month > 31": int((df["reviews_per_month"] > 31).sum()),
        "latitude fora de NYC": int(((df["latitude"] < 40.4) | (df["latitude"] > 41.0)).sum()),
        "longitude fora de NYC": int(((df["longitude"] < -74.3) | (df["longitude"] > -73.6)).sum()),
    }
    for rule, count in rules.items():
        print(f"    {rule:<26} {count}")

    identifiers = {
        col: {"nunique": int(df[col].nunique()), "ratio": float(df[col].nunique() / len(df))}
        for col in ["id", "host_id", "name", "host_name"]
    }
    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # --- 1C. o alvo ---------------------------------------------------------
    price = df[TARGET]
    log_price = np.log1p(price[price > 0])
    target_stats = {
        "count": int(price.count()),
        "mean": float(price.mean()),
        "median": float(price.median()),
        "std": float(price.std()),
        "min": float(price.min()),
        "q25": float(price.quantile(0.25)),
        "q75": float(price.quantile(0.75)),
        "q99": float(price.quantile(0.99)),
        "max": float(price.max()),
        "skew": float(price.skew()),
        "kurtosis": float(price.kurtosis()),
        "zeros": int((price == 0).sum()),
        "negatives": int((price < 0).sum()),
        "log1p_skew": float(log_price.skew()),
        "share_above_500": float((price > 500).mean()),
        "share_above_1000": float((price > 1000).mean()),
    }

    # Quanto da soma dos desvios quadráticos vem da cauda. É o número que decide
    # se vale treinar em log: com erro quadrático, o gradiente é proporcional a
    # isto, e não à contagem de anúncios.
    squared_deviation = (price - price.mean()) ** 2
    total_sq = float(squared_deviation.sum())
    target_stats["squared_error_share"] = {
        "above_500": float(squared_deviation[price > 500].sum() / total_sq),
        "above_1000": float(squared_deviation[price > 1000].sum() / total_sq),
        "at_or_below_median": float(
            squared_deviation[price <= price.median()].sum() / total_sq
        ),
        "n_at_or_below_median": int((price <= price.median()).sum()),
        "ratio_10000_vs_median": float(
            (10000 - price.mean()) ** 2 / (price.median() - price.mean()) ** 2
        ),
    }
    s = target_stats["squared_error_share"]
    print(f"  cauda e erro quadrático: {s['above_500'] * 100:.1f} % da soma dos desvios"
          f" quadráticos vem dos 2,14 % acima de US$ 500;"
          f" {s['above_1000'] * 100:.1f} % vem dos 0,49 % acima de US$ 1 000")
    print(f"    a metade mais barata ({s['n_at_or_below_median']} anúncios) responde por"
          f" {s['at_or_below_median'] * 100:.1f} %")
    print(f"  price: média {target_stats['mean']:.2f}, mediana {target_stats['median']:.2f}, "
          f"skew {target_stats['skew']:.3f} (log1p: {target_stats['log1p_skew']:.3f})")

    # --- Figura 1 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    clipped = price.clip(upper=1000)
    axes[0].hist(clipped, bins=80, color="#1f77b4", edgecolor="none")
    axes[0].axvline(target_stats["median"], color="#d62728", lw=2,
                    label=f"mediana = {target_stats['median']:.0f}")
    axes[0].axvline(target_stats["mean"], color="#2ca02c", lw=2, ls="--",
                    label=f"média = {target_stats['mean']:.2f}")
    axes[0].set_xlabel("price (US$/noite) — eixo truncado em 1000")
    axes[0].set_ylabel("Nº de anúncios")
    axes[0].set_title(f"Escala original — assimetria {target_stats['skew']:.2f}")
    axes[0].legend()
    axes[1].hist(log_price, bins=80, color="#9467bd", edgecolor="none")
    axes[1].set_xlabel("log(1 + price)")
    axes[1].set_ylabel("Nº de anúncios")
    axes[1].set_title(f"Após log1p — assimetria {target_stats['log1p_skew']:.2f}")
    fig.suptitle("Figura 1 — Distribuição de price, o alvo da regressão", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig1 = save_figure(fig, "figure_01_target_price.png")

    return {
        "adequacy": adequacy,
        "missing": {
            "table": [
                {"coluna": c, "ausentes": int(missing.loc[c, "ausentes"]),
                 "pct": float(missing.loc[c, "pct"])}
                for c in missing.index
            ],
            "worst_column": worst,
            "worst_count": int(missing.loc[worst, "ausentes"]),
            "worst_pct": float(missing.loc[worst, "pct"]),
            "n_columns_with_missing": int((missing["ausentes"] > 0).sum()),
        },
        "structural_missingness": structural,
        "duplicates": duplicates,
        "impossible_values": rules,
        "identifiers": identifiers,
        "constant_columns": constants,
        "dropped_columns": DROPPED_COLUMNS,
        "target": target_stats,
        "figures": [fig1],
    }


# ====================================================================== Etapa 2


def etapa_2(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Univariada — só no treino, para não deixar o teste influenciar decisão alguma."""
    section("Etapa 2 — análise univariada (somente treino)")

    numeric = NUM_HEAVY + ["latitude", "longitude", "availability_365"]
    desc = X_train[numeric].describe().T
    desc["skew"] = X_train[numeric].skew()
    desc["missing"] = X_train[numeric].isna().sum()
    print(desc.round(3).to_string())

    # --- Figura 2: as quatro de cauda pesada, em log1p para serem legíveis ---
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    for ax, col in zip(axes.ravel(), NUM_HEAVY):
        values = X_train[col].dropna()
        ax.hist(np.log1p(values), bins=60, color="#1f77b4", edgecolor="none")
        ax.set_xlabel(f"log(1 + {col})")
        ax.set_ylabel("Nº de anúncios")
        ax.set_title(f"{col} — assimetria {values.skew():.2f}, máx {values.max():.0f}")
        ax.grid(alpha=0.2, linestyle=":")
    fig.suptitle("Figura 2 — As quatro features numéricas de cauda pesada (escala log1p)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig2 = save_figure(fig, "figure_02_numeric_heavy.png")

    # --- Figura 3: as de escala bem-comportada -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, col in zip(axes, ["latitude", "longitude", "availability_365"]):
        ax.hist(X_train[col].dropna(), bins=60, color="#2ca02c", edgecolor="none")
        ax.set_xlabel(col)
        ax.set_ylabel("Nº de anúncios")
        ax.set_title(f"{col} — assimetria {X_train[col].skew():.2f}")
        ax.grid(alpha=0.2, linestyle=":")
    fig.suptitle("Figura 3 — Features numéricas de escala bem-comportada", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig3 = save_figure(fig, "figure_03_numeric_plain.png")

    # --- 2B categóricas ------------------------------------------------------
    categorical = CAT_LOW + CAT_HIGH
    cat_summary = {}
    for col in categorical:
        counts = X_train[col].value_counts(dropna=False)
        cat_summary[col] = {
            "cardinality": int(X_train[col].nunique(dropna=True)),
            "missing": int(X_train[col].isna().sum()),
            "top": [
                {"categoria": str(k), "n": int(v), "pct": float(v / len(X_train) * 100)}
                for k, v in counts.head(10).items()
            ],
            "rare_below_50": int((counts < 50).sum()),
            "rows_in_rare_below_50": int(counts[counts < 50].sum()),
            "share_rows_in_rare": float(counts[counts < 50].sum() / len(X_train)),
        }
        print(f"  {col}: cardinalidade {cat_summary[col]['cardinality']}, "
              f"categorias com < 50 linhas: {cat_summary[col]['rare_below_50']} "
              f"({cat_summary[col]['share_rows_in_rare'] * 100:.2f} % das linhas)")

    # --- Figura 4: as duas de baixa cardinalidade ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, col, color in zip(axes, CAT_LOW, ("#1f77b4", "#ff7f0e")):
        counts = X_train[col].value_counts()
        pct = counts / len(X_train) * 100
        bars = ax.bar(range(len(counts)), counts.values, color=color)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=20, ha="right")
        ax.set_ylabel("Nº de anúncios no treino")
        ax.set_title(f"{col} — {len(counts)} categorias")
        for bar, p in zip(bars, pct.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.2, linestyle=":", axis="y")
    fig.suptitle("Figura 4 — Categóricas de baixa cardinalidade", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig4 = save_figure(fig, "figure_04_categorical_low.png")

    # --- Figura 5: a de alta cardinalidade ----------------------------------
    counts = X_train["neighbourhood"].value_counts()
    cumulative = counts.cumsum() / counts.sum() * 100
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    top = counts.head(20)
    axes[0].barh(range(len(top))[::-1], top.values, color="#9467bd")
    axes[0].set_yticks(range(len(top))[::-1])
    axes[0].set_yticklabels(top.index, fontsize=8)
    axes[0].set_xlabel("Nº de anúncios no treino")
    axes[0].set_title("20 bairros mais frequentes")
    axes[0].grid(alpha=0.2, linestyle=":", axis="x")
    axes[1].plot(range(1, len(cumulative) + 1), cumulative.values, color="#d62728", lw=2)
    axes[1].axhline(90, color="grey", ls="--", lw=1)
    n_for_90 = int((cumulative < 90).sum() + 1)
    axes[1].axvline(n_for_90, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Bairros ordenados por frequência")
    axes[1].set_ylabel("Cobertura acumulada das linhas (%)")
    axes[1].set_title(f"{n_for_90} bairros cobrem 90 % do treino "
                      f"(de {len(counts)})")
    axes[1].grid(alpha=0.2, linestyle=":")
    fig.suptitle("Figura 5 — neighbourhood: 221 categorias, concentração forte", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig5 = save_figure(fig, "figure_05_neighbourhood_cardinality.png")
    cat_summary["neighbourhood"]["n_for_90pct_coverage"] = n_for_90

    return {
        "numeric_describe": json.loads(desc.round(6).to_json(orient="index")),
        "categorical": cat_summary,
        "figures": [fig2, fig3, fig4, fig5],
    }


# ====================================================================== Etapa 3


def etapa_3(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Bivariada e multivariada — treino apenas."""
    section("Etapa 3 — análise bivariada e multivariada (somente treino)")

    numeric = NUM_HEAVY + ["latitude", "longitude", "availability_365"]
    frame = X_train[numeric].copy()
    frame["price"] = y_train.values

    # Spearman: as features são fortemente assimétricas (skew de 1,3 a 22) e as
    # relações com o preço não são lineares. Pearson mediria a reta errada.
    spearman = frame.corr(method="spearman")
    pearson = frame.corr(method="pearson")

    # Par mais correlacionado ENTRE FEATURES (fora a diagonal e fora o alvo).
    # .to_numpy(copy=True): no pandas 3.0 o buffer devolvido por .values é
    # somente-leitura, e fill_diagonal escreveria nele.
    corr_values = spearman.loc[numeric, numeric].abs().to_numpy(copy=True)
    np.fill_diagonal(corr_values, np.nan)
    feature_corr = pd.DataFrame(corr_values, index=numeric, columns=numeric)
    flat = feature_corr.unstack().dropna().sort_values(ascending=False)
    best_pair = flat.index[0]
    best_value = float(spearman.loc[best_pair[0], best_pair[1]])
    print(f"  par de features mais correlacionado (Spearman): "
          f"{best_pair[0]} x {best_pair[1]} = {best_value:.4f}")

    target_corr = spearman["price"].drop("price").sort_values(key=abs, ascending=False)
    print("  correlação de Spearman com price:")
    for name, value in target_corr.items():
        print(f"    {name:<32} {value:+.4f}")

    redundant = [
        {"a": a, "b": b, "spearman": float(spearman.loc[a, b])}
        for (a, b), v in flat.items() if v >= 0.5 and a < b
    ]

    # --- Figura 6: matriz de correlação -------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    matrix = spearman.values
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = list(spearman.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(matrix[i, j]) > 0.55 else "black")
    ax.set_title("Figura 6 — Correlação de Spearman (treino)\n"
                 "escolhida por causa da assimetria forte e das relações não lineares")
    fig.colorbar(im, ax=ax, label="ρ de Spearman")
    fig6 = save_figure(fig, "figure_06_correlation.png")

    # --- Figura 7: o par mais correlacionado e o mapa geográfico ------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    axes[0].scatter(np.log1p(X_train[best_pair[0]]), np.log1p(X_train[best_pair[1]]),
                    s=4, alpha=0.15, color="#1f77b4", edgecolors="none")
    axes[0].set_xlabel(f"log(1 + {best_pair[0]})")
    axes[0].set_ylabel(f"log(1 + {best_pair[1]})")
    axes[0].set_title(f"Par mais correlacionado — ρ = {best_value:.3f}")
    axes[0].grid(alpha=0.2, linestyle=":")

    price_clip = y_train.clip(upper=y_train.quantile(0.99))
    sc = axes[1].scatter(X_train["longitude"], X_train["latitude"], c=price_clip,
                         s=3, alpha=0.5, cmap=PRICE_CMAP, edgecolors="none")
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel("latitude")
    axes[1].set_title("Preço no mapa (cor truncada no percentil 99)")
    fig.colorbar(sc, ax=axes[1], label="price (US$/noite)")
    fig.suptitle("Figura 7 — Relações numéricas representativas", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig7 = save_figure(fig, "figure_07_scatter.png")

    # --- 3B categórica x alvo ------------------------------------------------
    by_category = {}
    for col in CAT_LOW:
        grouped = frame.assign(cat=X_train[col].values).groupby("cat")["price"]
        table = pd.DataFrame({
            "n": grouped.size(),
            "mediana": grouped.median(),
            "media": grouped.mean(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
            "iqr": grouped.quantile(0.75) - grouped.quantile(0.25),
        }).sort_values("mediana", ascending=False)
        by_category[col] = json.loads(table.round(4).to_json(orient="index"))
        print(f"\n  price por {col}:")
        print(table.round(2).to_string())

    hood = frame.assign(cat=X_train["neighbourhood"].values).groupby("cat")["price"]
    hood_table = pd.DataFrame({"n": hood.size(), "mediana": hood.median()})
    hood_table = hood_table[hood_table["n"] >= 50].sort_values("mediana", ascending=False)

    # --- Figura 8: price por room_type e por neighbourhood_group ------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, col in zip(axes, CAT_LOW):
        order = frame.assign(cat=X_train[col].values).groupby("cat")["price"].median()
        order = order.sort_values(ascending=False).index.tolist()
        data = [y_train[X_train[col] == level].values for level in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_yscale("log")
        ax.set_ylabel("price (US$/noite, escala log)")
        ax.set_title(f"price por {col}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2, linestyle=":", axis="y")
    fig.suptitle("Figura 8 — Distribuição de price entre categorias "
                 "(outliers omitidos; escala log)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig8 = save_figure(fig, "figure_08_price_by_category.png")

    # --- Figura 9: bairros extremos ------------------------------------------
    top15, bottom15 = hood_table.head(15), hood_table.tail(15)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    combined = pd.concat([top15, bottom15])
    colors = ["#d62728"] * len(top15) + ["#1f77b4"] * len(bottom15)
    ax.barh(range(len(combined))[::-1], combined["mediana"].values, color=colors)
    ax.set_yticks(range(len(combined))[::-1])
    ax.set_yticklabels([f"{i}  (n={int(n)})" for i, n in
                        zip(combined.index, combined["n"])], fontsize=8)
    ax.set_xlabel("Mediana de price (US$/noite)")
    ax.set_title("Figura 9 — 15 bairros mais caros e 15 mais baratos\n"
                 f"(apenas os {len(hood_table)} bairros com ao menos 50 anúncios no treino)")
    ax.grid(alpha=0.2, linestyle=":", axis="x")
    fig9 = save_figure(fig, "figure_09_price_by_neighbourhood.png")

    # --- 3C numérica x categórica -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    pairs = [("availability_365", "room_type"), ("minimum_nights", "room_type"),
             ("number_of_reviews", "neighbourhood_group")]
    cross = {}
    for ax, (num_col, cat_col) in zip(axes, pairs):
        order = X_train.groupby(cat_col)[num_col].median().sort_values(ascending=False).index
        data = [X_train.loc[X_train[cat_col] == level, num_col].dropna().values
                for level in order]
        ax.boxplot(data, tick_labels=list(order), showfliers=False)
        ax.set_ylabel(num_col)
        ax.set_title(f"{num_col} por {cat_col}")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(alpha=0.2, linestyle=":", axis="y")
        stats = X_train.groupby(cat_col)[num_col].agg(["median", "mean", "std", "count"])
        cross[f"{num_col}__{cat_col}"] = json.loads(stats.round(4).to_json(orient="index"))
    fig.suptitle("Figura 10 — Features numéricas entre grupos categóricos "
                 "(outliers omitidos)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig10 = save_figure(fig, "figure_10_numeric_by_category.png")

    return {
        "correlation_method": "spearman",
        "spearman": json.loads(spearman.round(6).to_json(orient="index")),
        "pearson": json.loads(pearson.round(6).to_json(orient="index")),
        "best_feature_pair": {"a": best_pair[0], "b": best_pair[1], "spearman": best_value},
        "redundant_pairs": redundant,
        "target_correlation": {k: float(v) for k, v in target_corr.items()},
        "price_by_category": by_category,
        "neighbourhood_price": {
            "n_neighbourhoods_with_50plus": int(len(hood_table)),
            "most_expensive": [
                {"bairro": i, "n": int(r["n"]), "mediana": float(r["mediana"])}
                for i, r in hood_table.head(5).iterrows()
            ],
            "cheapest": [
                {"bairro": i, "n": int(r["n"]), "mediana": float(r["mediana"])}
                for i, r in hood_table.tail(5).iterrows()
            ],
        },
        "numeric_by_category": cross,
        "figures": [fig6, fig7, fig8, fig9, fig10],
    }


# ====================================================================== Etapa 4


def local_price_homogeneity(embedding: np.ndarray, price: np.ndarray, k: int = 15) -> float:
    """Diferença média de log-preço entre um ponto e seus ``k`` vizinhos no mapa.

    Para regressão não existe silhueta: o alvo é contínuo. Esta é a pergunta
    equivalente — *anúncios vizinhos no mapa têm preços parecidos?* Quanto menor,
    mais o mapa agrupa preços semelhantes. Compare sempre com a linha de base de
    pares aleatórios devolvida por :func:`random_price_gap`.
    """
    log_price = np.log1p(price)
    finder = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    _, neighbours = finder.kneighbors(embedding)
    gaps = np.abs(log_price[neighbours[:, 1:]] - log_price[:, None])
    return float(gaps.mean())


def random_price_gap(price: np.ndarray, rng: np.random.Generator, n_pairs: int = 200_000) -> float:
    """Linha de base: diferença média de log-preço entre pares sorteados ao acaso."""
    log_price = np.log1p(price)
    a = rng.integers(0, len(log_price), n_pairs)
    b = rng.integers(0, len(log_price), n_pairs)
    return float(np.abs(log_price[a] - log_price[b]).mean())


def etapa_4(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, Any]:
    """Pré-processamento, verificação e redução de dimensionalidade."""
    section("Etapa 4 — pré-processamento e redução de dimensionalidade")

    prep = build_preprocessor(X_train)
    Z_train = prep.fit_transform(X_train)
    Z_test = prep.transform(X_test)
    checks = verify(prep, Z_train, Z_test)
    names = feature_names(prep)

    print(f"  treino {checks['shape_train']} | teste {checks['shape_test']}")
    print(f"  NaN: treino {checks['nan_train']}, teste {checks['nan_test']}")
    print(f"  |média| máx nas colunas escaladas: treino {checks['max_abs_mean_train_scaled']:.2e}, "
          f"teste {checks['max_abs_mean_test_scaled']:.4f}  <- teste != 0 prova que não houve vazamento")

    # --- quantas linhas a winsorização toca ---------------------------------
    winsorizer = prep.named_steps["prep"].named_transformers_["heavy"].named_steps["win"]
    engineered_train = prep.named_steps["features"].transform(X_train)
    engineered_test = prep.named_steps["features"].transform(X_test)
    raw_heavy_train = engineered_train[NUM_HEAVY].fillna(0.0).to_numpy(dtype=float)
    raw_heavy_test = engineered_test[NUM_HEAVY].fillna(0.0).to_numpy(dtype=float)

    def clipped_report(values: np.ndarray) -> dict[str, Any]:
        below = values < winsorizer.lower_bounds_
        above = values > winsorizer.upper_bounds_
        touched = (below | above).any(axis=1)
        return {
            "rows_touched": int(touched.sum()),
            "share_rows_touched": float(touched.mean()),
            "per_column": {
                col: {"below": int(below[:, i].sum()), "above": int(above[:, i].sum())}
                for i, col in enumerate(NUM_HEAVY)
            },
        }

    outliers = {
        "bounds": {
            col: {"lower": float(winsorizer.lower_bounds_[i]),
                  "upper": float(winsorizer.upper_bounds_[i])}
            for i, col in enumerate(NUM_HEAVY)
        },
        "train": clipped_report(raw_heavy_train),
        "test": clipped_report(raw_heavy_test),
    }
    print(f"  winsorização: {outliers['train']['rows_touched']} linhas do treino "
          f"({outliers['train']['share_rows_touched'] * 100:.2f} %) e "
          f"{outliers['test']['rows_touched']} do teste "
          f"({outliers['test']['share_rows_touched'] * 100:.2f} %)")
    for col, b in outliers["bounds"].items():
        print(f"    {col:<32} corte em [{b['lower']:.2f}, {b['upper']:.2f}]")

    # Categoria inédita no teste: aconteceu de fato?
    unseen = {
        col: sorted(set(X_test[col].dropna().unique()) - set(X_train[col].dropna().unique()))
        for col in CAT_LOW + CAT_HIGH
    }
    print(f"  categorias inéditas no teste: "
          f"{ {k: len(v) for k, v in unseen.items()} }")

    # --- de onde vem a variância ---------------------------------------------
    # A PCA busca direções de máxima variância, então quem tem variância manda no
    # resultado. As colunas padronizadas têm variância 1 por construção; uma
    # coluna one-hot de categoria rara tem p(1-p), que é minúsculo. Medir a
    # repartição explica os loadings antes de olhar o gráfico.
    column_block = [str(n).split("__")[0] for n in names]
    variances = Z_train.var(axis=0)
    total_variance = float(variances.sum())
    variance_by_block = {}
    for block in dict.fromkeys(column_block):
        idx = [i for i, b in enumerate(column_block) if b == block]
        share = float(variances[idx].sum())
        variance_by_block[block] = {
            "n_columns": len(idx),
            "variance": share,
            "share_of_total": share / total_variance,
        }
    print("  repartição da variância entre os blocos de colunas:")
    for block, info in variance_by_block.items():
        print(f"    {block:<8} {info['n_columns']:>3} colunas | variância {info['variance']:>7.3f}"
              f" | {info['share_of_total'] * 100:>5.2f} % do total")

    # --- PCA ------------------------------------------------------------------
    pca_full = PCA(random_state=RANDOM_STATE).fit(Z_train)
    ratio = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    n_for_80 = int(np.searchsorted(cumulative, 0.80) + 1)
    n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    pc12 = float(cumulative[1])
    print(f"  PCA: PC1 {ratio[0] * 100:.2f} % + PC2 {ratio[1] * 100:.2f} % = {pc12 * 100:.2f} %")
    print(f"       {n_for_80} componentes para 80 %, {n_for_95} para 95 % "
          f"(de {len(ratio)} colunas)")

    pca2 = PCA(2, random_state=RANDOM_STATE).fit(Z_train)
    A_train = pca2.transform(Z_train)

    loadings = {}
    for i in (0, 1):
        series = pd.Series(pca2.components_[i], index=names).sort_values(key=abs, ascending=False)
        loadings[f"PC{i + 1}"] = [
            {"feature": str(k), "loading": float(v)} for k, v in series.head(8).items()
        ]
        print(f"  loadings dominantes da PC{i + 1}:")
        for k, v in series.head(5).items():
            print(f"    {str(k):<48} {v:+.4f}")

    # --- Figura 11: variância explicada --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    axes[0].bar(range(1, 21), ratio[:20] * 100, color="#1f77b4")
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Variância explicada (%)")
    axes[0].set_title("20 primeiras componentes")
    axes[0].grid(alpha=0.2, linestyle=":", axis="y")
    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100, color="#d62728", lw=2)
    for threshold, n in ((80, n_for_80), (95, n_for_95)):
        axes[1].axhline(threshold, color="grey", ls="--", lw=1)
        axes[1].axvline(n, color="grey", ls=":", lw=1)
        axes[1].text(n + 2, threshold - 6, f"{n} comp. → {threshold} %", fontsize=9)
    axes[1].set_xlabel("Nº de componentes")
    axes[1].set_ylabel("Variância explicada acumulada (%)")
    axes[1].set_title(f"Acumulada — PC1+PC2 = {pc12 * 100:.2f} %")
    axes[1].grid(alpha=0.2, linestyle=":")
    fig.suptitle("Figura 11 — PCA: quanta variância cabe em poucas direções", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig11 = save_figure(fig, "figure_11_pca_variance.png")

    # --- amostra reproduzível para os mapas não lineares --------------------
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(EMBED_SAMPLE, Z_train.shape[0])
    sample_idx = rng.choice(Z_train.shape[0], size=n_sample, replace=False)
    Z_sample = Z_train[sample_idx]
    price_sample = y_train.to_numpy()[sample_idx]
    A_sample = A_train[sample_idx]
    print(f"  amostra para t-SNE/UMAP: {n_sample} linhas sorteadas sem reposição "
          f"do treino, com default_rng({RANDOM_STATE})")

    baseline_gap = random_price_gap(price_sample, rng)
    norm = Normalize(vmin=np.percentile(price_sample, 1), vmax=np.percentile(price_sample, 99))

    def scatter_price(ax, emb: np.ndarray, title: str):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=price_sample, s=5, alpha=0.6,
                        cmap=PRICE_CMAP, norm=norm, edgecolors="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        return sc

    # --- Figura 12: PCA 2D colorido por preço --------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    sc = scatter_price(ax, A_sample,
                       f"PC1 ({ratio[0] * 100:.1f} %) × PC2 ({ratio[1] * 100:.1f} %)")
    fig.colorbar(sc, ax=ax, label="price (US$/noite, escala truncada em p1–p99)")
    ax.set_title("Figura 12 — Projeção PCA do treino pré-processado, colorida por price\n"
                 f"(amostra de {n_sample} anúncios; price não entrou nas features)")
    fig12 = save_figure(fig, "figure_12_pca_scatter.png")

    embeddings: dict[str, np.ndarray] = {"PCA": A_sample}
    quality: dict[str, dict[str, float]] = {}

    def evaluate(name: str, emb: np.ndarray, seconds: float) -> None:
        quality[name] = {
            "trustworthiness_k5": float(trustworthiness(Z_sample, emb, n_neighbors=5)),
            "trustworthiness_k30": float(trustworthiness(Z_sample, emb, n_neighbors=30)),
            "local_price_gap_k15": local_price_homogeneity(emb, price_sample),
            "seconds": seconds,
        }
        q = quality[name]
        print(f"    {name:<22} trust k5 {q['trustworthiness_k5']:.3f} | "
              f"trust k30 {q['trustworthiness_k30']:.3f} | "
              f"gap de preço {q['local_price_gap_k15']:.4f} "
              f"(aleatório {baseline_gap:.4f}) | {seconds:.1f} s")

    print("  qualidade dos mapas:")
    evaluate("PCA", A_sample, 0.0)

    # --- Figura 13: t-SNE com três perplexidades ----------------------------
    perplexities = [5, 30, 50]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), layout="constrained")
    for ax, perp in zip(axes, perplexities):
        start = time.perf_counter()
        emb = TSNE(2, perplexity=perp, random_state=RANDOM_STATE,
                   init="pca").fit_transform(Z_sample)
        elapsed = time.perf_counter() - start
        name = f"t-SNE perp={perp}"
        embeddings[name] = emb
        evaluate(name, emb, elapsed)
        sc = scatter_price(ax, emb, f"perplexity = {perp}")
    fig.colorbar(sc, ax=list(axes), label="price (US$/noite)", shrink=0.9)
    fig.suptitle("Figura 13 — t-SNE sob três perplexidades, colorido por price", fontsize=13)
    fig13 = save_figure(fig, "figure_13_tsne.png", tight=False)

    # --- Figura 14: UMAP com três vizinhanças -------------------------------
    import umap  # importado aqui: carregar numba custa alguns segundos

    neighbours_grid = [5, 15, 50]
    umap_models = {}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), layout="constrained")
    for ax, nn in zip(axes, neighbours_grid):
        start = time.perf_counter()
        reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=RANDOM_STATE)
        emb = reducer.fit_transform(Z_sample)
        elapsed = time.perf_counter() - start
        name = f"UMAP nn={nn}"
        embeddings[name] = emb
        umap_models[nn] = reducer
        evaluate(name, emb, elapsed)
        sc = scatter_price(ax, emb, f"n_neighbors = {nn}")
    fig.colorbar(sc, ax=list(axes), label="price (US$/noite)", shrink=0.9)
    fig.suptitle("Figura 14 — UMAP sob três vizinhanças, colorido por price", fontsize=13)
    fig14 = save_figure(fig, "figure_14_umap.png", tight=False)

    # --- Figura 15: o controle com dados embaralhados ------------------------
    # Embaralha cada coluna independentemente: destrói toda a correlação entre
    # features, preservando as distribuições marginais. Se o mapa do ruído
    # parecer tão estruturado quanto o real, a estrutura veio do algoritmo.
    shuffle_rng = np.random.default_rng(RANDOM_STATE)
    Z_shuffled = Z_sample.copy()
    for col in range(Z_shuffled.shape[1]):
        shuffle_rng.shuffle(Z_shuffled[:, col])
    emb_shuffled = TSNE(2, perplexity=30, random_state=RANDOM_STATE,
                        init="pca").fit_transform(Z_shuffled)
    control = {
        "trustworthiness_k5": float(trustworthiness(Z_shuffled, emb_shuffled, n_neighbors=5)),
        "local_price_gap_k15": local_price_homogeneity(emb_shuffled, price_sample),
    }
    print(f"  controle (features embaralhadas): trust k5 {control['trustworthiness_k5']:.3f}, "
          f"gap de preço {control['local_price_gap_k15']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained")
    scatter_price(axes[0], embeddings["t-SNE perp=30"], "Dados reais — t-SNE perp = 30")
    sc = scatter_price(axes[1], emb_shuffled, "Colunas embaralhadas — t-SNE perp = 30")
    fig.colorbar(sc, ax=list(axes), label="price (US$/noite)", shrink=0.9)
    fig.suptitle("Figura 15 — Controle: o mesmo t-SNE sobre features sem correlação alguma",
                 fontsize=13)
    fig15 = save_figure(fig, "figure_15_shuffle_control.png", tight=False)

    # --- o teste projetado ----------------------------------------------------
    A_test = pca2.transform(Z_test)
    projection = {
        "pca_test_shape": list(A_test.shape),
        "umap_has_transform": True,
        "tsne_has_transform": hasattr(TSNE(), "transform"),
    }

    return {
        "pipeline_checks": checks,
        "outliers": outliers,
        "unseen_categories_in_test": {k: v for k, v in unseen.items()},
        "variance_by_block": {
            "total": total_variance,
            "blocks": variance_by_block,
        },
        "pca": {
            "n_components_total": int(len(ratio)),
            "pc1": float(ratio[0]),
            "pc2": float(ratio[1]),
            "pc1_plus_pc2": pc12,
            "n_components_for_80pct": n_for_80,
            "n_components_for_95pct": n_for_95,
            "cumulative_first_10": [float(v) for v in cumulative[:10]],
            "loadings": loadings,
        },
        "embedding_sample": {
            "n": int(n_sample),
            "method": f"np.random.default_rng({RANDOM_STATE}).choice sem reposição, do treino",
            "random_price_gap_k15_baseline": baseline_gap,
        },
        "embedding_quality": quality,
        "shuffle_control": control,
        "projection": projection,
        "figures": [fig11, fig12, fig13, fig14, fig15],
    }


# ========================================================================= main


def main() -> int:
    t0 = time.perf_counter()

    df_raw = load_dataset()
    print(f"arquivo: {df_raw.attrs['source_file']}")

    metrics_1 = etapa_0_1(df_raw)

    df, target_filter = drop_invalid_target(df_raw)
    print(f"\n  linhas com price <= 0 removidas: {target_filter['rows_dropped']} "
          f"({target_filter['share_dropped'] * 100:.4f} %) -> {target_filter['rows_after']} linhas")

    X_train, X_test, y_train, y_test = split_data(df)
    split_report = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_in": int(X_train.shape[1]),
        "features_in": list(X_train.columns),
        "target_in_features": TARGET in X_train.columns,
        "train_price_mean": float(y_train.mean()),
        "test_price_mean": float(y_test.mean()),
        "train_price_median": float(y_train.median()),
        "test_price_median": float(y_test.median()),
    }
    print(f"  split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} com seed {RANDOM_STATE}: "
          f"treino {split_report['n_train']}, teste {split_report['n_test']}")
    print(f"  price média treino {split_report['train_price_mean']:.2f} x "
          f"teste {split_report['test_price_mean']:.2f} | "
          f"mediana {split_report['train_price_median']:.0f} x "
          f"{split_report['test_price_median']:.0f}")
    assert TARGET not in X_train.columns, "price vazou para dentro de X"

    metrics_2 = etapa_2(X_train, y_train)
    metrics_3 = etapa_3(X_train, y_train)
    metrics_4 = etapa_4(X_train, X_test, y_train, y_test)

    elapsed = time.perf_counter() - t0
    payload = {
        "random_state": RANDOM_STATE,
        "generated_in_seconds": round(elapsed, 1),
        "etapa_0_1": metrics_1,
        "target_filter": target_filter,
        "split": split_report,
        "etapa_2": metrics_2,
        "etapa_3": metrics_3,
        "etapa_4": metrics_4,
    }

    all_figures = (metrics_1["figures"] + metrics_2["figures"]
                   + metrics_3["figures"] + metrics_4["figures"])
    missing = [f for f in all_figures if not (FIGURES / f).exists()]
    if missing:
        for name in missing:
            print(f"  [FALHA] figura não gerada: {name}", file=sys.stderr)
        return 1

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
