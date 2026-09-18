"""Exercise 1 — Point Clouds: Geometry and Spread in 2D.

Quatro classes gaussianas de covariância diagonal em 2D, geradas em quatro
escalas de dispersão. O módulo calcula o *separation ratio* teórico, a
*mixing rate* empírica e produz as Figuras 1, 2 e 3.

Nenhuma função aqui cria a semente: todas recebem o mesmo ``rng`` de
``run_all.py``, para que a sequência aleatória seja única e reprodutível.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import CLASS_COLORS, nearest_center_labels, save_figure

# Parâmetros teóricos do enunciado. As médias nunca mudam; apenas os desvios
# são multiplicados pelo fator de escala.
MEANS = {
    0: [2.0, 3.0],
    1: [5.0, 6.0],
    2: [8.0, 1.0],
    3: [15.0, 4.0],
}
STDS = {
    0: [0.8, 2.5],
    1: [1.2, 1.9],
    2: [0.9, 0.9],
    3: [0.5, 2.0],
}
SCALES = [0.5, 1.0, 2.0, 4.0]
N_PER_CLASS = 100
LABELS = sorted(MEANS)

#: Centros teóricos empilhados na ordem das classes — a referência geométrica
#: usada tanto pela mixing rate quanto pelo esboço de fronteiras da Figura 1.
CENTERS = np.array([MEANS[c] for c in LABELS])


def generate(rng: np.random.Generator, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Amostra ``N_PER_CLASS`` pontos por classe com os desvios multiplicados por ``scale``.

    Retorna ``(X, y)`` com ``X`` de forma ``(400, 2)`` e ``y`` de forma ``(400,)``.
    """
    xs, ys = [], []
    for label in LABELS:
        mean = np.asarray(MEANS[label], dtype=float)
        std = np.asarray(STDS[label], dtype=float) * scale
        xs.append(rng.normal(loc=mean, scale=std, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def mean_spread(label: int, scale: float = 1.0) -> float:
    r"""Dispersão média teórica: :math:`\bar\sigma_k = (\sigma_{k,x} + \sigma_{k,y})/2`."""
    sigma_x, sigma_y = STDS[label]
    return scale * (sigma_x + sigma_y) / 2.0


def separation_ratios(scale: float = 1.0) -> list[dict[str, Any]]:
    r"""Tabela dos seis pares :math:`(i, j)` com o *separation ratio*.

    .. math:: r_{ij} = \frac{\lVert \mu_i - \mu_j \rVert}{\bar\sigma_i + \bar\sigma_j}

    Usa os **parâmetros teóricos** do enunciado, não estimativas amostrais. Como
    as médias não dependem da escala e os desvios são proporcionais a ``s``, vale
    exatamente :math:`r_{ij}(s) = r_{ij}(1)/s` — é por isso que o valor em
    ``s = 2`` sai por divisão, sem gerar dados novos.
    """
    rows = []
    for i, j in combinations(LABELS, 2):
        distance = float(np.linalg.norm(np.array(MEANS[i]) - np.array(MEANS[j])))
        spread_i, spread_j = mean_spread(i, scale), mean_spread(j, scale)
        rows.append(
            {
                "i": i,
                "j": j,
                "distance": distance,
                "spread_i": spread_i,
                "spread_j": spread_j,
                "r_ij": distance / (spread_i + spread_j),
            }
        )
    return rows


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro teórico mais próximo não é o da própria classe.

    Critério puramente geométrico (regra do centro mais próximo); nenhum
    classificador é treinado.
    """
    return float(np.mean(nearest_center_labels(X, CENTERS) != y))


def _scatter_classes(ax: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    """Desenha as quatro classes e marca os centros teóricos."""
    for label in LABELS:
        points = X[y == label]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            alpha=0.75,
            color=CLASS_COLORS[label],
            edgecolors="none",
            label=f"Classe {label}",
        )
    ax.scatter(
        CENTERS[:, 0],
        CENTERS[:, 1],
        marker="X",
        s=170,
        c="black",
        edgecolors="white",
        linewidths=1.4,
        zorder=5,
        label="Centros teóricos",
    )


def _draw_nearest_center_regions(
    ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    """Pinta as regiões de centro mais próximo (partição de Voronoi dos 4 centros).

    São fronteiras **geométricas**, não aprendidas: cada fronteira é a mediatriz
    entre dois centros, logo o conjunto é linear por partes. Serve como esboço
    plausível do que uma rede treinada poderia aprender.
    """
    grid_x = np.linspace(*xlim, 600)
    grid_y = np.linspace(*ylim, 600)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    mesh_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    regions = nearest_center_labels(mesh_points, CENTERS).reshape(mesh_x.shape)

    ax.contourf(
        mesh_x,
        mesh_y,
        regions,
        levels=np.arange(-0.5, len(LABELS), 1.0),
        colors=[CLASS_COLORS[c] for c in LABELS],
        alpha=0.12,
        zorder=0,
    )
    ax.contour(
        mesh_x,
        mesh_y,
        regions,
        levels=np.arange(0.5, len(LABELS) - 1, 1.0),
        colors="black",
        linewidths=1.2,
        linestyles="--",
        zorder=1,
    )


def figure_01(X: np.ndarray, y: np.ndarray) -> str:
    """Figura 1 — nuvens em ``s = 1.0``, centros teóricos e esboço de fronteiras."""
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    padding = 1.5
    xlim = (float(X[:, 0].min() - padding), float(X[:, 0].max() + padding))
    ylim = (float(X[:, 1].min() - padding), float(X[:, 1].max() + padding))

    _draw_nearest_center_regions(ax, xlim, ylim)
    _scatter_classes(ax, X, y)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(
        "Figura 1 — Quatro nuvens gaussianas ($s = 1.0$, 100 pontos por classe)\n"
        "com esboço geométrico de fronteiras (regra do centro mais próximo)"
    )

    handles, texts = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.2))
    texts.append("Esboço de fronteiras (não treinado)")
    ax.legend(handles, texts, loc="upper center", ncol=3, fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.2, linestyle=":")

    return save_figure(fig, "figure_01_point_clouds.png").name


def figure_02(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> str:
    """Figura 2 — os quatro datasets completos, um por escala, com eixos compartilhados."""
    all_points = np.vstack([X for X, _ in datasets.values()])
    padding = 2.0
    xlim = (float(all_points[:, 0].min() - padding), float(all_points[:, 0].max() + padding))
    ylim = (float(all_points[:, 1].min() - padding), float(all_points[:, 1].max() + padding))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)
    for index, (ax, scale) in enumerate(zip(axes.ravel(), datasets)):
        X, y = datasets[scale]
        _scatter_classes(ax, X, y)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"Dataset completo com $s = {scale}$ — 400 pontos (4 classes × 100)")
        # Eixos são compartilhados: rotular só a borda externa evita repetição.
        if index >= 2:
            ax.set_xlabel("$x_1$")
        if index % 2 == 0:
            ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.2, linestyle=":")

    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=5, fontsize=9.5, frameon=True)
    fig.suptitle(
        "Figura 2 — Quatro datasets independentes, cada um com as mesmas quatro classes,\n"
        "variando apenas a escala $s$ dos desvios-padrão (médias fixas; eixos compartilhados)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.93))
    return save_figure(fig, "figure_02_spread_scales.png").name


def figure_03(rates: dict[float, float]) -> str:
    """Figura 3 — mixing rate (em %) em função da escala ``s``."""
    scales = list(rates)
    values = [rates[s] * 100.0 for s in scales]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(
        scales,
        values,
        marker="o",
        markersize=9,
        linewidth=2.0,
        color="#d62728",
        label="Mixing rate (regra do centro mais próximo)",
    )
    for scale, value in zip(scales, values):
        ax.annotate(
            f"{value:.2f}%",
            (scale, value),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=9.5,
        )

    ax.set_xlabel("Escala dos desvios-padrão $s$")
    ax.set_ylabel("Mixing rate (% dos 400 pontos)")
    ax.set_title(
        "Figura 3 — Mixing rate versus escala de dispersão\n"
        "(fração de pontos cujo centro mais próximo não é o da própria classe)"
    )
    ax.set_xticks(scales)
    ax.set_ylim(-2.0, max(values) * 1.25 + 2.0)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper left")
    return save_figure(fig, "figure_03_mixing_rate.png").name


def run(rng: np.random.Generator) -> dict[str, Any]:
    """Executa o Exercise 1 inteiro e devolve todas as métricas citadas no relatório."""
    # Os quatro datasets são gerados uma única vez, em ordem crescente de escala.
    # O dataset do item A é o de s = 1.0 — a mesma amostra, para que a Figura 1 e o
    # painel s = 1.0 da Figura 2 sejam consistentes e nada seja gerado em dobro.
    datasets = {scale: generate(rng, scale) for scale in SCALES}

    for scale, (X, y) in datasets.items():
        assert X.shape == (400, 2), f"s={scale}: esperado (400, 2), obtido {X.shape}"
        counts = np.bincount(y, minlength=4).tolist()
        assert counts == [N_PER_CLASS] * 4, f"s={scale}: contagem por classe {counts}"

    X1, y1 = datasets[1.0]
    fig1 = figure_01(X1, y1)
    fig2 = figure_02(datasets)

    rates = {scale: mixing_rate(X, y) for scale, (X, y) in datasets.items()}
    fig3 = figure_03(rates)

    ratios = separation_ratios(scale=1.0)
    assert len(ratios) == 6, "a tabela de r_ij precisa ter exatamente 6 pares distintos"
    smallest = min(ratios, key=lambda row: row["r_ij"])

    print("Separation ratio r_ij em s = 1.0 (parâmetros teóricos do enunciado)")
    print(f"{'par':>7} | {'||mu_i-mu_j||':>14} | {'sigma_i':>8} | {'sigma_j':>8} | {'r_ij':>7}")
    print("-" * 60)
    for row in ratios:
        print(
            f"  ({row['i']},{row['j']}) | {row['distance']:14.4f} | {row['spread_i']:8.4f}"
            f" | {row['spread_j']:8.4f} | {row['r_ij']:7.4f}"
        )
    print(
        f"\nmenor r_ij em s = 1.0: {smallest['r_ij']:.4f} (par {smallest['i']}-{smallest['j']});"
        f" em s = 2.0 vale {smallest['r_ij'] / 2.0:.4f}"
    )

    print("\nMixing rate por escala")
    for scale, rate in rates.items():
        print(f"  s = {scale:<4} -> {rate:.4f}  ({rate * 100:.2f}% dos 400 pontos)")

    return {
        "n_total_per_dataset": 400,
        "n_per_class": N_PER_CLASS,
        "scales": SCALES,
        "separation_ratios_s1": ratios,
        "smallest_ratio_s1": smallest,
        "smallest_ratio_s2": smallest["r_ij"] / 2.0,
        "mixing_rates": {str(scale): rate for scale, rate in rates.items()},
        "figures": [fig1, fig2, fig3],
    }
