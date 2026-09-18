"""Exercise 2 — Non-Linearity in Higher Dimensions.

Constrói dois datasets em 5D com estruturas geométricas opostas:

* **Dataset I** — duas gaussianas multivariadas *deslocadas* (centros distintos,
  covariâncias distintas): a informação de classe está na posição do centro.
* **Dataset II** — duas *cascas concêntricas*: os centros praticamente coincidem
  e a informação de classe está no raio.

O módulo projeta ambos em 2D com PCA, mede a distância entre os centros
amostrais no espaço original de 5 dimensões e produz as Figuras 4 e 5.

Todas as funções recebem o mesmo ``rng`` de ``run_all.py``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import BINARY_COLORS, save_figure

N_PER_CLASS = 500
N_DIMS = 5

# --- Dataset I: gaussianas deslocadas -------------------------------------
MU_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
SIGMA_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)
MU_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
SIGMA_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

# --- Dataset II: cascas concêntricas --------------------------------------
# N(2.0, 0.4) e N(5.0, 0.4) são lidos como (média, desvio-padrão) — é a
# assinatura de ``Generator.normal``. Com desvio 0.4 as duas cascas ficam a
# 7.5 desvios uma da outra, que é a separação radial limpa que o exercício quer.
RHO_C = (2.0, 0.4)
RHO_D = (5.0, 0.4)

# O PCA do scikit-learn não aceita um ``Generator`` do NumPy, só um inteiro em
# ``random_state``; por isso a semente 42 aparece aqui como inteiro. Para esta
# forma de matriz o solver escolhido é o denso ("full"), que é determinístico —
# ``random_state`` não altera o resultado, e é passado apenas por explicitude.
PCA_RANDOM_STATE = 42


def sample_dataset_i(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dataset I — 500 amostras da classe A e 500 da classe B (rótulos 0 e 1)."""
    class_a = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    class_b = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([class_a, class_b])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def random_unit_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` direções uniformes na esfera unitária de 5D.

    Sorteia ``v ~ N(0, I_5)`` e normaliza. Um vetor exatamente nulo tem
    probabilidade zero, mas dividir por zero devolveria ``nan`` silenciosamente:
    as linhas degeneradas são reamostradas até que todas tenham norma positiva.
    """
    v = rng.normal(size=(n, N_DIMS))
    norms = np.linalg.norm(v, axis=1)
    while np.any(norms == 0.0):
        degenerate = norms == 0.0
        v[degenerate] = rng.normal(size=(int(degenerate.sum()), N_DIMS))
        norms = np.linalg.norm(v, axis=1)
    return v / norms[:, None]


def sample_dataset_ii(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dataset II — cascas concêntricas C (raio ≈ 2) e D (raio ≈ 5), rótulos 0 e 1.

    Os raios **não** são convertidos para valor absoluto: isso truncaria a
    distribuição especificada no enunciado.
    """
    directions_c = random_unit_directions(rng, N_PER_CLASS)
    rho_c = rng.normal(RHO_C[0], RHO_C[1], size=N_PER_CLASS)
    class_c = rho_c[:, None] * directions_c

    directions_d = random_unit_directions(rng, N_PER_CLASS)
    rho_d = rng.normal(RHO_D[0], RHO_D[1], size=N_PER_CLASS)
    class_d = rho_d[:, None] * directions_d

    X = np.vstack([class_c, class_d])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def center_distance(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Centros **amostrais** das duas classes em 5D e a distância euclidiana entre eles."""
    center_0 = X[y == 0].mean(axis=0)
    center_1 = X[y == 1].mean(axis=0)
    return {
        "center_0": center_0.tolist(),
        "center_1": center_1.tolist(),
        "distance": float(np.linalg.norm(center_0 - center_1)),
    }


def project_2d(X: np.ndarray) -> dict[str, Any]:
    """Ajusta um PCA de 2 componentes sobre as duas classes juntas e projeta."""
    pca = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    scores = pca.fit_transform(X)
    ratios = pca.explained_variance_ratio_
    assert scores.shape[1] == 2, "o PCA precisa devolver exatamente 2 componentes"
    return {
        "scores": scores,
        "pc1": float(ratios[0]),
        "pc2": float(ratios[1]),
        "total": float(ratios.sum()),
    }


def radius_stats(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Estatísticas descritivas de :math:`\\lVert x \\rVert` por classe."""
    stats = {}
    for label in (0, 1):
        radii = np.linalg.norm(X[y == label], axis=1)
        stats[str(label)] = {
            "mean": float(radii.mean()),
            "std": float(radii.std(ddof=1)),
            "min": float(radii.min()),
            "max": float(radii.max()),
        }
    return stats


def figure_04(datasets: dict[str, dict[str, Any]]) -> str:
    """Figura 4 — projeções PCA 2D lado a lado (Dataset I à esquerda, II à direita)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    for ax, (key, info) in zip(axes, datasets.items()):
        scores = info["pca"]["scores"]
        y = info["y"]
        for label, name, color in zip((0, 1), info["class_names"], BINARY_COLORS):
            points = scores[y == label]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=13,
                alpha=0.6,
                color=color,
                edgecolors="none",
                label=f"Classe {name}",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"{info['title']}\n"
            f"PC1 = {info['pca']['pc1'] * 100:.2f}%, PC2 = {info['pca']['pc2'] * 100:.2f}%"
            f" (soma = {info['pca']['total'] * 100:.2f}%)"
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2, linestyle=":")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "Figura 4 — Projeção PCA em 2 componentes, ajustada separadamente em cada dataset (5D → 2D)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_figure(fig, "figure_04_pca.png").name


def figure_05(datasets: dict[str, dict[str, Any]]) -> str:
    """Figura 5 — histogramas sobrepostos dos raios ``||x||`` no espaço original de 5D."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    for ax, (key, info) in zip(axes, datasets.items()):
        X, y = info["X"], info["y"]
        radii = np.linalg.norm(X, axis=1)
        # Mesmos bins para as duas classes dentro de cada comparação.
        bins = np.histogram_bin_edges(radii, bins=45)
        for label, name, color in zip((0, 1), info["class_names"], BINARY_COLORS):
            ax.hist(
                radii[y == label],
                bins=bins,
                alpha=0.55,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=f"Classe {name}",
            )
        ax.set_xlabel(r"Raio $\|x\|$ no espaço original de 5 dimensões")
        ax.set_ylabel("Frequência (nº de amostras)")
        ax.set_title(f"{info['title']}\ndistância entre centros 5D = {info['centers']['distance']:.4f}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2, linestyle=":")

    fig.suptitle(
        "Figura 5 — Distribuição dos raios por classe: o Dataset II separa em raio o que não separa em posição",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return save_figure(fig, "figure_05_radius_histograms.png").name


def run(rng: np.random.Generator) -> dict[str, Any]:
    """Executa o Exercise 2 inteiro e devolve todas as métricas citadas no relatório."""
    X1, y1 = sample_dataset_i(rng)
    X2, y2 = sample_dataset_ii(rng)

    for name, X, y in (("Dataset I", X1, y1), ("Dataset II", X2, y2)):
        assert X.shape == (2 * N_PER_CLASS, N_DIMS), f"{name}: forma inesperada {X.shape}"
        counts = np.bincount(y).tolist()
        assert counts == [N_PER_CLASS, N_PER_CLASS], f"{name}: contagem por classe {counts}"

    # As direções do Dataset II devem estar sobre a esfera unitária: o raio de
    # cada ponto tem de coincidir com |rho| usado para gerá-lo.
    unit_check = random_unit_directions(rng, 1000)
    unit_norms = np.linalg.norm(unit_check, axis=1)
    assert np.allclose(unit_norms, 1.0), "as direções sorteadas não têm norma 1"

    datasets = {
        "dataset_i": {
            "X": X1,
            "y": y1,
            "title": "Dataset I — gaussianas deslocadas (A vs. B)",
            "class_names": ("A", "B"),
            "centers": center_distance(X1, y1),
            "pca": project_2d(X1),
            "radii": radius_stats(X1, y1),
        },
        "dataset_ii": {
            "X": X2,
            "y": y2,
            "title": "Dataset II — cascas concêntricas (C vs. D)",
            "class_names": ("C", "D"),
            "centers": center_distance(X2, y2),
            "pca": project_2d(X2),
            "radii": radius_stats(X2, y2),
        },
    }

    fig4 = figure_04(datasets)
    fig5 = figure_05(datasets)

    # Limiar radial ilustrativo: ponto médio entre os raios médios observados.
    # É uma leitura das distribuições da Figura 5, não um parâmetro aprendido.
    radii_ii = datasets["dataset_ii"]["radii"]
    threshold = (radii_ii["0"]["mean"] + radii_ii["1"]["mean"]) / 2.0
    radii_all = np.linalg.norm(X2, axis=1)
    threshold_error = float(np.mean((radii_all > threshold).astype(int) != y2))

    for key, info in datasets.items():
        pca = info["pca"]
        print(f"{info['title']}")
        print(f"  distância entre centros amostrais (5D) = {info['centers']['distance']:.4f}")
        print(
            f"  variância explicada: PC1 = {pca['pc1']:.4f}, PC2 = {pca['pc2']:.4f},"
            f" soma = {pca['total']:.4f} ({pca['total'] * 100:.2f}%)"
        )
        for label, name in zip(("0", "1"), info["class_names"]):
            stats = info["radii"][label]
            print(
                f"  raio da classe {name}: média = {stats['mean']:.4f}, desvio = {stats['std']:.4f},"
                f" faixa = [{stats['min']:.4f}, {stats['max']:.4f}]"
            )
    print(
        f"\nLimiar radial ilustrativo para o Dataset II: ||x||^2 = {threshold ** 2:.4f}"
        f" (ou ||x|| = {threshold:.4f}) — erro geométrico = {threshold_error * 100:.2f}%"
    )

    return {
        "n_per_class": N_PER_CLASS,
        "n_dims": N_DIMS,
        "dataset_i": {
            "center_distance": datasets["dataset_i"]["centers"]["distance"],
            "pca": {k: v for k, v in datasets["dataset_i"]["pca"].items() if k != "scores"},
            "radii": datasets["dataset_i"]["radii"],
        },
        "dataset_ii": {
            "center_distance": datasets["dataset_ii"]["centers"]["distance"],
            "pca": {k: v for k, v in datasets["dataset_ii"]["pca"].items() if k != "scores"},
            "radii": datasets["dataset_ii"]["radii"],
        },
        "radial_threshold": threshold,
        "radial_threshold_squared": threshold**2,
        "radial_threshold_error": threshold_error,
        "unit_norm_max_deviation": float(np.abs(unit_norms - 1.0).max()),
        "figures": [fig4, fig5],
    }
