"""Ponto de entrada único da entrega *Perceptron*.

Gera os dois datasets, treina o perceptron escrito do zero em
:mod:`perceptron`, produz as seis figuras e imprime — além de salvar em
``results/metrics.json`` — todos os números citados no relatório.

Uso, a partir da raiz do repositório::

    python docs/exercises/perceptron/code/run_all.py

**A semente é criada aqui, uma única vez**, como ``rng =
np.random.default_rng(42)``, e o mesmo objeto é passado a cada geração de dados
e a cada treino, sempre na mesma ordem. Nada é reinicializado no meio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

# Backend sem interface gráfica: o script roda no GitHub Actions, que não tem display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptron import accuracy, predict, train  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
RESULTS = BASE / "results"
DPI = 150

SEED = 42
N_PER_CLASS = 1000
CLASS_COLORS = ("#1f77b4", "#d62728")

# Exercise 1 — nuvens bem separadas.
SEPARABLE = {
    0: {"mean": [1.5, 1.5], "cov": [[0.5, 0.0], [0.0, 0.5]]},
    1: {"mean": [5.0, 5.0], "cov": [[0.5, 0.0], [0.0, 0.5]]},
}
# Exercise 2 — médias próximas e dispersão três vezes maior.
OVERLAPPING = {
    0: {"mean": [3.0, 3.0], "cov": [[1.5, 0.0], [0.0, 1.5]]},
    1: {"mean": [4.0, 4.0], "cov": [[1.5, 0.0], [0.0, 1.5]]},
}

EXPECTED_FIGURES = [
    "figure_01_separable_scatter.png",
    "figure_02_separable_boundary.png",
    "figure_03_separable_accuracy.png",
    "figure_04_overlapping_scatter.png",
    "figure_05_overlapping_boundaries.png",
    "figure_06_overlapping_accuracy.png",
]


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Salva em ``figures/`` com nome determinístico e fecha a figura."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=DPI)
    plt.close(fig)
    return filename


def generate(rng: np.random.Generator, spec: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Amostra ``N_PER_CLASS`` pontos 2D por classe a partir de normais multivariadas."""
    parts_x, parts_y = [], []
    for label, params in spec.items():
        parts_x.append(
            rng.multivariate_normal(np.asarray(params["mean"]), np.asarray(params["cov"]), size=N_PER_CLASS)
        )
        parts_y.append(np.full(N_PER_CLASS, label))
    return np.vstack(parts_x), np.concatenate(parts_y)


def _scatter(ax: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    """Dispersão com uma cor por classe."""
    for label, color in zip((0, 1), CLASS_COLORS):
        points = X[y == label]
        ax.scatter(
            points[:, 0], points[:, 1], s=9, alpha=0.5, color=color,
            edgecolors="none", label=f"Classe {label}",
        )


def _boundary_line(w: np.ndarray, b: float, xlim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Pontos da reta ``w · x + b = 0`` dentro de ``xlim``.

    Isolando :math:`x_2`: :math:`x_2 = -(w_1 x_1 + b) / w_2`.
    """
    xs = np.linspace(*xlim, 200)
    return xs, -(w[0] * xs + b) / w[1]


def figure_scatter(X: np.ndarray, y: np.ndarray, title: str, filename: str) -> str:
    """Figuras 1 e 4 — dispersão dos 2000 pontos, uma cor por classe."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    _scatter(ax, X, y)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2, linestyle=":")
    return save_figure(fig, filename)


def figure_boundary(
    X: np.ndarray,
    y: np.ndarray,
    models: list[dict[str, Any]],
    title: str,
    filename: str,
) -> str:
    """Figuras 2 e 5 — fronteira(s) sobre os dados, com os erros destacados.

    ``models`` traz um ou dois conjuntos de pesos; os pontos mal classificados
    marcados são os do **primeiro** modelo da lista.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _scatter(ax, X, y)

    xlim = (float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5))
    ylim = (float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5))

    wrong = predict(X, models[0]["w"], models[0]["b"]) != y
    ax.scatter(
        X[wrong][:, 0], X[wrong][:, 1], s=46, facecolors="none",
        edgecolors="black", linewidths=0.8,
        label=f"Mal classificados por «{models[0]['label']}» ({int(wrong.sum())})",
    )

    for model, style in zip(models, ("-", "--")):
        xs, ys = _boundary_line(model["w"], model["b"], xlim)
        ax.plot(
            xs, ys, style, color=model.get("color", "black"), linewidth=2.2,
            label=f"Fronteira — {model['label']} (acurácia {model['accuracy'] * 100:.2f}%)",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.2, linestyle=":")
    return save_figure(fig, filename)


def figure_accuracy(
    curves: list[dict[str, Any]], title: str, filename: str, *, ylim: tuple[float, float] | None = None
) -> str:
    """Figuras 3 e 6 — acurácia (em %) por época."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for curve in curves:
        ax.plot(
            curve["epochs"], [v * 100 for v in curve["values"]],
            marker=curve.get("marker", "o"), markersize=4, linewidth=1.9,
            color=curve["color"], label=curve["label"],
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo (%)")
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right")
    return save_figure(fig, filename)


def exercise_1(rng: np.random.Generator) -> dict[str, Any]:
    """Exercise 1 — dados separáveis."""
    X, y = generate(rng, SEPARABLE)
    assert X.shape == (2 * N_PER_CLASS, 2) and np.bincount(y).tolist() == [N_PER_CLASS] * 2

    fig1 = figure_scatter(
        X, y,
        "Figura 1 — Dados separáveis: 1000 pontos por classe\n"
        r"($\mu_0 = [1.5, 1.5]$, $\mu_1 = [5, 5]$, $\Sigma = 0.5\,I$)",
        EXPECTED_FIGURES[0],
    )

    # A magnitude da inicialização e o ||x|| médio sustentam, no item D, o
    # argumento sobre o peso relativo entre o vetor inicial e as correções.
    # Reproduz o sorteio de train() sem consumir o rng principal.
    init_norm = float(np.linalg.norm(np.random.default_rng(SEED).normal(0.0, 0.01, size=2)))
    mean_norm = float(np.mean(np.linalg.norm(X, axis=1)))

    run = train(X, y, rng, eta=0.01, max_epochs=100)
    fig2 = figure_boundary(
        X, y,
        [{"w": run["w"], "b": run["b"], "accuracy": run["accuracy"], "label": "pesos finais", "color": "black"}],
        f"Figura 2 — Fronteira aprendida ($\\eta = 0.01$, {run['epochs']} épocas)\n"
        "com os pontos mal classificados destacados",
        EXPECTED_FIGURES[1],
    )
    fig3 = figure_accuracy(
        [{
            "epochs": [h["epoch"] for h in run["history"]],
            "values": [h["accuracy"] for h in run["history"]],
            "color": "#1f77b4", "label": r"Acurácia por época ($\eta = 0.01$)",
        }],
        "Figura 3 — Acurácia por época, dados separáveis\n"
        "(o treino para quando uma época inteira não produz nenhuma atualização)",
        EXPECTED_FIGURES[2],
    )

    # Mesmo treino, mudando só a taxa de aprendizado.
    run_fast = train(X, y, rng, eta=1.0, max_epochs=100)

    def direction(w: np.ndarray) -> np.ndarray:
        return w / np.linalg.norm(w)

    dir_slow, dir_fast = direction(run["w"]), direction(run_fast["w"])
    angle = float(np.degrees(np.arccos(np.clip(float(dir_slow @ dir_fast), -1.0, 1.0))))

    print("Exercise 1 — dados separáveis")
    print(f"  eta = 0.01 -> w = [{run['w'][0]:.6f}, {run['w'][1]:.6f}], b = {run['b']:.6f}")
    print(f"               épocas = {run['epochs']} (convergiu: {run['converged']}),"
          f" acurácia = {run['accuracy'] * 100:.2f}%")
    print(f"               direção w/||w|| = [{dir_slow[0]:.6f}, {dir_slow[1]:.6f}], ||w|| = {np.linalg.norm(run['w']):.6f}")
    print(f"  eta = 1.00 -> w = [{run_fast['w'][0]:.6f}, {run_fast['w'][1]:.6f}], b = {run_fast['b']:.6f}")
    print(f"               épocas = {run_fast['epochs']} (convergiu: {run_fast['converged']}),"
          f" acurácia = {run_fast['accuracy'] * 100:.2f}%")
    print(f"               direção w/||w|| = [{dir_fast[0]:.6f}, {dir_fast[1]:.6f}], ||w|| = {np.linalg.norm(run_fast['w']):.6f}")
    print(f"  ângulo entre as duas direções = {angle:.4f}°")
    print(f"  ||x|| médio = {mean_norm:.4f} -> passo em w por erro:"
          f" eta=0.01 soma ~{0.01 * mean_norm:.4f}, eta=1.0 soma ~{1.0 * mean_norm:.4f}")
    print(f"  magnitude típica da inicialização ||w_0|| ~ {init_norm:.6f};"
          f" razão ||w||/||w_0||: eta=0.01 -> {np.linalg.norm(run['w']) / init_norm:.1f}x,"
          f" eta=1.0 -> {np.linalg.norm(run_fast['w']) / init_norm:.1f}x")
    print(f"  atualizações por época (eta = 0.01): {[h['updates'] for h in run['history']]}")

    return {
        "n_per_class": N_PER_CLASS,
        "eta_slow": {
            "eta": 0.01,
            "w": run["w"].tolist(),
            "b": run["b"],
            "epochs": run["epochs"],
            "converged": run["converged"],
            "accuracy": run["accuracy"],
            "direction": dir_slow.tolist(),
            "w_norm": float(np.linalg.norm(run["w"])),
            "updates_per_epoch": [h["updates"] for h in run["history"]],
            "accuracy_per_epoch": [h["accuracy"] for h in run["history"]],
        },
        "eta_fast": {
            "eta": 1.0,
            "w": run_fast["w"].tolist(),
            "b": run_fast["b"],
            "epochs": run_fast["epochs"],
            "converged": run_fast["converged"],
            "accuracy": run_fast["accuracy"],
            "direction": dir_fast.tolist(),
            "w_norm": float(np.linalg.norm(run_fast["w"])),
            "updates_per_epoch": [h["updates"] for h in run_fast["history"]],
        },
        "direction_angle_degrees": angle,
        "mean_norm_x": mean_norm,
        "init_norm": init_norm,
        "step_in_w_slow": 0.01 * mean_norm,
        "step_in_w_fast": 1.0 * mean_norm,
        "ratio_to_init_slow": float(np.linalg.norm(run["w"]) / init_norm),
        "ratio_to_init_fast": float(np.linalg.norm(run_fast["w"]) / init_norm),
        "figures": [fig1, fig2, fig3],
    }


def exercise_2(rng: np.random.Generator) -> dict[str, Any]:
    """Exercise 2 — dados sobrepostos, com o pocket."""
    X, y = generate(rng, OVERLAPPING)
    assert X.shape == (2 * N_PER_CLASS, 2) and np.bincount(y).tolist() == [N_PER_CLASS] * 2

    fig4 = figure_scatter(
        X, y,
        "Figura 4 — Dados sobrepostos: 1000 pontos por classe\n"
        r"($\mu_0 = [3, 3]$, $\mu_1 = [4, 4]$, $\Sigma = 1.5\,I$)",
        EXPECTED_FIGURES[3],
    )

    # Mesma função do Exercise 1, sem nenhuma alteração.
    run = train(X, y, rng, eta=0.01, max_epochs=100)

    fig5 = figure_boundary(
        X, y,
        [
            {"w": run["w"], "b": run["b"], "accuracy": run["accuracy"], "label": "pesos finais", "color": "black"},
            {"w": run["pocket_w"], "b": run["pocket_b"], "accuracy": run["pocket_accuracy"],
             "label": "pesos pocket", "color": "#2ca02c"},
        ],
        "Figura 5 — Fronteiras final e pocket sobre os dados sobrepostos\n"
        "(destacados: os erros da fronteira final)",
        EXPECTED_FIGURES[4],
    )
    fig6 = figure_accuracy(
        [
            {"epochs": [h["epoch"] for h in run["history"]],
             "values": [h["accuracy"] for h in run["history"]],
             "color": "#d62728", "label": "Acurácia dos pesos atuais", "marker": "."},
            {"epochs": [h["epoch"] for h in run["history"]],
             "values": [h["pocket_accuracy"] for h in run["history"]],
             "color": "#2ca02c", "label": "Melhor acurácia já vista (pocket)", "marker": ""},
        ],
        "Figura 6 — Acurácia atual versus pocket, dados sobrepostos\n"
        "(o treino nunca para: sempre resta algum ponto do lado errado)",
        EXPECTED_FIGURES[5],
        ylim=(0, 100),
    )

    # Média de ||x|| do conjunto — sustenta o argumento do item D sobre o
    # tamanho relativo do passo em w e em b.
    mean_norm = float(np.mean(np.linalg.norm(X, axis=1)))

    print("\nExercise 2 — dados sobrepostos")
    print(f"  finais -> w = [{run['w'][0]:.6f}, {run['w'][1]:.6f}], b = {run['b']:.6f},"
          f" acurácia = {run['accuracy'] * 100:.2f}%")
    print(f"  pocket -> w = [{run['pocket_w'][0]:.6f}, {run['pocket_w'][1]:.6f}], b = {run['pocket_b']:.6f},"
          f" acurácia = {run['pocket_accuracy'] * 100:.2f}% (época {run['pocket_epoch']})")
    print(f"  épocas percorridas = {run['epochs']} (convergiu: {run['converged']})")
    print(f"  ||x|| médio = {mean_norm:.4f}")
    accs = [h["accuracy"] for h in run["history"]]
    print(f"  acurácia por época: mín = {min(accs) * 100:.2f}%, máx = {max(accs) * 100:.2f}%,"
          f" média = {np.mean(accs) * 100:.2f}%")

    return {
        "n_per_class": N_PER_CLASS,
        "eta": run["eta"],
        "final": {
            "w": run["w"].tolist(),
            "b": run["b"],
            "accuracy": run["accuracy"],
        },
        "pocket": {
            "w": run["pocket_w"].tolist(),
            "b": run["pocket_b"],
            "accuracy": run["pocket_accuracy"],
            "epoch": run["pocket_epoch"],
        },
        "epochs": run["epochs"],
        "converged": run["converged"],
        "mean_norm_x": mean_norm,
        "accuracy_per_epoch": accs,
        "pocket_per_epoch": [h["pocket_accuracy"] for h in run["history"]],
        "updates_per_epoch": [h["updates"] for h in run["history"]],
        "accuracy_stats": {
            "min": float(min(accs)),
            "max": float(max(accs)),
            "mean": float(np.mean(accs)),
            "last": accs[-1],
        },
        "figures": [fig4, fig5, fig6],
    }


def main() -> int:
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("Exercise 1 — Separable Data")
    print("=" * 78)
    metrics_1 = exercise_1(rng)

    print()
    print("=" * 78)
    print("Exercise 2 — Overlapping Data")
    print("=" * 78)
    metrics_2 = exercise_2(rng)

    missing = [name for name in EXPECTED_FIGURES if not (FIGURES / name).exists()]
    if missing:
        for name in missing:
            print(f"  [FALHA] figura não gerada: {name}", file=sys.stderr)
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"seed": SEED, "exercise_1": metrics_1, "exercise_2": metrics_2}
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{len(EXPECTED_FIGURES)} figuras geradas em {FIGURES}")
    print(f"métricas salvas em {RESULTS / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
