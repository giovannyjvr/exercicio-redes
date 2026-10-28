"""Ponto de entrada único da entrega *MLP*.

Executa os quatro exercícios em ordem fixa, produz as seis figuras e imprime —
além de salvar em ``results/metrics.json`` — todos os números citados no
relatório.

Uso, a partir da raiz do repositório::

    python docs/exercises/mlp/code/run_all.py

A semente da parte NumPy é criada aqui, uma única vez, como
``rng = np.random.default_rng(42)``, e o mesmo objeto inicializa cada rede. Os
geradores do scikit-learn recebem ``random_state=42`` porque não aceitam um
``Generator`` do NumPy. A única exceção deliberada é o Exercise 4, que o próprio
enunciado manda repetir com pelo menos três sementes diferentes: ali as
inicializações usam ``default_rng(42)``, ``(43)`` e ``(44)``, documentado no
relatório.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sem display: o script roda no GitHub Actions

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs, make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exercise1_by_hand  # noqa: E402
from mlp import MLP, one_hot  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
RESULTS = BASE / "results"
DPI = 150

SEED = 42
SKLEARN_SEED = 42
DEPTH_SEEDS = (42, 43, 44)  # Exercise 4 pede repetição com >= 3 sementes
CLASS_COLORS = ("#1f77b4", "#d62728", "#2ca02c")

EXPECTED_FIGURES = [
    "figure_01_blobs_scatter.png",
    "figure_02_binary_loss.png",
    "figure_03_decision_boundary.png",
    "figure_04_multiclass_loss.png",
    "figure_05_confusion_matrix.png",
    "figure_06_depth_comparison.png",
]


def save_figure(fig: plt.Figure, filename: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=DPI)
    plt.close(fig)
    return filename


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------- Exercise 1


def exercise_1() -> dict[str, Any]:
    """Um passo de retropropagação, calculado explicitamente."""
    result = exercise1_by_hand.run()
    exercise1_by_hand.report(result)
    return result


# ---------------------------------------------------------------- Exercise 2


def exercise_2(rng: np.random.Generator) -> dict[str, Any]:
    """Classificação binária que nenhuma reta resolve."""
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(
        n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=SKLEARN_SEED
    )
    # Classe 0 = 1 cluster central; classe 1 = 2 clusters em lados opostos.
    y = (c > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SKLEARN_SEED, stratify=y
    )

    # Figura 1 — dispersão dos 1000 pontos coloridos por y.
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for label, color in zip((0, 1), CLASS_COLORS):
        pts = X[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, color=color,
                   edgecolors="none", label=f"Classe {label}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — Dois clusters da classe 1 cercando a classe 0\n"
                 "(make_blobs, 1000 pontos, cluster_std = 1.2)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2, linestyle=":")
    fig1 = save_figure(fig, EXPECTED_FIGURES[0])

    # B — baseline linear. É a referência, não o modelo.
    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    baseline_accuracy = accuracy(y_test, baseline.predict(X_test))

    # Geometria do baseline, para o item D. Duas perguntas que o texto responde:
    # (i) onde a fronteira logística se posiciona, e (ii) qual seria a melhor
    # acurácia alcançável por *alguma* reta — obtida por busca exaustiva em
    # ângulo e deslocamento, não por otimização de alguma perda substituta.
    w_base, b_base = baseline.coef_[0], float(baseline.intercept_[0])
    baseline_geometry = {
        "w": w_base.tolist(),
        "b": b_base,
        "distance_to_origin": float(abs(b_base) / np.linalg.norm(w_base)),
        "recall_class_0": float(np.mean(baseline.predict(X[y == 0]) == 0)),
        "recall_class_1": float(np.mean(baseline.predict(X[y == 1]) == 1)),
    }
    best_line = 0.0
    for angle in np.linspace(0.0, np.pi, 721):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = X_test @ direction
        for offset in np.unique(np.round(projection, 3)):
            for sign in (1.0, -1.0):
                acc = accuracy(y_test, (sign * (projection - offset) >= 0).astype(int))
                best_line = max(best_line, acc)
    baseline_geometry["best_possible_line"] = float(best_line)

    # C — MLP do zero.
    layer_sizes = [2, 16, 1]
    learning_rate, epochs = 0.5, 3000
    model = MLP(layer_sizes, rng, output="sigmoid")
    history = model.fit(
        X_train, y_train.reshape(-1, 1).astype(float), epochs=epochs, learning_rate=learning_rate
    )
    test_accuracy = accuracy(y_test, model.predict(X_test))
    train_accuracy = accuracy(y_train, model.predict(X_train))

    # Figura 2 — perda de treino por época.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(range(1, epochs + 1), history, color="#1f77b4", linewidth=1.8,
            label="Entropia cruzada binária (treino)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda média por amostra (nats)")
    ax.set_title(f"Figura 2 — Perda de treino, MLP {model.architecture()}\n"
                 f"(lr = {learning_rate}, {epochs} épocas, full batch)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend()
    fig2 = save_figure(fig, EXPECTED_FIGURES[1])

    # Figura 3 — fronteira de decisão avaliada numa malha, sobre os pontos de teste.
    pad = 1.0
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    zones = model.predict(grid).reshape(mesh_x.shape)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.contourf(mesh_x, mesh_y, zones, levels=[-0.5, 0.5, 1.5],
                colors=CLASS_COLORS[:2], alpha=0.18)
    ax.contour(mesh_x, mesh_y, zones, levels=[0.5], colors="black", linewidths=1.6)
    for label, color in zip((0, 1), CLASS_COLORS):
        pts = X_test[y_test == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.85, color=color,
                   edgecolors="white", linewidths=0.4, label=f"Teste — classe {label}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Figura 3 — Regiões de decisão do MLP sobre os pontos de teste\n"
                 f"(acurácia de teste {test_accuracy * 100:.2f}%)")
    ax.legend(loc="upper left")
    fig3 = save_figure(fig, EXPECTED_FIGURES[2])

    print("\nExercise 2 — classificação binária")
    print(f"  baseline LogisticRegression: acurácia de teste = {baseline_accuracy * 100:.2f}%")
    print(f"    fronteira a {baseline_geometry['distance_to_origin']:.4f} da origem |"
          f" acerta {baseline_geometry['recall_class_0'] * 100:.2f}% da classe 0 e"
          f" {baseline_geometry['recall_class_1'] * 100:.2f}% da classe 1")
    print(f"    melhor reta possível (busca exaustiva no teste) ="
          f" {baseline_geometry['best_possible_line'] * 100:.2f}%")
    print(f"  MLP {model.architecture()} | lr = {learning_rate} | épocas = {epochs}"
          f" | parâmetros = {model.n_parameters()}")
    print(f"  acurácia: treino = {train_accuracy * 100:.2f}%, teste = {test_accuracy * 100:.2f}%")
    print(f"  perda de treino: inicial = {history[0]:.6f}, final = {history[-1]:.6f}")

    return {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline_accuracy": baseline_accuracy,
        "baseline_geometry": baseline_geometry,
        "architecture": model.architecture(),
        "layer_sizes": layer_sizes,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "n_parameters": model.n_parameters(),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "loss_first": history[0],
        "loss_final": history[-1],
        "figures": [fig1, fig2, fig3],
    }


# ---------------------------------------------------------------- Exercise 3


def make_multiclass() -> tuple[np.ndarray, ...]:
    """Dataset dos Exercises 3 e 4 — gerado uma vez e reutilizado."""
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
        n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=SKLEARN_SEED,
    )
    return train_test_split(X, y, test_size=0.2, random_state=SKLEARN_SEED, stratify=y)


def train_multiclass(
    layer_sizes: list[int], rng: np.random.Generator, data: tuple[np.ndarray, ...],
    *, learning_rate: float, epochs: int,
) -> dict[str, Any]:
    """Treina a rede multiclasse — o mesmo núcleo do Exercise 2, com softmax."""
    X_train, X_test, y_train, y_test = data
    model = MLP(layer_sizes, rng, output="softmax")
    history = model.fit(
        X_train, one_hot(y_train, 3), epochs=epochs, learning_rate=learning_rate
    )
    return {
        "model": model,
        "history": history,
        "train_accuracy": accuracy(y_train, model.predict(X_train)),
        "test_accuracy": accuracy(y_test, model.predict(X_test)),
        "loss_final": history[-1],
    }


def exercise_3(rng: np.random.Generator, data: tuple[np.ndarray, ...]) -> dict[str, Any]:
    """Multiclasse com o mesmo núcleo, trocando só saída e perda."""
    X_train, X_test, y_train, y_test = data
    layer_sizes, learning_rate, epochs = [4, 32, 3], 0.5, 4000

    run = train_multiclass(layer_sizes, rng, data, learning_rate=learning_rate, epochs=epochs)
    model, history = run["model"], run["history"]

    # O scikit-learn >= 1.9 removeu o parâmetro multi_class; o padrão para 3
    # classes já é a regressão logística multinomial (softmax).
    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    baseline_accuracy = accuracy(y_test, baseline.predict(X_test))

    # Figura 4 — perda de treino por época.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(range(1, epochs + 1), history, color="#2ca02c", linewidth=1.8,
            label="Entropia cruzada categórica (treino)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda média por amostra (nats)")
    ax.set_title(f"Figura 4 — Perda de treino, MLP {model.architecture()}\n"
                 f"(lr = {learning_rate}, {epochs} épocas, 3 classes)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend()
    fig4 = save_figure(fig, EXPECTED_FIGURES[3])

    # Figura 5 — matriz de confusão no teste.
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Figura 5 — Matriz de confusão no teste\n"
                 f"(acurácia {run['test_accuracy'] * 100:.2f}%, {len(y_test)} amostras)")
    fig.colorbar(im, ax=ax, label="Nº de amostras")
    fig5 = save_figure(fig, EXPECTED_FIGURES[4])

    # Par de classes mais confundido: soma os dois sentidos fora da diagonal.
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            pairs.append({"i": i, "j": j, "errors": int(cm[i, j] + cm[j, i]),
                          "i_to_j": int(cm[i, j]), "j_to_i": int(cm[j, i])})
    worst = max(pairs, key=lambda p: p["errors"])
    total_errors = int(cm.sum() - np.trace(cm))

    print("\nExercise 3 — multiclasse")
    print(f"  baseline LogisticRegression: acurácia de teste = {baseline_accuracy * 100:.2f}%")
    print(f"  MLP {model.architecture()} | lr = {learning_rate} | épocas = {epochs}"
          f" | parâmetros = {model.n_parameters()}")
    print(f"  acurácia: treino = {run['train_accuracy'] * 100:.2f}%,"
          f" teste = {run['test_accuracy'] * 100:.2f}%")
    print(f"  matriz de confusão:\n{cm}")
    print(f"  par mais confundido: classes {worst['i']} e {worst['j']}"
          f" ({worst['errors']} erros de {total_errors}:"
          f" {worst['i']}->{worst['j']} = {worst['i_to_j']},"
          f" {worst['j']}->{worst['i']} = {worst['j_to_i']})")

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline_accuracy": baseline_accuracy,
        "architecture": model.architecture(),
        "layer_sizes": layer_sizes,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "n_parameters": model.n_parameters(),
        "train_accuracy": run["train_accuracy"],
        "test_accuracy": run["test_accuracy"],
        "loss_first": history[0],
        "loss_final": history[-1],
        "confusion_matrix": cm.tolist(),
        "class_pairs": pairs,
        "worst_pair": worst,
        "total_errors": total_errors,
        "history": history,
        "figures": [fig4, fig5],
    }


# ---------------------------------------------------------------- Exercise 4


def exercise_4(data: tuple[np.ndarray, ...], shallow: dict[str, Any]) -> dict[str, Any]:
    """Comparação controlada: só a profundidade muda."""
    learning_rate = shallow["learning_rate"]
    epochs = shallow["epochs"]
    shallow_sizes = shallow["layer_sizes"]
    deep_sizes = [4, 32, 32, 3]

    # Mesmos dados, mesmo orçamento, mesma taxa; repetido com 3 sementes.
    runs: dict[str, list[dict[str, Any]]] = {"shallow": [], "deep": []}
    curves: dict[str, list[float]] = {}
    for seed in DEPTH_SEEDS:
        for name, sizes in (("shallow", shallow_sizes), ("deep", deep_sizes)):
            run = train_multiclass(
                sizes, np.random.default_rng(seed), data,
                learning_rate=learning_rate, epochs=epochs,
            )
            runs[name].append({
                "seed": seed,
                "test_accuracy": run["test_accuracy"],
                "train_accuracy": run["train_accuracy"],
                "loss_final": run["loss_final"],
                "n_parameters": run["model"].n_parameters(),
                "architecture": run["model"].architecture(),
            })
            if seed == DEPTH_SEEDS[0]:
                curves[name] = run["history"]

    def summarize(name: str) -> dict[str, Any]:
        rows = runs[name]
        return {
            "architecture": rows[0]["architecture"],
            "hidden_layers": len(rows[0]["architecture"].split("–")) - 2,
            "n_parameters": rows[0]["n_parameters"],
            "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in rows])),
            "test_accuracy_std": float(np.std([r["test_accuracy"] for r in rows], ddof=1)),
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in rows])),
            "loss_final_mean": float(np.mean([r["loss_final"] for r in rows])),
            "per_seed": rows,
        }

    summary = {"shallow": summarize("shallow"), "deep": summarize("deep")}
    difference = summary["deep"]["test_accuracy_mean"] - summary["shallow"]["test_accuracy_mean"]

    # Figura 6 — curvas de perda das duas redes nos mesmos eixos.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(range(1, epochs + 1), curves["shallow"], color="#2ca02c", linewidth=1.8,
            label=f"Exercise 3 — {summary['shallow']['architecture']} (1 camada escondida)")
    ax.plot(range(1, epochs + 1), curves["deep"], color="#9467bd", linewidth=1.8,
            label=f"Exercise 4 — {summary['deep']['architecture']} (2 camadas escondidas)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda média por amostra (nats)")
    ax.set_title("Figura 6 — Perda de treino: uma camada escondida contra duas\n"
                 f"(mesmos dados, mesma lr = {learning_rate}, mesmas {epochs} épocas, semente {DEPTH_SEEDS[0]})")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend()
    fig6 = save_figure(fig, EXPECTED_FIGURES[5])

    print("\nExercise 4 — a profundidade ajuda?")
    for name in ("shallow", "deep"):
        s = summary[name]
        accs = ", ".join(f"{r['test_accuracy'] * 100:.2f}%" for r in s["per_seed"])
        print(f"  {s['architecture']:<12} | camadas escondidas = {s['hidden_layers']}"
              f" | parâmetros = {s['n_parameters']:>5}"
              f" | perda final média = {s['loss_final_mean']:.6f}"
              f" | teste médio = {s['test_accuracy_mean'] * 100:.2f}%"
              f" (±{s['test_accuracy_std'] * 100:.2f}) [{accs}]")
    print(f"  diferença (profunda − rasa) = {difference * 100:+.2f} pontos percentuais")

    return {
        "seeds": list(DEPTH_SEEDS),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "summary": summary,
        "difference_mean": difference,
        "figures": [fig6],
    }


# ---------------------------------------------------------------- orquestração


def main() -> int:
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("Exercise 1 — One Step of Backpropagation, by Hand")
    print("=" * 78)
    metrics_1 = exercise_1()

    print()
    print("=" * 78)
    print("Exercise 2 — Binary Classification a Straight Line Cannot Do")
    print("=" * 78)
    metrics_2 = exercise_2(rng)

    print()
    print("=" * 78)
    print("Exercise 3 — Multi-Class, Same Network")
    print("=" * 78)
    data = make_multiclass()
    metrics_3 = exercise_3(rng, data)

    print()
    print("=" * 78)
    print("Exercise 4 — Does a Second Hidden Layer Help?")
    print("=" * 78)
    metrics_4 = exercise_4(data, metrics_3)

    missing = [name for name in EXPECTED_FIGURES if not (FIGURES / name).exists()]
    if missing:
        for name in missing:
            print(f"  [FALHA] figura não gerada: {name}", file=sys.stderr)
        return 1

    # A curva do Exercise 3 é longa e já está na Figura 4; não vai para o JSON.
    metrics_3 = {k: v for k, v in metrics_3.items() if k != "history"}

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "sklearn_seed": SKLEARN_SEED,
        "exercise_1": metrics_1,
        "exercise_2": metrics_2,
        "exercise_3": metrics_3,
        "exercise_4": metrics_4,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{len(EXPECTED_FIGURES)} figuras geradas em {FIGURES}")
    print(f"métricas salvas em {RESULTS / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
