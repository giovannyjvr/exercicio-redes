"""Ponto de entrada único da entrega *VAE*.

Treina os dois modelos (d = 16 e d = 2), produz as seis figuras e imprime —
além de salvar em ``results/metrics.json`` — todos os números citados no
relatório.

Uso, a partir da raiz do repositório::

    python docs/exercises/vae/code/run_all.py

Sementes, conforme o enunciado: ``torch.manual_seed(42)`` **antes de construir
cada modelo**, e o split 50 000 / 10 000 com
``torch.Generator().manual_seed(42)``. A grade de amostras do prior da Figura 3
usa ``torch.manual_seed(0)``, também como pedido.

O conjunto de teste é usado **apenas** para os números finais de cada modelo;
todo acompanhamento durante o treino é feito na validação.
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
import torch  # noqa: E402
from scipy.stats import norm  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vae import VAE, evaluate, train_epoch  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
RESULTS = BASE / "results"
DATA = BASE / "data"
DPI = 150

SEED = 42
SAMPLE_SEED = 0          # semente da grade de amostras do prior (Figura 3)
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
BETA = 1.0

EXPECTED_FIGURES = [
    "figure_01_loss_curves.png",
    "figure_02_reconstructions.png",
    "figure_03_prior_samples.png",
    "figure_04_latent_scatter.png",
    "figure_05_decoded_plane.png",
    "figure_06_interpolation.png",
]


def save_figure(fig: plt.Figure, filename: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=DPI)
    plt.close(fig)
    return filename


def load_data() -> dict[str, Any]:
    """Carrega o MNIST e faz o split exato pedido pelo enunciado."""
    transform = transforms.ToTensor()  # já coloca os pixels em [0, 1]
    train_full = datasets.MNIST(str(DATA), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(DATA), train=False, download=True, transform=transform)

    train, val = random_split(
        train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED)
    )

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_loader": DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        "val_loader": DataLoader(val, batch_size=512, shuffle=False),
        "test_loader": DataLoader(test, batch_size=512, shuffle=False),
    }


def train_model(latent_dim: int, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Treina um VAE e registra BCE e KL, separadamente, em treino e validação."""
    # Semente fixada imediatamente antes de construir o modelo, para que as duas
    # corridas (d = 16 e d = 2) partam da mesma inicialização pseudo-aleatória.
    torch.manual_seed(SEED)
    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_bce": [], "train_kl": [], "val_bce": [], "val_kl": []}
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_stats = train_epoch(model, data["train_loader"], optimizer, device, BETA)
        val_stats = evaluate(model, data["val_loader"], device, BETA)
        history["train_bce"].append(train_stats["bce"])
        history["train_kl"].append(train_stats["kl"])
        history["val_bce"].append(val_stats["bce"])
        history["val_kl"].append(val_stats["kl"])
        print(
            f"  d={latent_dim:>2} época {epoch:>2}/{EPOCHS} | "
            f"treino BCE {train_stats['bce']:7.3f} KL {train_stats['kl']:6.3f} | "
            f"val BCE {val_stats['bce']:7.3f} KL {val_stats['kl']:6.3f}"
        )
    elapsed = time.perf_counter() - started

    # O teste só é tocado aqui, para os números finais.
    test_stats = evaluate(model, data["test_loader"], device, BETA)
    print(
        f"  d={latent_dim} TESTE: BCE {test_stats['bce']:.3f} | KL {test_stats['kl']:.3f}"
        f" | -ELBO {test_stats['neg_elbo']:.3f} nats/imagem  ({elapsed:.1f}s)"
    )
    return {"model": model, "history": history, "test": test_stats, "seconds": elapsed}


# ---------------------------------------------------------------- figuras


def figure_01(history: dict[str, list[float]]) -> str:
    """Figura 1 — reconstrução e KL por época, treino e validação."""
    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    axes[0].plot(epochs, history["train_bce"], color="#1f77b4", marker="o", markersize=3, label="Treino")
    axes[0].plot(epochs, history["val_bce"], color="#d62728", marker="s", markersize=3, label="Validação")
    axes[0].set_title("Termo de reconstrução (BCE)")
    axes[0].set_ylabel("Nats por imagem (soma sobre 784 pixels)")

    axes[1].plot(epochs, history["train_kl"], color="#1f77b4", marker="o", markersize=3, label="Treino")
    axes[1].plot(epochs, history["val_kl"], color="#d62728", marker="s", markersize=3, label="Validação")
    axes[1].set_title("Termo de regularização (KL)")
    axes[1].set_ylabel("Nats por imagem (soma sobre as dimensões latentes)")

    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend()
    fig.suptitle("Figura 1 — Os dois termos da perda, separados ($d = 16$, $\\beta = 1$)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_figure(fig, EXPECTED_FIGURES[0])


def figure_02(model: VAE, data: dict[str, Any], device: torch.device) -> tuple[str, list[int]]:
    """Figura 2 — primeira imagem de teste de cada dígito e sua reconstrução de mu."""
    test = data["test"]
    targets = test.targets.numpy()
    # Primeira ocorrência de cada dígito 0-9 no conjunto de teste.
    indices = [int(np.where(targets == digit)[0][0]) for digit in range(10)]
    images = torch.stack([test[i][0].view(-1) for i in indices]).to(device)

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(images)
        # Reconstrói decodificando mu — sem amostrar z (ver item E).
        reconstructions = model.decode(mu)

    fig, axes = plt.subplots(2, 10, figsize=(14.0, 3.4))
    for column, digit in enumerate(range(10)):
        axes[0, column].imshow(images[column].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(reconstructions[column].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"{digit}", fontsize=11)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstrução", fontsize=10)
    fig.suptitle(
        "Figura 2 — Primeira imagem de teste de cada dígito (acima) e sua reconstrução (abaixo)\n"
        "obtida decodificando $\\mu$, sem amostrar $z$  —  modelo $d = 16$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return save_figure(fig, EXPECTED_FIGURES[1]), indices


def figure_03(model: VAE, device: torch.device) -> str:
    """Figura 3 — grade 8x8 decodificada de z ~ N(0, I), após torch.manual_seed(0)."""
    torch.manual_seed(SAMPLE_SEED)
    model.eval()
    samples = model.sample(64, device).cpu()

    fig, axes = plt.subplots(8, 8, figsize=(8.0, 8.4))
    for index, ax in enumerate(axes.ravel()):
        ax.imshow(samples[index].view(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Figura 3 — 64 amostras do prior $z \\sim \\mathcal{N}(0, I)$, decodificadas\n"
        f"($d = 16$, torch.manual_seed({SAMPLE_SEED}))",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return save_figure(fig, EXPECTED_FIGURES[2])


def encode_test_set(model: VAE, data: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Devolve ``mu`` de todas as 10 000 imagens de teste e seus rótulos."""
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for images, targets in data["test_loader"]:
            x = images.view(images.shape[0], -1).to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(mus), np.concatenate(labels)


def figure_04(mu: np.ndarray, labels: np.ndarray) -> str:
    """Figura 4 — mu das 10 000 imagens de teste, colorido por dígito."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    colormap = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels == digit
        ax.scatter(mu[mask, 0], mu[mask, 1], s=4, alpha=0.5,
                   color=colormap(digit), edgecolors="none", label=str(digit))
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 — Espaço latente do modelo $d = 2$\n"
                 "($\\mu$ das 10 000 imagens de teste, colorido por dígito)")
    legend = ax.legend(title="Dígito", loc="upper right", markerscale=3, fontsize=9)
    legend.get_title().set_fontsize(9)
    ax.grid(alpha=0.2, linestyle=":")
    return save_figure(fig, EXPECTED_FIGURES[3])


def latent_overlap(mu: np.ndarray, labels: np.ndarray, n_neighbors: int = 10) -> dict[str, Any]:
    """Mede quais dígitos se sobrepõem no espaço latente, em vez de descrevê-los de olho.

    Para cada uma das 10 000 imagens de teste, olha seus ``n_neighbors`` vizinhos
    mais próximos em :math:`\\mu` e conta quantos pertencem a outro dígito. A
    contagem agregada por par é uma medida direta de quanto duas nuvens se
    interpenetram — exatamente o que o item E pergunta sobre a Figura 4.
    """
    from sklearn.neighbors import NearestNeighbors

    # n_neighbors + 1 porque o primeiro vizinho de um ponto é ele mesmo.
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(mu)
    _, neighbors = finder.kneighbors(mu)
    neighbor_labels = labels[neighbors[:, 1:]]

    pair_counts: dict[tuple[int, int], int] = {}
    impurity = float(np.mean(neighbor_labels != labels[:, None]))
    for own, others in zip(labels, neighbor_labels):
        for other in others:
            if other != own:
                pair = (int(min(own, other)), int(max(own, other)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    total = sum(pair_counts.values())
    ranked = sorted(pair_counts.items(), key=lambda item: -item[1])
    return {
        "neighbor_impurity": impurity,
        "top_pairs": [
            {"a": a, "b": b, "count": n, "share": n / total} for (a, b), n in ranked[:6]
        ],
        "total_cross_links": total,
    }


def figure_05(model: VAE, device: torch.device) -> str:
    """Figura 5 — grade 15x15 decodificada nos quantis do prior."""
    grid_size = 15
    # z = Phi^-1(q), com q igualmente espaçado em [0.05, 0.95]: cada célula passa
    # a cobrir a mesma massa de probabilidade do prior (ver explicação no relatório).
    quantiles = np.linspace(0.05, 0.95, grid_size)
    coords = norm.ppf(quantiles)

    model.eval()
    # Linha de cima = maior z2, para que a figura case com a orientação da Figura 4.
    z = torch.tensor(
        [[x, y] for y in coords[::-1] for x in coords], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        images = model.decode(z).cpu().view(-1, 28, 28).numpy()

    canvas = np.zeros((grid_size * 28, grid_size * 28))
    for index, image in enumerate(images):
        row, column = divmod(index, grid_size)
        canvas[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28] = image

    fig, ax = plt.subplots(figsize=(8.5, 8.8))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ticks = np.arange(grid_size) * 28 + 14
    ax.set_xticks(ticks[::2], [f"{c:.2f}" for c in coords[::2]], fontsize=8)
    ax.set_yticks(ticks[::2], [f"{c:.2f}" for c in coords[::-1][::2]], fontsize=8)
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    ax.set_title("Figura 5 — O plano latente decodificado ($d = 2$)\n"
                 "grade 15 × 15 nos quantis $q \\in [0.05,\\ 0.95]$ do prior")
    return save_figure(fig, EXPECTED_FIGURES[4])


def figure_06(
    model16: VAE, data: dict[str, Any], device: torch.device, indices: tuple[int, int]
) -> str:
    """Figura 6 — interpolação em pixels e no espaço latente do modelo d = 16."""
    steps = 10
    test = data["test"]
    index_a, index_b = indices
    x_a = test[index_a][0].view(-1).to(device)
    x_b = test[index_b][0].view(-1).to(device)
    digit_a, digit_b = int(test.targets[index_a]), int(test.targets[index_b])

    model16.eval()
    with torch.no_grad():
        mu_pair, _ = model16.encode(torch.stack([x_a, x_b]))
        ts = torch.linspace(0.0, 1.0, steps, device=device)
        # Linha de cima: mistura dos próprios pixels.
        pixel_row = torch.stack([(1 - t) * x_a + t * x_b for t in ts])
        # Linha de baixo: mistura dos mu e decodificação.
        latent_z = torch.stack([(1 - t) * mu_pair[0] + t * mu_pair[1] for t in ts])
        latent_row = model16.decode(latent_z)

    fig, axes = plt.subplots(2, steps, figsize=(14.0, 3.6))
    for column in range(steps):
        axes[0, column].imshow(pixel_row[column].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(latent_row[column].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"t = {ts[column]:.2f}", fontsize=8.5)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Pixels", fontsize=10)
    axes[1, 0].set_ylabel("Latente", fontsize=10)
    fig.suptitle(
        f"Figura 6 — Interpolação entre o teste #{index_a} (dígito {digit_a}) e"
        f" o #{index_b} (dígito {digit_b})\n"
        "acima: no espaço de pixels  |  abaixo: no espaço latente do modelo $d = 16$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    return save_figure(fig, EXPECTED_FIGURES[5]), (digit_a, digit_b)


# ---------------------------------------------------------------- orquestração


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | torch {torch.__version__}")

    data = load_data()
    print(f"Split: treino {len(data['train'])}, validação {len(data['val'])}, teste {len(data['test'])}")

    print("\n" + "=" * 78)
    print("Exercise 1 — A VAE on MNIST (d = 16)")
    print("=" * 78)
    run16 = train_model(16, data, device)
    fig1 = figure_01(run16["history"])
    fig2, digit_indices = figure_02(run16["model"], data, device)
    fig3 = figure_03(run16["model"], device)

    print("\n" + "=" * 78)
    print("Exercise 2 — A Latent Space You Can See (d = 2)")
    print("=" * 78)
    run2 = train_model(2, data, device)
    mu2, labels2 = encode_test_set(run2["model"], data, device)
    fig4 = figure_04(mu2, labels2)
    overlap = latent_overlap(mu2, labels2)
    print("\n  Sobreposição no espaço latente (vizinhos de outro dígito, k = 10):")
    print(f"    impureza média da vizinhança = {overlap['neighbor_impurity'] * 100:.2f}%")
    for row in overlap["top_pairs"]:
        print(f"    dígitos {row['a']} e {row['b']}: {row['count']:>5} ligações"
              f" ({row['share'] * 100:.1f}% das cruzadas)")
    fig5 = figure_05(run2["model"], device)

    # Dois dígitos diferentes para a interpolação: a primeira imagem de teste que
    # é um 0 e a primeira que é um 8 — escolha fixa, não aleatória.
    targets = data["test"].targets.numpy()
    index_a = int(np.where(targets == 0)[0][0])
    index_b = int(np.where(targets == 8)[0][0])
    fig6, (digit_a, digit_b) = figure_06(run16["model"], data, device, (index_a, index_b))

    print(f"\nInterpolação: teste #{index_a} (dígito {digit_a}) -> #{index_b} (dígito {digit_b})")

    missing = [name for name in EXPECTED_FIGURES if not (FIGURES / name).exists()]
    if missing:
        for name in missing:
            print(f"  [FALHA] figura não gerada: {name}", file=sys.stderr)
        return 1

    history16 = run16["history"]
    payload = {
        "seed": SEED,
        "sample_seed": SAMPLE_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE, "beta": BETA,
        },
        "split": {"train": len(data["train"]), "val": len(data["val"]), "test": len(data["test"])},
        "exercise_1": {
            "latent_dim": 16,
            "test": run16["test"],
            "history": history16,
            "val_kl_first_epoch": history16["val_kl"][0],
            "val_kl_last_epoch": history16["val_kl"][-1],
            "val_bce_first_epoch": history16["val_bce"][0],
            "val_bce_last_epoch": history16["val_bce"][-1],
            "digit_indices": digit_indices,
            "seconds": run16["seconds"],
            "figures": [fig1, fig2, fig3],
        },
        "exercise_2": {
            "latent_dim": 2,
            "test": run2["test"],
            "history": run2["history"],
            "latent_overlap": overlap,
            "interpolation": {
                "index_a": index_a, "digit_a": digit_a,
                "index_b": index_b, "digit_b": digit_b,
            },
            "seconds": run2["seconds"],
            "figures": [fig4, fig5, fig6],
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{len(EXPECTED_FIGURES)} figuras geradas em {FIGURES}")
    print(f"métricas salvas em {RESULTS / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
