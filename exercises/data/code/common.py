"""Infraestrutura compartilhada pelos três exercícios da entrega *Data*.

Centraliza o que não pode divergir entre exercícios: os caminhos do projeto, a
configuração do Matplotlib para rodar sem interface gráfica e o salvamento
determinístico das figuras.

A semente aleatória **não** mora aqui: ela é criada uma única vez em
``run_all.py`` e passada explicitamente para cada função que gera dados, para
que exista um único fluxo reproduzível (ver ``run_all.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Backend "Agg": sem janela e sem display. Sem isto o script falha no GitHub
# Actions, que roda em um runner sem interface gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (precisa vir depois de matplotlib.use)
import numpy as np  # noqa: E402

# Caminhos derivados do próprio arquivo: o script funciona a partir de qualquer
# diretório de trabalho, inclusive da raiz do repositório.
BASE = Path(__file__).resolve().parents[1]
FIGURES = BASE / "figures"
DATA = BASE / "data"
RESULTS = BASE / "results"

DPI = 150

#: Cor fixa por classe, para que a mesma classe tenha a mesma cor em todas as figuras.
CLASS_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#2ca02c",
    3: "#9467bd",
}
BINARY_COLORS = ("#1f77b4", "#d62728")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Salva ``fig`` em ``figures/`` com nome determinístico e fecha a figura.

    Fechar é obrigatório: o script gera seis figuras em sequência e o Matplotlib
    acumula todas em memória se elas não forem descartadas.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def nearest_center_labels(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Índice do centro mais próximo (distância euclidiana) de cada ponto.

    É uma regra puramente geométrica — nenhum modelo é treinado. Usada tanto
    para a *mixing rate* quanto para desenhar o esboço de fronteiras da Figura 1,
    de modo que as duas coisas descrevam exatamente a mesma partição do plano.
    """
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    return distances.argmin(axis=1)


def _jsonable(value: Any) -> Any:
    """Converte tipos NumPy para tipos nativos, para serializar em JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def save_metrics(metrics: dict[str, Any]) -> Path:
    """Grava todas as métricas em ``results/metrics.json``.

    O relatório cita números; este arquivo é a evidência de que cada número veio
    de uma execução real e permite conferi-los sem reexecutar tudo.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "metrics.json"
    path.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def section(title: str) -> None:
    """Cabeçalho de seção na saída do terminal."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
