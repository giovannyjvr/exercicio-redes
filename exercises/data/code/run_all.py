"""Ponto de entrada único da entrega *Data*.

Executa os três exercícios em ordem fixa, regenera as seis figuras em
``figures/`` e imprime — além de salvar em ``results/metrics.json`` — todas as
métricas citadas no relatório.

Uso, a partir da raiz do repositório::

    python docs/exercises/data/code/run_all.py

**A semente é criada aqui, uma única vez**, e o mesmo objeto ``rng`` é passado
para cada função que gera dados sintéticos. Ela nunca é reinicializada entre
exercícios ou escalas: é justamente por existir um único fluxo, sempre na mesma
ordem, que os números do relatório são reprodutíveis. Rodar um exercício
isolado consumiria outra parte da sequência aleatória e daria outros números —
por isso este arquivo é o único com bloco ``__main__``.

O Exercise 3 não usa ``rng``: sua única fonte de aleatoriedade é o
``train_test_split``, que não aceita um ``Generator`` do NumPy e é fixado com
``random_state=42`` (mesma semente, na forma que a API aceita).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite tanto ``python docs/exercises/data/code/run_all.py`` (a partir de
# qualquer diretório) quanto ``import run_all`` de dentro da pasta code/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import exercise1_point_clouds as exercise1  # noqa: E402
import exercise2_high_dim as exercise2  # noqa: E402
import exercise3_spaceship_titanic as exercise3  # noqa: E402
from common import FIGURES, save_metrics, section  # noqa: E402

SEED = 42

EXPECTED_FIGURES = [
    "figure_01_point_clouds.png",
    "figure_02_spread_scales.png",
    "figure_03_mixing_rate.png",
    "figure_04_pca.png",
    "figure_05_radius_histograms.png",
    "figure_06_foodcourt_before_after.png",
]


def verify_figures() -> list[str]:
    """Confere que as seis figuras existem no disco e não estão vazias."""
    problems = []
    for name in EXPECTED_FIGURES:
        path = FIGURES / name
        if not path.exists():
            problems.append(f"{name}: não foi gerada")
        elif path.stat().st_size == 0:
            problems.append(f"{name}: arquivo vazio")
    return problems


def main() -> int:
    # Semente única de toda a entrega.
    rng = np.random.default_rng(SEED)

    section("Exercise 1 — Point Clouds: Geometry and Spread in 2D")
    metrics_1 = exercise1.run(rng)

    section("Exercise 2 — Non-Linearity in Higher Dimensions")
    metrics_2 = exercise2.run(rng)

    section("Exercise 3 — Preparing Real-World Data for a Neural Network")
    try:
        metrics_3 = exercise3.run()
    except exercise3.DatasetNotFound as exc:
        print(f"\n[BLOQUEADO] {exc}", file=sys.stderr)
        return 1

    section("Verificação final")
    problems = verify_figures()
    if problems:
        for problem in problems:
            print(f"  [FALHA] {problem}", file=sys.stderr)
        return 1
    print(f"  {len(EXPECTED_FIGURES)} figuras geradas em {FIGURES}")

    metrics_path = save_metrics(
        {
            "seed": SEED,
            "exercise_1": metrics_1,
            "exercise_2": metrics_2,
            "exercise_3": metrics_3,
        }
    )
    print(f"  métricas salvas em {metrics_path}")
    print("\n  OK — todos os exercícios executaram e todas as figuras existem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
