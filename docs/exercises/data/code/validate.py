"""Validação técnica da entrega *Data*.

Confere programaticamente cada item que o enunciado exige — contagens, formas,
ausência de vazamento, ausência de `NaN` e de infinitos, existência das figuras e
integridade do relatório — e devolve código de saída diferente de zero se algo
falhar.

Roda depois de ``run_all.py``, porque lê o ``results/metrics.json`` que ele grava::

    python docs/exercises/data/code/run_all.py
    python docs/exercises/data/code/validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import exercise1_point_clouds as exercise1  # noqa: E402
import exercise2_high_dim as exercise2  # noqa: E402
import exercise3_spaceship_titanic as exercise3  # noqa: E402
from common import BASE, FIGURES, RESULTS  # noqa: E402

FIGURE_NAMES = [
    "figure_01_point_clouds.png",
    "figure_02_spread_scales.png",
    "figure_03_mixing_rate.png",
    "figure_04_pca.png",
    "figure_05_radius_histograms.png",
    "figure_06_foodcourt_before_after.png",
]

SUMMARY_ROWS = 13


class Checklist:
    """Acumula o resultado das verificações e imprime um relatório legível."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.results.append((name, bool(condition), detail))
        status = "PASS " if condition else "FALHA"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  {status} {name}{suffix}")

    @property
    def passed(self) -> bool:
        return all(result for _, result, _ in self.results)


def main() -> int:
    metrics_path = RESULTS / "metrics.json"
    if not metrics_path.exists():
        print(f"[FALHA] {metrics_path} não existe — rode run_all.py primeiro.", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    checklist = Checklist()
    # A mesma semente e a mesma ordem de run_all.py, para reproduzir as amostras.
    rng = np.random.default_rng(metrics["seed"])

    print("Exercise 1")
    datasets = {scale: exercise1.generate(rng, scale) for scale in exercise1.SCALES}
    checklist.check(
        "quatro datasets de 400 observações",
        len(datasets) == 4 and all(X.shape == (400, 2) for X, _ in datasets.values()),
    )
    checklist.check(
        "100 observações em cada classe",
        all(np.bincount(y).tolist() == [100] * 4 for _, y in datasets.values()),
    )
    checklist.check(
        "seis pares distintos na tabela de r_ij",
        len(metrics["exercise_1"]["separation_ratios_s1"]) == 6,
    )
    checklist.check("quatro mixing rates", len(metrics["exercise_1"]["mixing_rates"]) == 4)

    print("\nExercise 2")
    X_i, y_i = exercise2.sample_dataset_i(rng)
    X_ii, y_ii = exercise2.sample_dataset_ii(rng)
    checklist.check(
        "500 observações por classe nos dois datasets",
        np.bincount(y_i).tolist() == [500, 500] and np.bincount(y_ii).tolist() == [500, 500],
    )
    checklist.check("dimensionalidade igual a 5", X_i.shape[1] == 5 and X_ii.shape[1] == 5)
    directions = exercise2.random_unit_directions(rng, 2000)
    deviation = float(np.abs(np.linalg.norm(directions, axis=1) - 1.0).max())
    checklist.check(
        "direções unitárias com norma 1", deviation < 1e-12, f"desvio máximo {deviation:.1e}"
    )
    checklist.check(
        "PCA com duas componentes nos dois datasets",
        all(
            {"pc1", "pc2", "total"} <= set(metrics["exercise_2"][key]["pca"])
            for key in ("dataset_i", "dataset_ii")
        ),
    )

    print("\nExercise 3")
    df = exercise3.load_dataset()
    parts = exercise3.split(df)
    ratio = parts["n_test"] / len(df)
    checklist.check("split 80/20", abs(ratio - 0.20) < 0.001, f"teste = {ratio:.4f} do total")
    gap = abs(parts["share_train"] - parts["share_test"]) * 100
    checklist.check("split estratificado", gap < 0.1, f"diferença de {gap:.4f} ponto percentual")

    prepared = exercise3.preprocess(parts["X_train"], parts["X_test"])
    # Ausência de vazamento: as estatísticas de imputação têm de coincidir com as
    # do treino e não com as do dataset completo.
    train_medians = parts["X_train"][exercise3.NUMERIC_COLUMNS].median()
    checklist.check(
        "estatísticas de imputação vêm apenas do treino",
        all(
            abs(prepared["numeric_medians"][column] - float(train_medians[column])) < 1e-9
            for column in exercise3.NUMERIC_COLUMNS
        ),
    )
    checks = metrics["exercise_3"]["checks"]
    checklist.check("nenhum NaN nas matrizes finais", checks["nan_train"] == 0 and checks["nan_test"] == 0)
    checklist.check("nenhum infinito", checks["inf_train"] == 0 and checks["inf_test"] == 0)
    checklist.check(
        "matrizes finais totalmente numéricas",
        checks["all_numeric_train"] and checks["all_numeric_test"],
    )
    checklist.check(
        "shapes corretos e consistentes",
        checks["shape_train"][1] == checks["shape_test"][1]
        and checks["shape_train"][0] == parts["n_train"]
        and checks["shape_test"][0] == parts["n_test"],
        f"treino {tuple(checks['shape_train'])}, teste {tuple(checks['shape_test'])}",
    )

    print("\nFiguras e relatório")
    checklist.check(
        "seis figuras existem e não estão vazias",
        all((FIGURES / name).exists() and (FIGURES / name).stat().st_size > 1000 for name in FIGURE_NAMES),
    )
    report = (BASE / "index.md").read_text(encoding="utf-8")
    checklist.check(
        "todas as figuras referenciadas no relatório",
        all(f"figures/{name}" in report for name in FIGURE_NAMES),
    )
    checklist.check("front matter com exercise e ai_use", report.startswith("---\nexercise: data\nai_use:"))
    checklist.check(
        "Results summary é a última seção",
        report.rstrip().split("\n## ")[-1].startswith("Results summary"),
    )

    summary = report.split("## Results summary")[1]
    rows = [
        line
        for line in summary.splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ") and "Your value" not in line
    ]
    checklist.check(f"Results summary com {SUMMARY_ROWS} linhas", len(rows) == SUMMARY_ROWS, f"{len(rows)} linhas")
    empty = [row for row in rows if any(cell.strip() == "" for cell in row.strip().strip("|").split("|"))]
    checklist.check("nenhuma célula vazia no Results summary", not empty)

    total = len(checklist.results)
    passed = sum(1 for _, result, _ in checklist.results if result)
    print(f"\n  {passed}/{total} verificações passaram")
    return 0 if checklist.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
