# exercicio-redes

Portfólio de entregas da disciplina **Redes Neurais Artificiais & Deep Learning**
(Insper, edição 2026.2) — Giovanny José Vieira Russo.

Site publicado: <https://giovannyjvr.github.io/exercicio-redes/>

## Entregas

| Entrega | Relatório | Código |
|---|---|---|
| 1. Data | `docs/exercises/data/index.md` | `docs/exercises/data/code/` |

## Como reproduzir

```bash
python -m pip install -r requirements.txt
python docs/exercises/data/code/run_all.py
mkdocs build --strict
```

`run_all.py` regenera as seis figuras em `docs/exercises/data/figures/` e imprime
todas as métricas citadas no relatório.

### Dataset do Exercise 3

O Exercise 3 usa o `train.csv` rotulado da competição
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).
O arquivo precisa estar em:

```
docs/exercises/data/data/train.csv
```

## Preview local do site

```bash
mkdocs serve -o
```
