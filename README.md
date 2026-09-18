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
python docs/exercises/data/code/validate.py
mkdocs build --strict
```

`run_all.py` regenera as seis figuras em `docs/exercises/data/figures/`, imprime
todas as métricas citadas no relatório e as grava em
`docs/exercises/data/results/metrics.json`. `validate.py` confere as 21
verificações técnicas exigidas pelo enunciado.

### Dataset do Exercise 3

O Exercise 3 usa o `train.csv` rotulado da competição
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).
O arquivo já está versionado em `docs/exercises/data/data/train.csv` (8693 linhas,
14 colunas), para que a análise rode a partir de um clone limpo.

Se precisar repor o arquivo, baixe-o pela página da competição e salve-o
exatamente nesse caminho — `load_dataset` confere o esquema e falha com uma
mensagem explícita se o arquivo não estiver lá.

## Preview local do site

```bash
mkdocs serve -o
```
