# exercicio-redes

Portfólio de entregas da disciplina **Redes Neurais Artificiais & Deep Learning**
(Insper, edição 2026.2) — Giovanny José Vieira Russo.

Site publicado: <https://giovannyjvr.github.io/exercicio-redes/>

## Entregas

| # | Entrega | Relatório | Código | Figuras |
|---|---|---|---|---|
| 1 | Data | `docs/exercises/data/index.md` | `docs/exercises/data/code/` | 6 |
| 2 | Perceptron | `docs/exercises/perceptron/index.md` | `docs/exercises/perceptron/code/` | 6 |
| 3 | MLP | `docs/exercises/mlp/index.md` | `docs/exercises/mlp/code/` | 6 |
| 4 | VAE | `docs/exercises/vae/index.md` | `docs/exercises/vae/code/` | 6 |

## Como reproduzir

```bash
python -m pip install -r requirements.txt

python docs/exercises/data/code/run_all.py
python docs/exercises/data/code/validate.py
python docs/exercises/perceptron/code/run_all.py
python docs/exercises/mlp/code/run_all.py

mkdocs build --strict
```

Cada `run_all.py` regenera as seis figuras da sua entrega em `figures/`, imprime
todos os números citados no relatório e os grava em `results/metrics.json`.

### Exercício 4 (VAE) — dependências extras

O VAE precisa de PyTorch, que fica num arquivo separado porque o GitHub Actions
apenas constrói o site e não precisa baixá-lo:

```bash
python -m pip install -r requirements-vae.txt
python docs/exercises/vae/code/run_all.py
```

Em máquina sem GPU, as rodas de CPU bastam e são bem menores:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

O MNIST é baixado automaticamente pelo `torchvision` em
`docs/exercises/vae/data/` (ignorado pelo Git).

### Dataset do exercício 1

O exercício Data usa o `train.csv` rotulado da competição
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data),
versionado em `docs/exercises/data/data/train.csv` (8693 linhas, 14 colunas), para
que a análise rode a partir de um clone limpo.

## Preview local do site

```bash
mkdocs serve -o
```
