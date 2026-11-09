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

## Projeto — entrega 1 (EDA)

Análise exploratória do **New York City Airbnb Open Data** para uma tarefa de
**regressão** sobre `price`. Relatório em `docs/projects/eda/index.md`; código em
`docs/projects/eda/code/`.

O CSV **não** é versionado: tem 6,8 MB e o KaggleHub o reconstrói em qualquer
clone limpo. Datasets públicos do Kaggle não exigem credencial.

```bash
python -m pip install -r requirements.txt -r requirements-eda.txt
python docs/projects/eda/code/run_eda.py
```

`run_eda.py` baixa a base, roda as etapas 0–5, grava as 15 figuras em
`figures/` e todos os números do relatório em `results/metrics.json`. Leva cerca
de 10 minutos — o t-SNE é a parte lenta.

Se preferir apontar para um CSV já baixado, defina a variável de ambiente
`NYC_AIRBNB_CSV` com o caminho do arquivo; sem ela, o KaggleHub resolve sozinho:

```bash
NYC_AIRBNB_CSV=/caminho/para/AB_NYC_2019.csv python docs/projects/eda/code/run_eda.py
```

A pipeline de pré-processamento é um módulo importável, pensada para a entrega
de regressão reaproveitá-la sem copiar código:

```python
from nyc_preprocessing import load_dataset, drop_invalid_target, split_data, build_preprocessor

df, _ = drop_invalid_target(load_dataset())
X_train, X_test, y_train, y_test = split_data(df)     # price fora de X
prep = build_preprocessor(X_train)
Z_train = prep.fit_transform(X_train)                 # aprende só no treino
Z_test = prep.transform(X_test)
```

## Preview local do site

```bash
mkdocs serve -o
```
