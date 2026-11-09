# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Giovanny José Vieira Russo | giovannyjvrusso@gmail.com | [giovannyjvr](https://github.com/giovannyjvr) |
| *a preencher* | | |
| *a preencher* | | |

!!! warning "Equipe incompleta"

    Só há um integrante confirmado — o dono do repositório, retirado de
    `mkdocs.yml` e de `docs/index.md`. Os demais nomes **não** foram inventados.
    Preencha antes de submeter: times são de 2 a 3 pessoas, e o cabeçalho de cada
    entrega repete esses nomes.

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação **ou** Regressão | [Classificação](classification/index.md) · [Regressão](regression/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! warning "Escolha uma: classificação ou regressão"

    A segunda entrega é **uma das duas**, não as duas. Este template traz as duas pastas
    para você escolher; depois de decidir, apague a que não vai usar — da pasta `docs/projects/`
    **e** da `nav` no `mkdocs.yml`.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | New York City Airbnb Open Data (`AB_NYC_2019.csv`) |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> |
| **Licença / termos de uso** | **CC0: Public Domain** (campo `licenseName` da API do Kaggle) |
| **Amostras** | 48 895 anúncios (48 884 após remover `price = 0`) |
| **Features** | 15 colunas além do alvo — 9 numéricas e 6 de texto/categóricas |
| **Variável alvo** | `price` — diária anunciada, em dólares (contínua) |
| **Tarefa escolhida** | **Regressão** |

Cada linha é um anúncio ativo em NYC em 2019, com localização, tipo de quarto,
regras de estadia e histórico de avaliações. O alvo é contínuo e vem no próprio
arquivo, o que dispensa junção com outra fonte. A tarefa não é trivial: `price`
tem assimetria de 19,1 e varia de 10 a 10 000 dólares, a localização é o sinal
mais forte mas chega em 221 bairros de cardinalidade desbalanceada, e 20,6 % dos
anúncios nunca foram avaliados — uma ausência estrutural, não aleatória. São três
problemas que a rede vai sentir e que o EDA precisa resolver antes.

!!! danger "Status da proposta: não confirmado"

    Não há no repositório nenhum registro de que esta proposta de dataset tenha
    sido **aprovada** pelo professor. A análise técnica foi feita mesmo assim,
    como manda o enunciado, mas a aprovação precisa ser confirmada antes da
    entrega — e este aviso só sai daqui com evidência.

## Status

- [x] **1. EDA**
- [ ] **2. Classificação ou Regressão**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-11-05 | Dataset: NYC Airbnb Open Data (`AB_NYC_2019.csv`) | Público (CC0), tabular, 48 895 linhas, alvo contínuo no próprio arquivo, e não é dataset clássico de aula. |
| 2026-11-05 | Tarefa: **regressão** sobre `price` | O alvo é contínuo; transformá-lo em classes jogaria fora a informação que torna o problema interessante. |
| 2026-11-05 | Remover as 11 linhas com `price = 0` | Diária de zero dólar não é observação da grandeza que se quer prever. Efeito documentado na seção 1C do EDA. |
| 2026-11-05 | Descartar `id`, `host_id`, `name`, `host_name` | Identificadores e texto livre; a justificativa de cada um está na seção 1B do EDA. |
| 2026-11-05 | Pendente: apagar a pasta de **classificação** | A segunda entrega é uma só. Com a tarefa definida como regressão, `docs/projects/classification/` e a linha correspondente na `nav` devem sair — não foram apagadas aqui para não remover trabalho sem confirmação. |
