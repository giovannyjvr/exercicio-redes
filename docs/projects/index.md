# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em partes ao
longo do semestre, cada uma com data e peso próprios.

!!! warning "Este repositório entrega duas das três partes"

    A disciplina prevê três entregas: EDA, classificação **ou** regressão, e generativo.
    Aqui estão as **duas primeiras**. A pasta de classificação saiu porque a equipe escolheu
    regressão — é o que o enunciado manda fazer. A de **generativo foi removida em
    18/11/2026 por decisão da equipe**, que não fará a terceira entrega; ela não estava
    começada e continua recuperável no commit `55c41c5`.

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

## As entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | **Regressão** (escolhida) | [Regressão](regression/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! success "Escolha feita: regressão"

    A segunda entrega é **uma das duas**. Como o alvo `price` é contínuo, a equipe escolheu
    **regressão**; a pasta `docs/projects/classification/` e a linha correspondente da `nav`
    foram removidas em 18/11/2026.

## Dataset

O mesmo dataset atravessa as duas entregas — escolhê-lo bem no EDA é o que torna a
regressão viável.

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
- [x] **2. Regressão**

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
| 2026-11-18 | Pasta de classificação **removida** | Confirmado: ela só continha o template, sem nenhum trabalho (conferido no histórico do Git). A `nav` do `mkdocs.yml` ficou com uma única linha para a entrega 2. |
| 2026-11-18 | Entrega 2: treinar sobre `log1p(price)` | O EDA mediu que 2,14 % dos anúncios respondem por 86,1 % da soma dos desvios quadráticos. As métricas voltam para dólares com `expm1`. |
| 2026-11-18 | Entrega 3 (**generativo**) não será feita; pasta removida | Decisão da equipe. A pasta só tinha o template, sem trabalho; sai também da `nav` e dos checklists. Recuperável no commit `55c41c5` se a decisão mudar. |
| 2026-11-18 | Validação separada de dentro do treino (80/20 do treino) | O split externo continua o do EDA, com a mesma semente. A pipeline passa a ser ajustada só no sub-treino, o que reduz as colunas de 106 para 91 — menos bairros alcançam `min_frequency = 50`. |
