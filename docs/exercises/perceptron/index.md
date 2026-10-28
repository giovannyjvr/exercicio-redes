---
exercise: perceptron
ai_use: "Claude Code (Anthropic) foi utilizado para interpretar os requisitos, gerar e revisar o código Python, produzir visualizações, calcular métricas e apoiar a redação do relatório. O aluno revisou o resultado e deve compreender e conseguir explicar todas as decisões e linhas de código."
---

# 2. Perceptron

!!! abstract "Enunciado"

    [Exercises → Perceptron](https://insper.github.io/ann-dl/){:target='_blank'}

!!! info "Como reproduzir"

    Todos os números deste relatório saem de uma única execução. A partir da raiz
    do repositório:

    ```bash
    python -m pip install -r requirements.txt
    python docs/exercises/perceptron/code/run_all.py
    mkdocs build --strict
    ```

    `run_all.py` regenera as seis figuras em `figures/`, imprime todos os números
    no terminal e os grava em `results/metrics.json`.

**Estrutura do código.** O perceptron vive em
[`code/perceptron.py`](https://github.com/giovannyjvr/exercicio-redes/blob/main/docs/exercises/perceptron/code/perceptron.py)
e é incluído neste relatório pelo próprio arquivo, via `--8<--`.

| Arquivo | Papel |
|---|---|
| `perceptron.py` | A implementação do zero: `step`, `predict`, regra de atualização e laço de treino. |
| `run_all.py` | Ponto de entrada único: gera os dados, treina, produz as seis figuras e salva as métricas. |

!!! warning "Nenhuma biblioteca fornece o modelo"

    A ativação, a predição, a regra de atualização e o laço de treino são escritos
    à mão. O NumPy entra apenas como álgebra linear (produto escalar, permutação,
    amostragem das gaussianas). **Não há `sklearn` neste exercício** — nem
    `Perceptron`, nem `SGDClassifier`, nem qualquer `fit`.

    A semente é criada uma única vez, em `run_all.py`, como
    `rng = np.random.default_rng(42)`, e o mesmo objeto é passado à geração dos
    dados e a cada treino, sempre na mesma ordem.

??? note "`perceptron.py` — a implementação"

    ``` { .python .copy .select linenums='1' title="docs/exercises/perceptron/code/perceptron.py" }
    --8<-- "docs/exercises/perceptron/code/perceptron.py"
    ```

??? note "`run_all.py` — geração dos dados, treino e figuras"

    ``` { .python .copy .select linenums='1' title="docs/exercises/perceptron/code/run_all.py" }
    --8<-- "docs/exercises/perceptron/code/run_all.py"
    ```

## Exercise 1

### A — Generate the data

**Abordagem.** Duas classes de pontos 2D, 1000 amostras cada, de normais
multivariadas com os parâmetros do enunciado:

| Classe | Média | Covariância |
|---|---|---|
| 0 | $[1.5,\ 1.5]$ | $[[0.5,\ 0],\ [0,\ 0.5]]$ |
| 1 | $[5,\ 5]$ | $[[0.5,\ 0],\ [0,\ 0.5]]$ |

A distância entre as médias é $\lVert [5,5] - [1.5,1.5] \rVert = 4.9497$, contra
um desvio-padrão de $\sqrt{0.5} = 0.7071$ por eixo — cerca de **7 desvios**. As
nuvens ficam bem separadas.

![Dispersão das duas classes separáveis](figures/figure_01_separable_scatter.png)
/// caption
**Figura 1** — Os 2000 pontos, uma cor por classe. As nuvens não se tocam.
///

*Código: `run_all.py` → `generate()` e `figure_scatter()`.*

### B — Implement the perceptron

**Abordagem.** Implementação do zero, em `perceptron.py`, reutilizada sem
alteração no Exercise 2.

- **Predição.** $\hat{y} = \text{step}(\mathbf{w}\cdot\mathbf{x} + b)$, com
  $\text{step}(z) = 1$ se $z \geq 0$ e $0$ caso contrário.
- **Atualização.** Para cada amostra $(\mathbf{x}, y)$:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat{y})\,\mathbf{x},
\qquad
b \leftarrow b + \eta\,(y - \hat{y})
$$

- **Inicialização.** $\mathbf{w} \sim$ `rng.normal(0, 0.01, size=2)` e $b = 0$.
- **Taxa de aprendizado.** $\eta = 0.01$.
- **Parada.** Quando uma passagem completa não produz nenhuma atualização, ou ao
  atingir 100 épocas. A acurácia sobre o conjunto completo é registrada ao fim de
  cada época.

!!! note "Por que não $\mathbf{w} \leftarrow \mathbf{w} + \eta\,y\,\mathbf{x}$"

    Essa forma pertence à convenção $y \in \{-1, +1\}$. Com rótulos $0/1$ ela
    nunca atualizaria na classe 0 — o termo $y\mathbf{x}$ seria sempre nulo — e o
    perceptron jamais corrigiria um falso positivo. O fator $(y - \hat{y})$ vale
    $0$ quando a predição está certa (amostras corretas não geram atualização) e
    $\pm 1$ nos dois tipos de engano, que é exatamente o comportamento desejado.

**Uma escolha de implementação a declarar:** os dados são gerados agrupados por
classe (1000 da classe 0, depois 1000 da classe 1). Percorrê-los nessa ordem faria
o perceptron ver uma classe inteira antes da outra, o que distorce o treino. O laço
**embaralha a cada época** com `rng.permutation`, usando o mesmo gerador — continua
inteiramente reprodutível.

### C — Train and measure

**Resultados com $\eta = 0.01$:**

| Quantidade | Valor |
|---|---|
| $\mathbf{w}$ inicial | $[0.017029,\ 0.006467]$ |
| $\mathbf{w}$ final | $[0.004378,\ 0.004702]$ |
| $b$ final | $-0.030000$ |
| Épocas até convergir | **2** |
| Acurácia final | **100.00 %** (2000 de 2000) |
| Atualizações por época | época 1: **5**; época 2: **0** |

O treino parou por convergência, não por limite de épocas: a segunda passagem
percorreu os 2000 pontos sem precisar de uma única correção. Bastaram **5**
correções no total — as nuvens estão a sete desvios uma da outra, então quase
nenhum ponto cai do lado errado da reta inicial.

![Fronteira aprendida sobre os dados separáveis](figures/figure_02_separable_boundary.png)
/// caption
**Figura 2** — A fronteira $\mathbf{w}\cdot\mathbf{x} + b = 0$ sobre os dados. Não
há nenhum ponto mal classificado para destacar — o contador na legenda marca 0.
///

![Acurácia por época](figures/figure_03_separable_accuracy.png)
/// caption
**Figura 3** — Acurácia por época. Já ao fim da primeira época, depois das 5
correções, a acurácia é 100 %; a segunda época apenas confirma que nada mais muda.
///

*Código: `perceptron.py` → `train()`; `run_all.py` → `figure_boundary()` e
`figure_accuracy()`.*

### D — Analysis

#### Por que dados separáveis convergem rápido

A resposta está na própria regra de atualização: **o fator $(y - \hat{y})$ é zero
para toda amostra classificada corretamente**. Uma amostra certa não move
$\mathbf{w}$ nem $b$ — ela é efetivamente ignorada. Só os erros empurram a
fronteira, e cada empurrão a move na direção que corrige aquele erro específico.

A consequência é que **o número de atualizações por época mede quanto trabalho
resta**, e em dados separáveis ele desce a zero rapidamente. Foi o que aconteceu:

| Época | Atualizações ($\eta = 0.01$) | Acurácia ao fim da época |
|---:|---:|---:|
| 1 | 5 | 100.00 % |
| 2 | **0** | 100.00 % |

Como existe um hiperplano que acerta todos os pontos, há um conjunto de pesos em
que nenhuma amostra gera erro — e, atingido esse conjunto, o laço vira um ponto
fixo: zero atualizações, zero movimento, parada. É isso que o teorema de
convergência do perceptron garante para dados linearmente separáveis.

#### O que $\eta$ controla — a comparação $\eta = 0.01$ contra $\eta = 1.0$

Mesmo treino, **mesma inicialização, mesma ordem de embaralhamento**; só $\eta$
muda. Isso não é automático: se os dois treinos recebessem o gerador compartilhado
em sequência, o segundo partiria de um `rng` já avançado pelo primeiro — outra
inicialização e outra ordem de exemplos — e a diferença entre as fronteiras não
poderia ser atribuída a $\eta$. Por isso cada treino recebe um gerador novo,
construído a partir de uma mesma semente sorteada do `rng` compartilhado; o código
confere a igualdade com `assert np.array_equal(run["w0"], run_fast["w0"])`.

| | $\eta = 0.01$ | $\eta = 1.0$ |
|---|---|---|
| $\mathbf{w}$ inicial (idêntico) | $[0.017029,\ 0.006467]$ | $[0.017029,\ 0.006467]$ |
| $\mathbf{w}$ final | $[0.004378,\ 0.004702]$ | $[0.939008,\ 1.986582]$ |
| $b$ final | $-0.030000$ | $-10.000000$ |
| Épocas | **2** | **2** |
| Acurácia final | **100.00 %** | **100.00 %** |
| $\lVert\mathbf{w}\rVert$ | 0.006425 | 2.197327 |
| Direção $\mathbf{w}/\lVert\mathbf{w}\rVert$ | $[0.681437,\ 0.731877]$ | $[0.427341,\ 0.904090]$ |
| Atualizações por época | $[5,\ 0]$ | $[22,\ 0]$ |

As duas corridas chegam a 100 % e param na mesma época, mas **por fronteiras
diferentes**: o ângulo entre as duas direções é de **17.6571°**.

O enunciado dá a chave: cada atualização soma $\eta\mathbf{x}$ a pesos que começaram
com magnitude $\lVert\mathbf{w}_0\rVert = 0.018216$. Portanto **$\eta$ controla o
peso relativo entre a inicialização aleatória e as correções acumuladas**. Nestes
dados $\lVert\mathbf{x}\rVert$ vale **4.6543** em média, então:

- Com $\eta = 0.01$, cada correção tem magnitude
  $\eta\lVert\mathbf{x}\rVert \approx 0.0465$ — da mesma ordem de grandeza da
  inicialização (**2.6 vezes** $\lVert\mathbf{w}_0\rVert$). O $\lVert\mathbf{w}\rVert$
  final é **0.006425**, ou **0.35 vez** o inicial: as 5 correções não apenas giraram
  o vetor, elas **encolheram** o que já estava lá. O vetor inicial aleatório ainda
  pesa na direção final.
- Com $\eta = 1.0$, cada correção tem magnitude $\approx 4.6543$ — **255 vezes**
  $\lVert\mathbf{w}_0\rVert$. O $\lVert\mathbf{w}\rVert$ final é **2.197327**,
  **120.6 vezes** a inicialização. O vetor inicial foi **apagado**; a direção final é
  essencialmente a soma dos $\mathbf{x}$ que geraram erro.

Logo $\eta$ **não** controla "se" o perceptron converge em dados separáveis — ambos
convergem, e aqui até na mesma época — mas **para qual** dos infinitos hiperplanos
separadores ele converge.

Um detalhe que vale reportar porque contraria a intuição de que passo maior é
sempre "mais eficiente": **$\eta = 1.0$ precisou de 22 correções contra 5**, mais
que o quádruplo, para chegar ao mesmo lugar em acurácia. Com passo grande, cada
correção joga a fronteira longe e cria novos erros do outro lado, que precisam ser
corrigidos por sua vez; com passo pequeno, as 5 correções sutis sobre um vetor
inicial que já apontava para um lado razoável bastaram. Taxa maior não significa
menos trabalho.

#### Por que partir de $\mathbf{w} = \mathbf{0}$, $b = 0$ tornaria $\eta$ irrelevante

Afirmação: se o treino inteiro for executado duas vezes a partir de
$\mathbf{w} = \mathbf{0}$, $b = 0$ — uma com $\eta_1$ e outra com $\eta_2$ — os
pesos resultantes diferem apenas pelo fator constante $c = \eta_2/\eta_1$, de modo
que **a fronteira de decisão e a contagem de épocas são idênticas**.

*Demonstração por indução sobre o número de atualizações.* Seja
$(\mathbf{w}^{(t)}_1, b^{(t)}_1)$ o estado da corrida com $\eta_1$ após $t$ passos, e
$(\mathbf{w}^{(t)}_2, b^{(t)}_2)$ o da corrida com $\eta_2$. A hipótese de indução é

$$
\mathbf{w}^{(t)}_2 = c\,\mathbf{w}^{(t)}_1, \qquad b^{(t)}_2 = c\,b^{(t)}_1,
\qquad c = \eta_2/\eta_1 > 0 .
$$

**Base ($t = 0$).** Ambas começam em $\mathbf{0}$ e $0$, e $c \cdot \mathbf{0} =
\mathbf{0}$. Vale.

**Passo.** Suponha a hipótese válida em $t$ e considere a próxima amostra
$\mathbf{x}$ (a ordem é a mesma nas duas corridas, pois depende só do `rng`). A
pré-ativação da segunda corrida é

$$
\mathbf{w}^{(t)}_2 \cdot \mathbf{x} + b^{(t)}_2
= c\left(\mathbf{w}^{(t)}_1 \cdot \mathbf{x} + b^{(t)}_1\right).
$$

Como $c > 0$, **a multiplicação por $c$ não muda o sinal**. Logo $\text{step}$
devolve o mesmo $\hat{y}$ nas duas corridas, e o erro $(y - \hat{y})$ é o mesmo —
chame-o de $e$. As atualizações ficam

$$
\mathbf{w}^{(t+1)}_2 = c\,\mathbf{w}^{(t)}_1 + \eta_2\,e\,\mathbf{x}
= c\left(\mathbf{w}^{(t)}_1 + \tfrac{\eta_2}{c}\,e\,\mathbf{x}\right)
= c\left(\mathbf{w}^{(t)}_1 + \eta_1\,e\,\mathbf{x}\right)
= c\,\mathbf{w}^{(t+1)}_1,
$$

usando $\eta_2/c = \eta_1$. O mesmo cálculo vale para $b$. A hipótese se propaga.
$\blacksquare$

**Três consequências.** (i) As duas corridas cometem exatamente os mesmos erros, na
mesma ordem, logo fazem o **mesmo número de atualizações** e param na **mesma
época**. (ii) A fronteira é o conjunto
$\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\}$, e multiplicar
$(\mathbf{w}, b)$ por $c > 0$ não altera esse conjunto — é a **mesma reta**.
(iii) Portanto $\eta$ não tem efeito algum: é puro fator de escala.

É exatamente por isso que o item B proíbe o início em zero. Com
$\mathbf{w} \sim \mathcal{N}(0, 0.01^2)$ existe uma escala de referência fixa e
não nula, que não é reescalada junto com $\eta$ — e aí $\eta$ passa a importar,
como as duas direções separadas por 17.6571° mostram.

## Exercise 2

### A — Generate the data

**Abordagem.** Mesma estrutura, parâmetros do enunciado:

| Classe | Média | Covariância |
|---|---|---|
| 0 | $[3,\ 3]$ | $[[1.5,\ 0],\ [0,\ 1.5]]$ |
| 1 | $[4,\ 4]$ | $[[1.5,\ 0],\ [0,\ 1.5]]$ |

Agora a distância entre as médias é $\lVert [4,4]-[3,3] \rVert = 1.4142$, contra
desvio $\sqrt{1.5} = 1.2247$ por eixo. As médias ficaram **3.5 vezes mais perto** e
a dispersão **1.7 vez maior** que no Exercise 1: as nuvens se interpenetram e
nenhuma reta as separa.

![Dispersão das duas classes sobrepostas](figures/figure_04_overlapping_scatter.png)
/// caption
**Figura 4** — Os 2000 pontos. As duas classes ocupam praticamente a mesma região.
///

### B — Train, keeping the best weights

**Abordagem.** A função `train()` do Exercise 1 é reutilizada **sem nenhuma
alteração**, com $\eta = 0.01$ e o mesmo teto de 100 épocas. O *pocket* é a única
coisa acrescentada ao laço: sempre que uma atualização deixa a acurácia sobre o
conjunto completo acima da melhor já vista, copia-se $(\mathbf{w}, b)$.

!!! note "O pocket está sempre ligado, nos dois exercícios"

    Para que a função fosse literalmente a mesma, o rastreamento do pocket roda
    também no Exercise 1. Lá ele é inofensivo: como o treino termina em 100 % de
    acurácia, o pocket converge para os próprios pesos finais e não altera nada.

| | $\mathbf{w}$ | $b$ | Acurácia | Erros (de 2000) |
|---|---|---|---:|---:|
| **Pesos finais** | $[0.057028,\ 0.046427]$ | $-0.420000$ | **66.10 %** | 678 |
| **Pesos pocket** | $[0.063451,\ 0.057988]$ | $-0.410000$ | **71.85 %** | 563 |

O melhor pocket ocorreu na **época 27**. O treino percorreu as **100 épocas sem
convergir** — como esperado, já que nenhuma reta acerta todos os pontos, sempre
resta erro e portanto sempre há atualização. Foram **75 608 atualizações** no
total, entre 706 e 818 por época.

O pocket chegou a 71.85 %, perto dos ~73 % que o enunciado antecipa como teto
prático para uma reta aqui. Os pesos finais ficaram quase **6 pontos abaixo**, em
66.10 %, e o item D mostra que esse número é um sorteio: a curva da Figura 6 visita
valores de 50.00 % a 71.40 % e o treino simplesmente para onde a centésima época o
deixou. **Esse é o resultado correto, não um defeito.**

### C — Figures

![Fronteiras final e pocket](figures/figure_05_overlapping_boundaries.png)
/// caption
**Figura 5** — As duas fronteiras sobre os dados. A **pocket** (verde, tracejada)
atravessa o meio da nuvem; a **final** (preta, contínua) está deslocada para cima e
para a direita, deixando a maior parte dos pontos do lado da classe 0 — ela rotula
só 29.20 % do conjunto como classe 1, quando 50 % o são. Os círculos marcam os 678
pontos que a fronteira final erra.
///

![Acurácia atual versus pocket](figures/figure_06_overlapping_accuracy.png)
/// caption
**Figura 6** — Acurácia dos pesos atuais (vermelho) e melhor acurácia já vista
(verde) por época. A curva vermelha oscila entre 50.00 % e 71.40 % durante as 100
épocas e nunca assenta; a verde é monótona por construção.
///

### D — Analysis

#### A distância entre a acurácia final e a do pocket

A melhor reta possível para estes dados acerta cerca de 73 %, e o pocket chegou a
**71.85 %** — praticamente lá. Os pesos finais ficaram em **66.10 %**. A diferença
de 5.75 pontos não vem de o perceptron "não ter aprendido": vem de **onde a regra
de atualização deixa a fronteira**.

**Onde a fronteira final está.** O centróide dos dados é $[3.4733,\ 3.5264]$. A
distância com sinal do centróide até cada fronteira, medida em unidades de
$\lVert\mathbf{w}\rVert$, é:

| | Distância com sinal do centróide | Pontos classificados como classe 1 |
|---|---:|---:|
| Fronteira final | $-0.7915$ | **29.20 %** |
| Fronteira pocket | $+0.1730$ | **54.75 %** |

A fronteira final passa **0.79 unidade do lado da classe 1 em relação ao centro da
nuvem** — ela corta a nuvem acima do meio, como a Figura 5 mostra. O resultado é que
rotula só **29.20 % dos pontos como classe 1**, embora 50 % o sejam: acerta quase
toda a classe 0 e erra a maior parte da classe 1, e daí os 66.10 %. A fronteira
pocket, em contraste, passa a $0.17$ do centróide — praticamente pelo meio — e
divide os rótulos de forma equilibrada (54.75 % positivos).

**Por que o laço a deixa ali.** A dica do enunciado é comparar quanto $b$ se move
por erro contra quanto $\mathbf{w}$ se move. Por atualização:

$$
\Delta\lVert\mathbf{w}\rVert \le \eta\lVert\mathbf{x}\rVert
= 0.01 \times 5.1111 = 0.0511,
\qquad
|\Delta b| = \eta = 0.0100 .
$$

Com $\lVert\mathbf{x}\rVert \approx 5.11$ para estes dados, **$\mathbf{w}$ se move
5.11 vezes mais depressa que $b$ a cada erro**. Mas quem posiciona a fronteira
*ao longo* da direção $\mathbf{w}$ é justamente a razão $-b/\lVert\mathbf{w}\rVert$.
Como $\mathbf{w}$ cresce rápido e $b$ se arrasta, o deslocamento não consegue
acompanhar a rotação: a reta gira com agilidade mas não desliza até o centro da
nuvem. Ela termina onde o último punhado de erros a jogou — e, como em dados
sobrepostos há 700+ erros por época, "o último punhado" é essencialmente aleatório.
A acurácia final é um sorteio entre os valores que a curva vermelha da Figura 6
visita; nesta execução o sorteio caiu em 66.10 %. Em outra execução poderia cair em
50 % ou em 71 % — é precisamente essa imprevisibilidade que o pocket elimina.

#### Figura 3 contra Figura 6, e a hipótese que o teorema exige

Na **Figura 3** a acurácia sobe a 100 % e fica: zero atualização, zero movimento,
ponto fixo. Na **Figura 6** ela oscila entre 50.00 % e 71.40 % por 100 épocas, com
média de 61.56 %, e termina em 66.10 % — sem nenhuma tendência de assentar: o valor
final é apenas o ponto onde a centésima época parou.

O **teorema de convergência do perceptron** garante que, se os dados forem
**linearmente separáveis**, o algoritmo encontra um hiperplano separador em um
número finito de atualizações, limitado por $(R/\gamma)^2$, onde $R$ é o raio dos
dados e $\gamma$ a margem do melhor separador. A hipótese violada aqui é exatamente
essa: **a separabilidade linear**. Com as médias a 1.41 de distância e desvio 1.22,
as nuvens se sobrepõem e **não existe $(\mathbf{w}^\*, b^\*)$ com erro zero** — a
margem $\gamma$ não existe, o limite $(R/\gamma)^2$ é infinito, e o teorema
simplesmente não diz nada. Sem um ponto fixo onde pousar, o laço nunca para de
corrigir.

Vale sublinhar o que o teorema **nunca** prometeu: ele não garante a *melhor* reta,
nem que a acurácia cresça monotonicamente. Mesmo no caso separável ele só promete
*alguma* reta separadora em tempo finito.

#### Mais épocas resolvem? E um $\eta$ menor?

**Nenhum dos dois**, e o argumento vem da regra de atualização, não de tentativa e
erro.

**Mais épocas, não.** A condição de parada é "uma passagem sem nenhuma
atualização". Ela exige que *todos* os 2000 pontos estejam do lado certo —
impossível aqui. Então, para qualquer número de épocas, a última época ainda
conterá centenas de erros (foram entre 706 e 818 por época nas 100 que rodamos), e
cada um deles continuará movendo a fronteira. Rodar 10 000 épocas apenas gera mais
amostras da mesma oscilação: o estado final continua sendo o resíduo arbitrário das
últimas correções. É por isso que o pocket é necessário — ele é a única coisa que
transforma uma trajetória que não converge em uma resposta utilizável, e ele **já
havia encontrado 71.65 % na primeira época**, subindo só até 71.85 % na época 27.

**Um $\eta$ menor, também não.** $\eta$ multiplica igualmente os dois termos da
atualização, $\eta(y-\hat{y})\mathbf{x}$ e $\eta(y-\hat{y})$. Diminuí-lo encolhe
cada passo, mas **não muda quais amostras erram** numa dada configuração, nem muda
a direção de nenhum passo. A trajetória fica com passos menores e amplitude de
oscilação menor — e não é isso que está em jogo. O problema não é o tamanho do
passo; é que **não existe um ponto de parada**. Nenhum valor positivo de $\eta$
cria uma reta separadora que a geometria dos dados não oferece. No limite
$\eta \to 0$ os pesos praticamente não saem da inicialização aleatória, o que piora
a situação em vez de melhorá-la.

A saída real é mudar o critério, não o passo: guardar o melhor (pocket), trocar a
função de perda por uma que penalize a *margem* em vez de apenas contar erros, ou
dar ao modelo uma fronteira não linear.

## Results summary

| # | Quantity | Value |
|---:|---|---|
| 1 | Exercise 1 — final \(\mathbf{w}\) and \(b\) | \(\mathbf{w} = [0.004378,\ 0.004702]\), \(b = -0.030000\) |
| 2 | Exercise 1 — epochs to convergence | 2 épocas (parou por ausência de atualizações, não pelo teto de 100) |
| 3 | Exercise 1 — final accuracy | 100.00 % (2000 de 2000) |
| 4 | Exercise 1 — epochs and final accuracy with \(\eta = 1.0\) | 2 épocas, 100.00 % (mesma inicialização; direção de \(\mathbf{w}\) a 17.6571° da obtida com \(\eta = 0.01\), e 22 correções contra 5) |
| 5 | Exercise 2 — final \(\mathbf{w}\) and \(b\) | \(\mathbf{w} = [0.057028,\ 0.046427]\), \(b = -0.420000\) |
| 6 | Exercise 2 — accuracy of the final weights | 66.10 % (678 erros em 2000) |
| 7 | Exercise 2 — accuracy of the pocket weights | 71.85 % (563 erros em 2000) |
| 8 | Exercise 2 — epoch at which the pocket best occurred | época 27 (de 100; o treino não convergiu) |
