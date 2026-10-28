# Prompts para Claude — ANN-DL Insper 2026.2

Use o **Prompt mestre** primeiro, dentro do repositório que será entregue. Ele orienta Claude a auditar o projeto existente, verificar integralmente o exercício 1, completar os exercícios 2, 3 e 4 em sequência e preparar o GitHub Pages. Os quatro prompts individuais podem ser usados separadamente quando quiser trabalhar ou revisar uma entrega por vez.

> As páginas do curso estão em inglês e são a fonte de verdade. Se houver diferença entre este resumo e as páginas, siga a página oficial. Não invente resultados, arquivos, testes, gráficos ou etapas concluídas. Registre no relatório o uso de IA de forma verdadeira e revise as explicações antes da entrega.

---

# Prompt mestre — auditar e concluir o repositório completo

```text
Atue como assistente de programação e revisão acadêmica para concluir, neste repositório aberto, os quatro exercícios individuais de Artificial Neural Networks and Deep Learning (ANN-DL), Insper, semestre 2026.2.

OBJETIVO
Entregar um único repositório público, organizado como portfólio, com quatro páginas independentes (abas/itens de navegação) para os quatro exercícios: Data, Perceptron, MLP e VAE. Audite primeiro o repositório atual, preserve o que estiver correto e complete/repare o necessário. Verifique se o exercício Data (exercício 1) está completo antes de seguir; depois faça Perceptron, MLP e VAE. Configure a publicação por GitHub Pages conforme o formato oficial do curso.

FONTES OFICIAIS — LEIA CADA PÁGINA INTEIRA, inclusive critérios de avaliação e resumo de resultados:
1. Data: https://insper.github.io/ann-dl/2026.2/exercises/data/
2. Perceptron: https://insper.github.io/ann-dl/2026.2/exercises/perceptron/
3. MLP: https://insper.github.io/ann-dl/2026.2/exercises/mlp/
4. VAE: https://insper.github.io/ann-dl/2026.2/exercises/vae/
Formato de submissão: https://insper.github.io/ann-dl/2026.2/exercises/submission/

REGRAS DE EXECUÇÃO
1. Comece sem alterar arquivos: liste a estrutura e o estado do Git, leia README, configurações, notebooks, relatórios, código, figuras e workflows existentes. Identifique o framework/site já usado. Não apague trabalho do aluno nem recomece se uma parte estiver correta.
2. Faça uma auditoria do exercício Data item por item, comparando implementação, evidências numéricas, análise, figuras e tabela final com a página e a rubrica. Crie uma checklist com status completo/parcial/ausente e evidência/localização. Em seguida corrija os itens faltantes e valide-os.
3. Trabalhe na sequência Data → Perceptron → MLP → VAE. Conclua e verifique uma página antes de avançar. Mantenha cada exercício isolado em seu caminho obrigatório e seus próprios `code/` e `figures/` quando houver código/figuras.
4. Siga integralmente as páginas oficiais: parâmetros, quantidades, seeds, arquiteturas, bibliotecas permitidas, itens A/B/C/D/E, perguntas, números esperados apenas como referência, figuras numeradas e tabela “Results summary”. Os valores finais devem vir da execução real. Não force resultados para coincidir com valores de referência.
5. Reprodutibilidade: respeite exatamente os geradores/seeds pedidos; mantenha seeds independentes quando cada modelo ou repetição exigir reinicialização. Não deixe o estado aleatório de um exercício contaminar outro. Registre dependências e como executar a partir de um checkout limpo.
6. Restrições importantes:
   - Data: use `numpy`, `pandas`, `matplotlib`/`seaborn` e scikit-learn apenas para PCA e preprocessing; não treine modelo.
   - Perceptron: implemente step, predição, regra de atualização e loop do zero; não use `sklearn` nem outra biblioteca para treinar um perceptron/classificador.
   - MLP: implemente forward, perdas, gradientes/backprop e atualização em NumPy, sem TensorFlow, PyTorch, `MLPClassifier` ou autodiff; scikit-learn apenas para geradores/split e matriz de confusão, além da regressão logística baseline permitida no exercício 2.
   - VAE: PyTorch/autograd são permitidos; implemente manualmente reparameterization, BCE com soma dos pixels, KL fechado e amostragem. Não use classe VAE de biblioteca nem `torch.distributions.kl_divergence`.
7. Não use o conjunto de teste para decisões de treinamento, seleção de hiperparâmetros ou comparação durante treino. Siga o protocolo específico de cada página. No VAE, decisões devem usar validação; teste só para números finais.
8. Toda figura deve estar salva no repositório, numerada, legível, titulada e com rótulos de eixos; legendas de classe nas figuras que comparam classes. Inclua as figuras no Markdown usando caminhos relativos corretos.
9. Cada relatório deve refletir os títulos/subtítulos e a ordem da página oficial. A última seção deve ser a tabela Results summary oficial, na mesma ordem, com a coluna `#` e todas as linhas preenchidas.
10. Siga exatamente o layout oficial MkDocs + Material e mantenha o site como um único repositório. Layout esperado (adapte sem duplicar se parte já existir):
    docs/index.md
    docs/exercises/data/index.md
    docs/exercises/data/code/...
    docs/exercises/data/figures/...
    docs/exercises/perceptron/index.md
    docs/exercises/perceptron/code/...
    docs/exercises/perceptron/figures/...
    docs/exercises/mlp/index.md
    docs/exercises/mlp/code/...
    docs/exercises/mlp/figures/...
    docs/exercises/vae/index.md
    docs/exercises/vae/code/...
    docs/exercises/vae/figures/...
    mkdocs.yml
    requirements.txt
    O curso fixa os slugs `data`, `perceptron`, `mlp`, `vae`. Inclua quatro entradas distintas de navegação, uma para cada página.
11. Cada relatório começa com front matter obrigatório, por exemplo:
    ---
    exercise: data
    ai_use: "Claude ajudou a revisar/implementar [descrever fielmente o uso real]; análise revisada pelo aluno."
    ---
    Não invente descrição da contribuição. `ai_use` não pode faltar.
12. Código deve viver em arquivos sob `code/` e ser incluído no relatório a partir deles com o mecanismo compatível com o site/template (por exemplo, snippet MkDocs `--8<--`). Não deixe a única cópia executável dentro do Markdown. Figuras devem ser arquivos commitáveis e efetivamente exibidos.
13. Analise template/configuração/workflow existente. Se este repositório não for ainda baseado no template do curso, compare com o formato obrigatório e adapte sem remover conteúdo útil. Configure GitHub Actions/MkDocs Pages para compilar e publicar em push, testar links/caminhos e corrigir build. Não exponha segredos. Não altere a visibilidade do repositório, não submeta o formulário do curso e não faça push em nome do aluno sem acesso/autorização já disponível; deixe claro qualquer etapa de publicação que exija ação/credencial do proprietário. Se o repositório já estiver publicado e as permissões permitirem, verifique o site publicado e reporte a URL real obtida, sem inventá-la.
14. Se dado externo obrigatório não existir localmente (por exemplo, `train.csv` do Spaceship Titanic), procure o arquivo no projeto. Se não estiver, não fabrique dados nem alegue que a análise real foi feita. Deixe caminho de aquisição reproduzível e prossiga com as outras tarefas que não dependem dele; solicite o arquivo ao usuário somente se for indispensável para concluir esse exercício.
15. Execute os scripts necessários e valide os resultados concretos. Inspecione logs/figuras, confirme arquivos referenciados, front matter, tabela completa, execução limpa e build do site. Não diga “testado” sem executar. Não esconda falhas; documente claramente o que não pôde ser concluído.
16. Como é trabalho acadêmico com política explícita de integridade, escreva explicações claras e rastreáveis aos resultados obtidos. Não simule autoria exclusiva nem oculte ajuda de IA. Ao final, destaque que o aluno deve compreender e revisar o código e a análise para poder explicá-los.

PLANO E RELATO FINAL
- Entregue primeiro uma auditoria curta do que já havia no repositório e do que faltava.
- Depois implemente e atualize os quatro relatórios, scripts, figuras, configuração, navegação e workflow necessários.
- Antes de encerrar, apresente uma checklist dos requisitos por exercício com evidência, comandos executados e status real do build/Pages.
- Informe arquivos principais alterados e quaisquer dados/credenciais/ação manual pendentes. Não encerre com template vazio ou resultados inventados.
```

---

# Prompt 1 — Data (exercício 1)

```text
Conclua e revise somente o exercício Data dentro do repositório ANN-DL 2026.2. Leia a página completa, incluindo Results summary e Evaluation Criteria: https://insper.github.io/ann-dl/2026.2/exercises/data/ e confira o formato do relatório em https://insper.github.io/ann-dl/2026.2/exercises/submission/.

Antes de editar, audite os arquivos existentes em `docs/exercises/data/` e compare cada item com o enunciado. Preserve o que já estiver correto. A página final obrigatória é `docs/exercises/data/index.md`, front matter `exercise: data` e `ai_use:` verdadeiro, código executável em `code/`, figuras commitáveis em `figures/`, seções na ordem exata e tabela Results summary final completa.

Implemente/conclua todos os 3 exercícios:
1. Nuvens 2D: gerar 400 pontos, 100 por classe, Gaussianas com médias [2,3], [5,6], [8,1], [15,4] e desvios [0.8,2.5], [1.2,1.9], [0.9,0.9], [0.5,2.0]. Fixar `rng=np.random.default_rng(42)` e reutilizar o mesmo gerador. Fig. 1 scatter com centros. Repetir os mesmos centros para escalas s={0.5,1,2,4}; Fig. 2 em quatro subplots com limites iguais; calcular as seis razões de separação para s=1, identificar a menor e calcular seu valor em s=2; calcular as quatro mixing rates por centro mais próximo e gerar Fig. 3. Responder sobre separabilidade linear e esboçar limites de decisão coerentes sobre Fig. 1; relacionar espalhamento e região de erros.
2. Não linearidade em 5D: dataset I com 500 exemplos por classe, médias [0,0,0,0,0] e [1.5,1.5,1.5,1.5,1.5], usando exatamente as matrizes de covariância da página. Dataset II também com 500 por classe, direções uniformes obtidas normalizando vetores gaussianos em R5, raios normais N(2,0.4) e N(5,0.4) para core/shell. Fig. 4: PCA 2D lado a lado; reportar variância explicada PC1+PC2; distâncias entre centros calculadas em 5D; Fig. 5 histogramas dos raios por classe no mesmo eixo. Explicar hiperplano, por que cascas não são linearmente separáveis, por que projeção PCA misturada não prova inseparabilidade e dar função separadora baseada em `||x||²`.
3. Spaceship Titanic: usar o `train.csv` rotulado. Descrever alvo Transported, balanço, tipos de atributos, tabela de missing absoluto e percentual; média, mediana e máximo das cinco colunas de gasto. Dividir 80/20 estratificado com seed fixa ANTES de ajustar qualquer estatística/imputer/encoder/scaler; explicar leakage. Tratar missing por tipo, one-hot HomePlanet/CryoSleep/Destination/VIP com categoria desconhecida segura, criar TotalSpend, remover Cabin/Name/PassengerId, usar log1p nas cinco colunas de gasto, mostrar histograma antes/depois e padronizar ou normalizar para escala compatível com tanh. Ajustar toda transformação somente no treino, aplicar sem refit no teste. Informar limites após escala, ausência de NaN e shape final; refletir sobre decisão mais impactante. Figura da página é chamada Figure 6.

Regras: apenas numpy, pandas, matplotlib/seaborn e scikit-learn para PCA/preprocessing; nenhum modelo treinado. Toda figura com título/eixos e legenda de classe. `rng` único. Reporte números dentro do texto. Results summary inclui 13 linhas da página (mixing rates, menor r e par, distâncias, variâncias PCA, proporção positiva de Transported, estatísticas FoodCourt, shape final, mínimo/máximo de treino e teste). Use somente valores que o código realmente calculou. Se `train.csv` estiver ausente, não invente: informe o bloqueio e deixe instrução reproduzível de aquisição.

Rode o código a partir de ambiente limpo/explicado, confira figuras e caminhos. Termine com checklist da rubrica e lista de lacunas reais.
```

---

# Prompt 2 — Perceptron (exercício 2)

```text
Conclua e revise somente o exercício Perceptron no repositório ANN-DL 2026.2. Leia por inteiro https://insper.github.io/ann-dl/2026.2/exercises/perceptron/ e o formato oficial https://insper.github.io/ann-dl/2026.2/exercises/submission/.

Audite antes os arquivos de `docs/exercises/perceptron/`. Preserve conteúdo correto. Entregue `docs/exercises/perceptron/index.md`, front matter `exercise: perceptron` e `ai_use:` fiel, scripts em `code/`, figuras em `figures/`, seções conforme a página e Results summary final preenchido.

Exercício 1 (dados separáveis): 1000 pontos por classe 2D; classe 0 média [1.5,1.5], classe 1 [5,5], ambas covariância [[0.5,0],[0,0.5]]. Usar `rng=np.random.default_rng(42)`. Fig. 1 scatter. Implemente do zero step(z)=1 se z>=0 senão 0, predição, atualização para rótulos 0/1 `w += eta*(y-yhat)*x`, `b += eta*(y-yhat)`, inicialização `rng.normal(0,0.01,size=2)`, b=0, eta=0.01, embaralhamento/loop conforme implementação reproduzível e parada após época sem update ou 100 épocas. Registrar acurácia após cada época. Reportar w, b, epochs e acurácia finais; Fig. 2 limite e erros destacados; Fig. 3 acurácia por época. Rodar também eta=1.0, comparar direção de w normalizada e explicar efeito do learning rate. Demonstrar algebricamente por que iniciar w=0,b=0 torna eta simples fator de escala e não muda fronteira/épocas, como pede o item D.

Exercício 2 (classes sobrepostas): 1000 pontos por classe, médias [3,3] e [4,4], ambas covariância [[1.5,0],[0,1.5]], mesmo rng. Fig. 4 scatter. Reusar implementação sem alterar com eta=.01 e limite 100 épocas. Manter pesos finais e pocket (melhor acurácia global depois de cada atualização), reportar w,b, acurácias e época do melhor pocket. Fig. 5 as duas fronteiras e erros; Fig. 6 acurácia atual e pocket-best por época. Discutir por que final pode degradar/próximo ao acaso, explicar o resultado esperado do melhor linear ~73%, comparar Fig. 3/6, hipótese do teorema de convergência que falha, e por que mais épocas ou eta menor não resolve a não separabilidade, referindo-se à regra de update e ao movimento de b e w.

Não use sklearn/qualquer pacote para fornecer o modelo, `fit`, Perceptron ou SGDClassifier. Use apenas NumPy/pandas/matplotlib/seaborn. `rng` fixo, figuras tituladas e rotuladas. Preencha as 8 linhas da tabela Results summary com números calculados. O critério não premia uma acurácia bonita: reporte com honestidade o comportamento obtido. Rode os scripts e confira caminhos/figuras/build; feche com checklist e lacunas reais.
```

---

# Prompt 3 — MLP (exercício 3)

```text
Conclua e revise somente o exercício MLP no repositório ANN-DL 2026.2. Leia toda a página https://insper.github.io/ann-dl/2026.2/exercises/mlp/ e o formato oficial https://insper.github.io/ann-dl/2026.2/exercises/submission/.

Audite primeiro `docs/exercises/mlp/`; mantenha o que estiver correto. Entregue `docs/exercises/mlp/index.md`, front matter `exercise: mlp` e `ai_use:` verdadeiro, scripts em `code/`, figuras em `figures/`, seções e tabela Results summary exatamente como a página requer.

Exercício 1: mostre manualmente, sem biblioteca, forward, derivação de cada gradiente e atualização dos 8 parâmetros para x=[0.5,-0.2], y=1, W1=[[.3,-.1],[.2,.4]], b1=[.1,-.2], W2=[.5,-.3], b2=.2, tanh em ambos os níveis, MSE `(y-yhat)^2`, eta=.3. Apresente intermediários e parâmetros atualizados com pelo menos 4 casas decimais e diga se a loss deve subir ou cair no segundo forward e por quê.

Exercício 2: use `make_blobs` exatamente com centers [[0,0],[3,3],[-3,-3]], `n_samples=[500,250,250]`, `cluster_std=1.2`, random_state=42 e y=(c>0).astype(int); split 80/20 estratificado com random_state=42. Fig. 1. Ajuste baseline LogisticRegression permitido e reporte acurácia teste. Implemente MLP NumPy genérico (layer sizes argumentos) com forward, BCE, backprop derivado e update próprios; sem autodiff. Treine centenas/milhares de épocas e registre loss. Informe arquitetura, LR, epochs, teste (esperado aprox 90–93%, mas sem forçar), Fig. 2 loss, Fig. 3 fronteira em grid. Analise o baseline <50% e geometria da região classe 1.

Exercício 3: `make_classification(n_samples=1500,n_features=4,n_informative=4,n_redundant=0,n_repeated=0,n_classes=3,n_clusters_per_class=2,class_sep=1.0,random_state=42)`, split 80/20 estratificado. Reuse o mesmo núcleo forward/backward/update genérico; adaptação output 3, softmax e categorical cross-entropy. Derive no relatório `dL/du` para softmax+cross-entropy. Reporte arquitetura, hiperparâmetros e teste; Fig. 4 loss e Fig. 5 matriz confusão. Identifique par mais confundido, explique usando 4 dimensões informativas/6 clusters e reporte baseline de LogisticRegression (~66% como expectativa, número real deve ser calculado). Se possível, mantenha o núcleo do Ex.2 literalmente igual e exponha saída/loss como argumentos para o extra credit.

Exercício 4: mesmo dataset do Ex.3. Comparação controlada: somente profundidade muda; ≥2 hidden layers versus 1 hidden layer, mesmo budget de treino e learning rate, repetir ambos com ao menos 3 seeds e reportar médias. Use splits comparáveis e explique protocolo. Fig. 6 com curvas de loss dos dois. Tabela compare hidden layers, número de parâmetros, final train loss e test accuracy. Analise honestamente se profundidade ajudou e o tradeoff representação/otimização; não faça tuning até produzir um vencedor.

Bibliotecas: NumPy/pandas/matplotlib/seaborn; sklearn somente para geradores, splits e confusion matrix, além do LogisticRegression baseline permitido no Ex.2; não TensorFlow, PyTorch, MLPClassifier ou autodiff. `rng=np.random.default_rng(42)` na parte NumPy e seeds dos geradores conforme enunciado. Figuras tituladas, eixos e legendas. Results summary completo (9 linhas). Execute e confira os scripts/figuras e build. Termine com checklist e evidências, sem inventar números.
```

---

# Prompt 4 — VAE (exercício 4)

```text
Conclua e revise somente o exercício VAE no repositório ANN-DL 2026.2. Leia integralmente https://insper.github.io/ann-dl/2026.2/exercises/vae/ e o formato oficial https://insper.github.io/ann-dl/2026.2/exercises/submission/.

Audite primeiro `docs/exercises/vae/`, preserve o correto. Entregue `docs/exercises/vae/index.md`, front matter `exercise: vae` e `ai_use:` fiel, scripts em `code/`, figuras commitáveis em `figures/`, títulos/ordem do enunciado e tabela Results summary final completa.

Reprodutibilidade e regras: `torch.manual_seed(42)` antes de construir cada modelo; split exato com `random_split` e `torch.Generator().manual_seed(42)`. Use torch/torchvision para MNIST, NumPy, matplotlib/seaborn e scipy/sklearn para utilidades quantis/PCA. Autograd permitido, mas escreva manualmente reparameterization, BCE de reconstrução, KL fechado e sampling. Sem classe VAE de biblioteca e sem `torch.distributions.kl_divergence`. reportar BCE e KL separadamente em nats/imagem (BCE soma 784 pixels e média sobre imagens). Teste somente para números finais; decisões de treino pela validação.

Exercício 1, MNIST: torchvision MNIST em [0,1], flatten 784; 60k treino em 50k treino +10k validação pelo split especificado; 10k oficial teste. Arquitetura encoder 784→512→256 ReLU, heads separadas 256→d para mu e logvar, decoder d→256→512→784 ReLU e sigmoid, d=16. Loss BCE somada nos pixels mais beta=1 vezes KL fechado `0.5*sum(mu²+sigma²-1-logvar)`, média por imagens; use BCE `reduction='sum'`/batch size. Adam lr=1e-3, batch 128, 20 epochs; guardar BCE e KL em train e validation por época. Reporte BCE/KL/neg ELBO no teste; Fig. 1 curvas separadas treino/val; Fig. 2 primeira imagem teste de cada dígito e reconstrução de mu (não amostrar z); Fig. 3 grade 8x8 de prior samples após `torch.manual_seed(0)`. Responda por que KL sobe enquanto total cai, dois motivos distintos para amostras do prior parecerem piores/mais borradas e por que reconstruir usando mu.

Exercício 2: repetir mesma arquitetura/dados/seed/beta/budget mudando apenas d=2. Reportar teste BCE/KL ao lado de d=16. Fig. 4 mu das 10k imagens teste coloridas por dígito. Fig. 5 decodificar grade 15x15 em valores z=Phi^-1(q), q uniformemente espaçado [0.05,0.95], scipy.stats.norm.ppf; explique por que isso equaliza massa de prior por célula. Fig. 6 duas imagens teste de dígitos diferentes, reportar índices, 10 passos t: linha de cima interpolação pixel, linha inferior interpolação latente entre os mu no modelo d=16 do exercício 1. Analise sobreposição entre dígitos em Fig. 4/5, custo de comprimir a 2 dimensões e diferenças entre as interpolações.

Results summary exato de 4 linhas: teste BCE/KL/neg ELBO d=16; KL de validação primeira e última época; teste BCE/KL d=2; índices interpolados. Figures 1–6 tituladas e com eixos/legendas quando aplicável. Código comentado, resultados reais e reprodutíveis. Execute em CPU/GPU disponível, reporte ambiente/limitações se impactarem; confira gráficos, assets e build. Não use uma imagem ou score de referência como se fosse resultado seu. Feche com checklist da rubrica e lacunas reais.
```

---

## Conferência manual antes de entregar

- A navegação aponta para quatro páginas separadas: Data, Perceptron, MLP e VAE.
- Cada página tem seu slug obrigatório, front matter completo e `ai_use` declarado.
- Relatórios espelham a ordem do enunciado; Results summary é a última seção e nenhuma linha está vazia.
- Código está em arquivos versionáveis, figuras estão salvas e aparecem no relatório.
- Os números citados foram produzidos pelo código e as análises se referem a eles.
- Workflow do GitHub Pages concluiu a compilação; link público foi confirmado, se publicação e permissões estiverem disponíveis.
- O aluno revisou e consegue explicar escolhas, código, resultados e uso de IA.

## Fontes

- [Data](https://insper.github.io/ann-dl/2026.2/exercises/data/)
- [Perceptron](https://insper.github.io/ann-dl/2026.2/exercises/perceptron/)
- [MLP](https://insper.github.io/ann-dl/2026.2/exercises/mlp/)
- [VAE](https://insper.github.io/ann-dl/2026.2/exercises/vae/)
- [Submission Format](https://insper.github.io/ann-dl/2026.2/exercises/submission/)
