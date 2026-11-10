# Etapa 2 · Semântica e sentimento (issues do GitHub)

Pipeline do spaCy, embeddings e transformers aplicados às mesmas 979 issues
da etapa 1. A etapa 1 contou palavras; a etapa 2 mede o que elas querem dizer.

Slides: [`apresentacao/etapa2-devia-radar.pdf`](apresentacao/etapa2-devia-radar.pdf) ·
Diário de bordo: [`diario-de-bordo.md`](diario-de-bordo.md)

---

## Execução

```powershell
cd etapa2
python main.py
```

Abre o menu interativo, que é o modo usado na apresentação.
Para rodar tudo de uma vez, sem interação:

```powershell
python main.py --tudo
```

Cada módulo também roda sozinho, útil para demonstrar uma parte isolada:

```powershell
python pipeline.py      # o pipeline do spaCy numa frase, componente a componente
python embeddings.py    # vetor estático x vetor contextual
python sentimento.py    # a mesma frase lida pelo VADER e pelo transformer
```

### Antes da apresentação

A etapa 2 não coleta nada: lê o corpus limpo da etapa 1
(`etapa1/dados/processados/corpus_limpo.json`, que já está no Git). Precisa
de três modelos, que ficam guardados no computador depois da primeira vez:

| Modelo | Para quê | Download | Como chega |
|---|---|---|---|
| `en_core_web_md` (spaCy) | pipeline e vetores estáticos | ~40 MB | `python -m spacy download en_core_web_md` |
| `all-MiniLM-L6-v2` (sentence-transformers) | vetores de frase | ~90 MB | sozinho, na primeira opção 4, 5 ou 6 |
| `twitter-roberta-base-sentiment-latest` (Cardiff NLP) | sentimento | ~500 MB | sozinho, na primeira opção 7 ou 8 |

As partes lentas ficam em cache em `dados/brutos/` (fora do Git, 66 MB).
Tempos medidos na CPU de um notebook, sem placa de vídeo:

| Parte | Opção | Primeira vez | Com o cache |
|---|---|---|---|
| pipeline do spaCy nas 979 issues | 3 | ~3 min | instantâneo |
| vetores das 34.460 frases (MiniLM) | 5 | ~14 min | instantâneo |
| sentimento das 34.460 frases (RoBERTa) | 8 | ~64 min | instantâneo |

Rode a opção 0 uma vez antes do dia. Depois disso o programa roda sem
internet: quando os dois modelos do Hugging Face já estão no computador, o
`config.py` liga o modo offline sozinho.

---

## Opções do programa

| Opção | O que faz |
|---|---|
| 1 | Lê o corpus da etapa 1 (979 issues) e mostra o resumo |
| 2 | Demonstra o pipeline do spaCy numa frase, componente a componente, com e sem a nossa regra de ferramentas |
| 3 | Passa o corpus pelo pipeline: frases, ferramentas e entidades |
| 4 | Demonstra embeddings: vetor estático (spaCy) x vetor contextual (MiniLM) |
| 5 | Calcula os vetores do corpus e acha os temas (KMeans e t-SNE) |
| 6 | Busca semântica: acha frases pelo significado, sem exigir as mesmas palavras |
| 7 | Demonstra o transformer de sentimento numa frase, ao lado do VADER |
| 8 | Classifica o sentimento de todas as frases: transformer x VADER |
| 9 | Gera os 9 gráficos e o dashboard |
| 0 | Roda o pipeline completo |
| r | Refaz tudo do zero, ignorando o cache |

As opções 2 e 7 aceitam uma frase digitada na hora (em inglês); com enter
vazio, usam as frases de exemplo.

---

## Estrutura

```
etapa2/
├── config.py          parâmetros centrais da etapa
├── pipeline.py        spaCy: o pipeline com entity_ruler e o componente vader
├── embeddings.py      vetores estáticos e contextuais, temas e busca semântica
├── sentimento.py      transformer RoBERTa x VADER, frase a frase
├── visualizacao.py    matplotlib: 9 gráficos
├── main.py            menu e orquestração do pipeline
├── diario-de-bordo.md erros da implementação e as correções
├── dados/
│   ├── brutos/        frases anotadas, vetores e classificação (cache, fora do Git)
│   └── processados/   resultados do pipeline, dos temas e do sentimento
├── saida/             gráficos PNG prontos para os slides
└── apresentacao/      slides em PDF (o pptx e os roteiros de fala ficam só no computador da dupla)
```

## Pipeline

```
corpus limpo da etapa 1 (979 issues)
        ↓  pipeline.py        spaCy + entity_ruler + vader: 1.033.315 tokens, 39.279 frases
sentence-transformers (MiniLM)
        ↓  embeddings.py      34.460 frases e 979 issues viram vetores de 384 números
KMeans + t-SNE
        ↓  embeddings.py      4 temas e o mapa semântico das issues
transformer (RoBERTa)
        ↓  sentimento.py      34.460 frases classificadas e comparadas com o VADER
matplotlib
        ↓  visualizacao.py    9 gráficos, incluindo o dashboard consolidado
```

---

## O que cada técnica faz aqui

### Pipeline do spaCy (`pipeline.py`)

O spaCy processa o texto numa linha de montagem, o pipeline. O tokenizer
pica o texto em tokens e cria o `Doc`; depois, cada componente acrescenta uma
anotação e passa o `Doc` adiante. O `en_core_web_md` vem com seis:

`tok2vec → tagger → parser → attribute_ruler → lemmatizer → ner`

A etapa acrescenta dois componentes:

- **`entity_ruler`**, antes do `ner`: o dicionário de ferramentas da etapa 1
  vira regra. Cada variação ("github copilot", "copilot") vira uma entidade
  `FERRAMENTA`, com o nome canônico guardado no `id`. Sem ela, o `ner` do
  modelo chama o Claude de pessoa (ver o diário de bordo).
- **`vader`**, um componente nosso (`@Language.component`), no fim: dá a cada
  frase a nota do VADER num atributo novo, `frase._.vader`.

O `parser` também corta as frases (`doc.sents`), que viram a unidade de todas
as análises seguintes. Frase com menos de 6 palavras ("Thanks!", "Steps to
reproduce:") fica de fora: 34.460 das 39.279 entram.

### Embeddings (`embeddings.py`)

Embedding é uma lista de números que representa o significado de uma palavra
ou de uma frase. Significado parecido dá vetor parecido, e "parecido" vira
conta: a similaridade do cosseno, que mede o ângulo entre dois vetores
(1 = mesma direção; perto de 0 = sem relação).

- **Estático** (spaCy): um vetor fixo de 300 números por palavra. "cursor"
  tem um vetor só, seja o do mouse, seja o editor, e o vetor de uma frase é a
  média dos vetores das palavras dela.
- **Contextual** (sentence-transformers, modelo MiniLM): um transformer lê a
  frase inteira e devolve 384 números para ela. A mesma palavra muda de vetor
  conforme as vizinhas.

Com os vetores de frase, duas aplicações:

- **temas**: o KMeans junta as issues de vetor parecido em grupos. O número de
  grupos, de 4 a 12, é o de maior silhueta (nota de -1 a +1 que mede se cada
  issue está mais perto do próprio grupo que do vizinho). O c-TF-IDF dá nome a
  cada grupo, com as palavras frequentes nele e raras nos outros, e o t-SNE
  desenha os 384 números num plano;
- **busca semântica**: a consulta vira vetor, e o programa devolve as frases
  de vetor mais parecido, mesmo sem nenhuma palavra em comum. A busca da
  etapa 1 era por regex, palavra exata.

### Transformer de sentimento (`sentimento.py`)

O VADER da etapa 1 é um léxico: uma tabela de 7.500 palavras com nota, somadas
com algumas regras (negação, ênfase, "but"). Ele soma a nota das palavras que
conhece, sem ler a frase como um todo.

O transformer lê a frase inteira. Em cada camada, cada pedaço de palavra
"olha" para todos os outros (o mecanismo de atenção) e ajusta o próprio vetor
conforme o contexto. O modelo usado é o RoBERTa (125 milhões de parâmetros)
da Cardiff NLP, que continuou o pré-treino em 124 milhões de tweets e foi
ajustado para três classes: negativo, neutro e positivo, as mesmas do VADER.
Ele devolve uma nota crua (logit) para cada classe, e a softmax transforma as
três notas em probabilidades que somam 1.

As 34.460 frases passam pelos dois medidores. A comparação usa a matriz de
confusão (a tabela que cruza as duas classificações) e o kappa de Cohen (a
concordância descontando a que o acaso daria: 0 é o acaso, 1 é concordar
sempre).

---

## Principais resultados

**O pipeline.** As 979 issues somam 1.033.315 tokens e 39.279 frases, e
34.460 delas têm 6 palavras ou mais. 2.079 frases citam uma ferramenta de IA.
Sem a regra de ferramentas, o `ner` do modelo marca o Claude como pessoa
(`PERSON`) em 44% das 845 menções; com o `entity_ruler` antes dele, em 0%.

**Os embeddings.** Em 200 pares de frases sorteados do corpus, sem relação
entre si, a similaridade média é 0,73 no vetor estático e 0,09 no
contextual: na média de vetores de palavra, duas frases quaisquer já saem
parecidas. Nas frases de exemplo da opção 4, o estático acha "Cursor wrote the
whole module for me." e "Move the cursor to the end of the line." mais
parecidas (0,86) que duas frases de mesmo sentido (0,80); o contextual faz o
contrário (0,42 e 0,65).

**Os temas.** O KMeans achou 4 temas (silhueta de 0,08):

| Tema | Issues | Frases | Termos característicos (c-TF-IDF) | Frases negativas (transformer) |
|---|---|---|---|---|
| 1 | 362 | 21.448 | review, test, evidence, task, path, require | 13% |
| 2 | 276 | 8.092 | agent, tool, model, code, llm, mcp | 8% |
| 3 | 228 | 3.092 | code, chatgpt, generate, version, license, error | 19% |
| 4 | 113 | 1.828 | cursor, editor, code, claude, agent, user | 13% |

O tema 3, de código gerado, versão e licença, é o mais negativo.

**A busca semântica.** A consulta "who owns the copyright of code written by
a machine" devolve frases sobre código com copyright, licença e processo
judicial. A busca literal pelas mesmas palavras (e variações delas) casaria
com 4.887 frases, sem ordem de relevância.

**O sentimento das 34.460 frases.**

| Medidor | Positivas | Neutras | Negativas |
|---|---|---|---|
| VADER, issue inteira (como a etapa 1 mediu) | 65% | 4% | 31% |
| VADER, frase a frase | 34% | 42% | 24% |
| transformer, frase a frase | 3% | 85% | 12% |

- Os dois medidores concordam em 50% das frases, com kappa de Cohen de 0,16
  (concordância fraca).
- Para o transformer, quase tudo é neutro. Quando há tom, 78% das vezes é
  queixa: 4.159 frases negativas contra 1.169 positivas.
- **O VADER lê jargão como elogio.** "commit" (+1,2 no léxico), "acceptance"
  (+2,0) e "authority" (+0,3) são positivas para ele, e o transformer
  discordou em 1.504 das 1.513 frases com elas. Numa issue, "commit" é o
  comando do git, e "acceptance" aparece em critério de aceite (302 das 581
  frases com a palavra). Isso ajuda a explicar as 65% de issues positivas da
  etapa 1.
- Do lado da queixa, "no" (-1,2) engana mais: o transformer discordou em
  1.616 das 2.143 frases com a palavra.
- A soma do léxico também esconde queixa. Em "That is wrong for this work in
  every respect.", "wrong" (-2,1) e "respect" (+2,1) se anulam, e o compound
  dá 0,00; o transformer dá 93% negativo.
- Por ferramenta, a parcela de frases negativas (transformer) vai de 17% no
  ChatGPT (691 frases) a 3% no Codeium (74): GitHub Copilot 11% (302),
  Claude 10% (631), Cursor 9% (485), Gemini 9% (93) e Ollama 7% (73).

---

## Limitações

- **Os temas se sobrepõem.** As silhuetas ficaram entre 0,02 e 0,08 para
  qualquer número de grupos, e 119 das 362 issues do tema 1 vêm de só 4
  repositórios, com títulos no mesmo molde. O tema 1 mede, em parte, o jeito
  de escrever de poucos projetos.
- **O transformer foi ajustado em tweets.** Ele lê instrução técnica como
  queixa: "Kill the process and restart the server." sai 79% negativa. Os
  dois medidores discordam em metade das frases, e nenhum é gabarito: medir
  qual erra mais pediria uma amostra rotulada à mão.
- **Issue longa pesa mais.** A unidade é a frase, e 4 repositórios somam 31%
  das frases (o maior, sozinho, 13%). O tema 1, de issues longas e
  estruturadas, tem 62% das frases, e os totais por frase seguem esse tema.
- **O corpus é o da etapa 1**, com o mesmo viés: a Search API do GitHub
  favorece issue recente, e 190 issues vieram do termo `ai code review`.
- **28 frases ainda passam de 300 palavras.** São listas sem pontuação, que
  nem o parser separa.

---

## Onde aprender as bibliotecas usadas

| Biblioteca | Material |
|---|---|
| spaCy | Pipeline: <https://spacy.io/usage/processing-pipelines> · regras e `entity_ruler`: <https://spacy.io/usage/rule-based-matching#entityruler> · vetores: <https://spacy.io/usage/linguistic-features#vectors-similarity> |
| sentence-transformers | <https://sbert.net> (busca semântica e agrupamento têm exemplo pronto) |
| Hugging Face transformers | <https://huggingface.co/docs/transformers/quicktour> e o curso gratuito <https://huggingface.co/learn/llm-course> |
| Transformer, com desenhos | <https://jalammar.github.io/illustrated-transformer/> |
| scikit-learn | KMeans e silhueta: <https://scikit-learn.org/stable/modules/clustering.html#k-means> · t-SNE: <https://scikit-learn.org/stable/modules/manifold.html#t-sne> |
| NLTK / VADER | <https://www.nltk.org/howto/sentiment.html> |
