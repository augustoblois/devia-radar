# DevIA Radar

Analisador de como desenvolvedores discutem **Inteligência Artificial no
desenvolvimento de software**, a partir de issues reais coletadas do GitHub.

Projeto da disciplina **Dados Não Estruturados** (UFPB), feito em dupla por
**Augusto Blois** e **Pedro Flávio Nogueira Ribeiro de Luna**.
Tema 1 da lista de temas, dentro do grande tema de Engenharia de Software.

**Pergunta de pesquisa:** como desenvolvedores discutem o uso de Inteligência
Artificial no desenvolvimento de software?

---

## Estado das etapas

| Etapa | Assunto | Situação |
|---|---|---|
| 1 | PLN com expressões regulares e spaCy + coleta via Web API | **pronta** |
| 2 | Análise de áudio (podcasts e entrevistas sobre o tema) | a fazer |
| 3 | Análise de vídeo (palestras e vídeos técnicos sobre o tema) | a fazer |

As três etapas compartilham o mesmo tema, como o próprio enunciado sugere na
dica final: a fonte de texto não precisa ser a mesma fonte de áudio e vídeo.

---

## Instalação

```powershell
git clone https://github.com/augustoblois/devia-radar.git
cd devia-radar
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

O download do modelo é obrigatório. O spaCy instala a biblioteca, mas o modelo
de idioma vem separado.

## Execução

```powershell
cd etapa1
python main.py
```

Abre o menu interativo, que é o modo usado na apresentação.
Para rodar tudo de uma vez, sem interação:

```powershell
python main.py --tudo
```

Cada módulo também roda sozinho, útil para demonstrar uma parte isolada:

```powershell
python limpeza.py      # mostra o antes e o depois da regex
python pln.py          # anota uma frase token a token com o spaCy
python sentimento.py   # roda só a análise de sentimento
```

---

## Opções do programa

| Opção | O que faz |
|---|---|
| 1 | Coleta issues do GitHub pela Search API |
| 2 | Resumo do corpus coletado (volume, repositórios, período) |
| 3 | Demonstra as expressões regulares em texto real, antes e depois |
| 4 | Limpa o corpus e extrai padrões com regex |
| 5 | Demonstra o spaCy token a token (lema, POS, dependência, stopword) |
| 6 | Analisa o corpus com spaCy (lemas, classes gramaticais, entidades) |
| 7 | Analisa sentimento com VADER |
| 8 | Gera os 8 gráficos e o dashboard |
| 9 | Busca por padrão regex no corpus, com a palavra em contexto (KWIC) |
| 0 | Roda o pipeline completo |
| r | Recoleta do zero, ignorando o cache |

---

## Estrutura

```
dados-nao-estruturados/
├── README.md
├── requirements.txt
├── etapa1-devia-radar.pdf         slides da apresentação da etapa 1 (9 slides)
├── pergunta-de-pesquisa.pdf       pergunta de pesquisa e visão geral do projeto
├── docs/
│   └── diario-de-bordo.md         erros da implementação e as correções
└── etapa1/
    ├── config.py          parâmetros centrais do projeto
    ├── coleta.py          GitHub Search API + cache em disco
    ├── limpeza.py         expressões regulares: limpeza e extração
    ├── pln.py             spaCy: lemas, POS, NER, bigramas
    ├── sentimento.py      VADER: polaridade por documento e por ferramenta
    ├── visualizacao.py    matplotlib e wordcloud: 8 gráficos
    ├── main.py            menu e orquestração do pipeline
    ├── dados/
    │   ├── brutos/        resposta crua da API (cache)
    │   └── processados/   corpus limpo e resultados das análises
    └── saida/             gráficos PNG prontos para os slides
```

## Pipeline

```
GitHub Search API
        ↓  coleta.py          1.180 issues, 746 repositórios
expressões regulares
        ↓  limpeza.py         979 documentos úteis, 29,6% de ruído removido
spaCy (en_core_web_sm)
        ↓  pln.py             356.966 tokens, 13.391 lemas únicos
VADER (NLTK)
        ↓  sentimento.py      polaridade por documento e por ferramenta
matplotlib + wordcloud
        ↓  visualizacao.py    8 gráficos, incluindo o dashboard consolidado
```

---

## O que cada técnica faz aqui

### Expressões regulares (`limpeza.py`)

Duas funções distintas.

**Limpar.** Issue do GitHub mistura linguagem natural com bloco de código,
stack trace, HTML, link e markdown. Sem remover isso, as palavras mais
frequentes do corpus seriam `import`, `def` e `https`. A regex identifica cada
tipo de ruído pela forma e apaga.

**Extrair.** A mesma técnica colhe informação estruturada antes de apagar:
quais ferramentas de IA a issue cita, quais linguagens de programação, quais
versões, se mostra código, se tem stack trace.

### spaCy (`pln.py`)

Pipeline `tokenizer → tagger → lemmatizer → parser → ner`, aplicado em lote
com `nlp.pipe()`. O módulo produz:

- frequência de **lemas** (forma de dicionário da palavra);
- ranking separado por **classe gramatical** (substantivo, verbo, adjetivo);
- **entidades nomeadas** (organização, produto, pessoa, lugar);
- **bigramas**, pares de palavras vizinhas;
- **frases nominais** (noun chunks);
- estatísticas do corpus, incluindo riqueza lexical (TTR).

### VADER (`sentimento.py`)

Análise de sentimento baseada em léxico, sem necessidade de treino. Devolve um
`compound` entre −1 e +1 por documento, agregado depois por ferramenta de IA e
por mês.

---

## Principais resultados

Corpus de **979 issues** de **633 repositórios**, coletadas em setembro de 2026.

- O volume de issues na amostra **cresceu cerca de 11 vezes**, de 13 issues em
  janeiro de 2025 para 149 em setembro de 2026, e setembro ainda é mês
  incompleto. Ressalva importante: parte dessa curva é viés de amostragem, já
  que a Search API favorece conteúdo recente. A tendência de alta é real, a
  magnitude exata não deve ser lida como estatística de população.
- O tom geral é **positivo**: 65,5% das issues positivas contra 30,5%
  negativas, com `compound` médio de +0,28.
- As ferramentas mais citadas são **Claude** (305 issues), **ChatGPT** (302) e
  **GitHub Copilot** (169).
- Os bigramas revelam o assunto dominante: `claude code`, `generate code`,
  `mcp server`, `code review`.
- O termo `review` é o segundo lema mais frequente do corpus, o que aponta
  revisão de código como o ponto de atrito principal do uso de IA.

Os números saem do estado atual do cache em `etapa1/dados/`. Recoletar com a
opção `r` atualiza tudo, e os valores mudam porque o GitHub recebe issues
novas todos os dias.

---

## Onde aprender as bibliotecas usadas

| Biblioteca | Material |
|---|---|
| spaCy | Curso oficial gratuito: <https://course.spacy.io> e <https://spacy.io/usage/spacy-101> |
| Expressões regulares | <https://regex101.com> (testador com explicação passo a passo) e `re` na doc do Python |
| NLTK / VADER | <https://www.nltk.org/howto/sentiment.html> |
| GitHub Search API | <https://docs.github.com/en/rest/search/search#search-issues-and-pull-requests> |
| matplotlib | <https://matplotlib.org/stable/gallery/index.html> |
