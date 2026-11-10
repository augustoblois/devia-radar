# Diário de bordo · Etapa 2 (DevIA Radar)

Registro honesto dos erros que apareceram durante a implementação e de como
cada um foi resolvido. A professora pede esse conteúdo em **um slide**, então
o slide leva os 4 destacados com ⭐ e o resto fica de reserva para pergunta.

---

## ⭐ 1. O corte de frases por pontuação emendava listas inteiras

**O que aconteceu.** O plano era reaproveitar o corte de frases da etapa 3:
uma regex que corta depois de ponto, exclamação ou interrogação, ou numa
quebra de linha (`(?<=[.!?])\s+|\n+`). Antes de adotar, o corte foi medido
no corpus inteiro: 148 "frases" passavam de 300 palavras.

**Diagnóstico.** A limpeza da etapa 1 apagou as quebras de linha. Item de
lista e linha de checklist não terminam com ponto, então, sem a quebra, um
item emendava no outro. Uma das "frases" começava assim:

> Worker instruction lacks repository/worktree identity: workers probe
> unrelated checkouts before scoping: labels Review evidence CORTXT T5-001
> Deterministic provider-assurance policy gate ( ): contract fit MISSING;
> repository fit MISSING CORTXT T5-002 ...

Uma frase dessas mistura vários assuntos, e o vetor e o sentimento dela não
representam nenhum.

**Correção.** Quem corta as frases é o parser do spaCy (`doc.sents`, em
`pipeline.processar`). Além da pontuação, ele usa a estrutura da frase: onde
termina a árvore de dependências (quem é sujeito e complemento de quem).

| Corte | Frases | Com 6 palavras ou mais | O 1% mais longo passa de | Mais de 300 palavras |
|---|---|---|---|---|
| regex (pontuação) | 33.965 | 30.625 | 189 palavras | 148 |
| parser do spaCy | 39.279 | 34.460 | 116 palavras | 28 |

**Ressalva.** As 28 que sobraram são listas sem pontuação nenhuma, que nem o
parser consegue separar.

**Lição.** A unidade de análise vem antes da análise. Medir o tamanho das
frases levou dois minutos e mudou a base de todas as opções seguintes.

---

## ⭐ 2. O ner do spaCy chamava o Claude de pessoa

**O que aconteceu.** Com o pipeline original do `en_core_web_md`, o ner
(o reconhecedor de entidades) rotulou "Claude Code" como `PERSON` na frase de
demonstração. Medindo nas 2.079 frases que citam ferramenta:

| Ferramenta | Rótulo que o ner deu |
|---|---|
| Claude | `PERSON` (pessoa) em 44% das menções, `LAW` (lei) em 6% |
| GitHub Copilot | `ORG` (organização) em 61% |
| Gemini | `ORG` em 44%, `FAC` (prédio) em 9%, `LOC` (lugar) em 7% |
| ChatGPT | `GPE` (país ou cidade) em 12% |
| Ollama | `LOC` em 11%, `DATE` (data) em 6% |

No resto das menções dessas cinco, o ner não marcou entidade nenhuma (de
30% a 79%, conforme a ferramenta).

**Diagnóstico.** O ner aprendeu com o OntoNotes 5, um conjunto de textos
anotados à mão publicado em 2013, quase dez anos antes dessas ferramentas.
Num texto dessa época, "Claude" só pode ser nome de gente, e "Gemini", a
constelação ou o programa espacial.

**Correção.** Um `entity_ruler` antes do ner, com o mesmo dicionário de
ferramentas da etapa 1 (`pipeline.carregar_modelo`):

```python
regras = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
regras.add_patterns(padroes_ferramentas())
```

Cada variação ("github copilot", "copilot") vira uma entidade `FERRAMENTA`,
com o nome canônico guardado no `id`. Como a regra vem antes, o ner respeita
o que ela marcou. A opção 2 do menu mostra as duas versões lado a lado.

**Lição.** Modelo pré-treinado não conhece o que surgiu depois do treino. A
regra cobre o que a gente sabe; o modelo cobre o resto.

---

## ⭐ 3. A regra parecia perder 61 issues do Claude

**O que aconteceu.** A primeira comparação entre a regra nova e a regex da
etapa 1 mostrava a regra achando menos: Claude em 244 issues, contra 305 da
etapa 1. O ChatGPT caía de 302 para 273, e o Copilot, de 169 para 134.

**Diagnóstico.** A comparação usava o campo `ferramentas_ia` do corpus, e a
etapa 1 extraiu esse campo do texto **bruto**, antes de apagar código e
link. A regra roda no texto **limpo**. Um `claude` dentro de bloco de código
contava para a regex e não existia mais para a regra.

**Correção.** A regex da etapa 1 roda de novo, no mesmo texto limpo
(`pipeline._regex_etapa1`):

| Ferramenta | regex, texto bruto | regex, texto limpo | regra (spaCy) |
|---|---|---|---|
| Claude | 305 | 250 | 244 |
| ChatGPT | 302 | 276 | 273 |
| GitHub Copilot | 169 | 134 | 134 |

A diferença que sobra é nome dentro de arquivo, domínio ou caminho
(`CLAUDE.md`, `chatgpt.com`, `.claude/projects`, `llama.cpp`, `gpt-4.1`). O
tokenizer do spaCy guarda cada um desses como um token só, e a regra compara
token inteiro.

O Cursor anda no sentido contrário (44 issues pela regex, 154 pela regra),
porque a regra conta "Cursor" sozinho com maiúscula e a regex da etapa 1 só
contava "cursor ai", "cursor editor" e "cursor ide". Numa amostra de 25 das
399 frases com "Cursor" sozinho, conferidas uma a uma, as 25 falavam do
editor.

**Lição.** Comparar duas medidas só vale quando as duas olham para o mesmo
texto.

---

## ⭐ 4. O vizinho mais parecido de "copilot" era um hidroavião

**O que aconteceu.** A primeira versão da opção 4 mostrava os vizinhos mais
parecidos de cada ferramenta nos vetores de palavra do spaCy
(`most_similar`). Saiu isto:

| Palavra | Vizinhos com similaridade 1,00 |
|---|---|
| copilot | Seaplane, Parkzone, hks |
| claude | DUVALIER, christiane, Boulevard |
| cursor | Initialize, mappings, Header |

Similaridade 1,00 quer dizer vetor idêntico, e "copilot" não é sinônimo de
hidroavião.

**Diagnóstico.** O `en_core_web_md` guarda 20.000 vetores para 684.830
palavras. Para caber, o modelo foi podado: cada palavra rara passou a usar o
vetor de uma palavra frequente parecida, e todas as que usam o mesmo vetor
saem com similaridade 1,00 entre si. "chatgpt" e "ollama" nem vetor têm: os
vetores foram treinados com texto da web coletado em 2021 (mais Wikipédia,
legendas de filme e notícias), antes das duas existirem.

**Correção.** A opção 4 mostra, para cada palavra, quais outras usam
exatamente o mesmo vetor (`embeddings.mesmo_vetor`), que é o que o modelo
sabe de fato. "copilot" divide o vetor com 29 palavras, quase todas de
aviação (biplane, cessna, glider); "claude", com 20, quase todas nomes de
pessoa; "gemini", com 14, entre elas o signo aquarius. Para as análises do
corpus, o projeto usa vetores contextuais (o MiniLM), que leem a frase
inteira.

**Lição.** O vetor estático é uma foto do vocabulário no dia do treino.
Palavra nova, ou com sentido novo, fica de fora da foto.

---

## 5. Os temas do KMeans eram, em parte, repositórios

**O que aconteceu.** O número de temas foi escolhido pela maior silhueta,
entre 4 e 12 grupos, e deu 4. Abrindo o tema 1 (362 issues), 119 delas (33%)
vinham de só 4 repositórios, com títulos no mesmo molde: "Close production
broker provider-authority and runtime-assurance gaps", "Ratified plans can
bypass governed issue-record policy".

**Diagnóstico.** Alguns projetos abrem dezenas de issues com o mesmo modelo
de texto, aparentemente com agentes de IA. Texto parecido dá vetor parecido,
e o KMeans junta. As silhuetas confirmam que os grupos se sobrepõem: todas
ficaram entre 0,02 e 0,08 (a escala vai de -1 a +1). Testar mais grupos
piorou: com 7, apareceu um grupo de 47 issues em vietnamita; com 6, um de 28
issues sobre o PySimpleGUI.

**Decisão.** Ficou o critério fixado antes de olhar o resultado (a maior
silhueta, k = 4), e o README registra a concentração. Os outros três temas
vêm espalhados: o maior repositório de cada um tem de 5 a 21 issues.

**Lição.** Agrupamento acha o que é parecido, e parecido pode ser o assunto
ou o autor. Só abrindo o grupo dá para saber qual dos dois.

---

## 6. Instalar o transformers quebrou outro projeto do computador

**O que aconteceu.** O `pip install` do `sentence-transformers` atualizou o
`huggingface-hub` para a versão 2.4.0. O `gradio-client`, de outro projeto do
mesmo computador, exige uma versão abaixo da 2.0, e o outro projeto ficou com
uma dependência incompatível.

**Correção.** Voltar para a última versão que atende os dois:

```powershell
pip install "huggingface-hub>=1.31,<2"
```

Ficou a 1.33.0. O `transformers` 5.22 aceita de 1.31 até antes da 3.0, e o
`gradio-client`, qualquer uma abaixo da 2.0.

**Lição.** O pip instala tudo no mesmo Python, e um projeto pode quebrar o
outro. Um ambiente virtual (`python -m venv .venv`) por projeto isola as
versões.

---

## 7. O transformer levou 64 minutos na CPU

**O que aconteceu.** O primeiro teste do transformer de sentimento
classificou 640 frases sorteadas em 51 segundos: 13 frases por segundo.
Pela conta, as 34.460 frases do corpus levariam 46 minutos. A execução
completa levou 64 (3.840 segundos). Os vetores do MiniLM, mais leve, já
tinham levado 837 segundos.

**Diagnóstico.** A opção 8 mostra o progresso a cada 3.200 frases, e as
frases vão em ordem de tamanho, da mais curta para a mais longa. O log mostra
quanto pesa cada faixa:

| Frases | Tempo |
|---|---|
| as 16.000 mais curtas (46%) | 7 minutos |
| as 2.460 mais longas (7%) | 24 minutos |

O tempo de uma frase cresce com o tamanho dela. E o computador rodava outros
programas ao mesmo tempo, entre eles um teste de velocidade do próprio
transformer. O log mostra a disputa pela CPU: o bloco de 16.000 a 19.200
frases levou 321 segundos, mais que o seguinte, de frases mais longas (134
segundos). A conta feita com 640 frases não previa nenhum dos dois efeitos.

**Correção.** Três medidas, nenhuma muda o resultado:

- as frases vão em lotes de 32, ordenadas por tamanho. Num lote, a frase
  curta é completada com enchimento até o tamanho da maior; com frases de
  tamanho parecido no mesmo lote, sobra pouco enchimento. Num teste com
  1.280 frases sorteadas, a ordem por tamanho levou 130 segundos, e a ordem
  original, 697, com o mesmo resultado;
- o resultado fica em cache (`dados/brutos/`), e a opção 8 só recalcula com
  o `r` do menu;
- a demonstração ao vivo (opção 7) classifica uma frase de cada vez: as
  quatro frases de exemplo saem em 0,6 segundo.

**Lição.** Modelo grande na CPU pede planejamento: rodar uma vez antes da
apresentação e mostrar o cache. E uma estimativa de tempo feita com amostra
pequena, num computador ocupado, precisa de folga.

---

## 8. Avisos do Hugging Face no meio da demonstração

**O que aconteceu.** Toda carga do transformer imprimia um quadro
`LOAD REPORT` avisando que os pesos `roberta.pooler.dense` sobravam, e cada
modelo consultava a internet com o aviso "You are sending unauthenticated
requests to the HF Hub".

**Diagnóstico.** Os dois avisos são inofensivos. O pooler é uma camada do
pré-treino que o classificador de sentimento não usa, e a consulta só
confere se o modelo tem versão nova. Mas apareciam no meio da tela da
demonstração, e a consulta dependia da internet da sala.

**Correção.** `logging.set_verbosity_error()` do `transformers` cala os
avisos de carga, e o `config.py` liga o modo offline do Hugging Face
(`HF_HUB_OFFLINE`) quando os dois modelos já estão no computador.

---

## 9. A busca dizia "nenhuma palavra em comum" para frases com "AI"

**O que aconteceu.** No ensaio da opção 6, a consulta "the AI invented a
function that does not exist" trouxe 8 frases, todas com "AI", e o programa
marcou as 8 como "nenhuma palavra em comum com a consulta". A última linha
também saiu estragada: `['exist'. 'function'. 'invented']` e "322 frases.
sem ordem de relevância".

**Diagnóstico.** Dois erros no `embeddings.py`:

- a comparação só contava palavras de 4 letras ou mais, para pular "the" e
  "of". Com isso, "AI" (2 letras) nunca contava;
- o `.replace(",", ".")`, que põe o ponto do milhar no número (34.460), valia
  para a frase inteira e trocou também as vírgulas do texto.

E a consulta era ruim: o corpus quase não tem frase sobre função inventada,
e o vetor da consulta ficou dominado por "AI".

**Correção.** A comparação passou a usar as stopwords do spaCy (a lista de
palavras gramaticais) e o radical de cada palavra (stemmer de Porter, do
NLTK), que junta "copyright" e "copyrighted". O ponto do milhar vai só no
número. Frase repetida aparece uma vez só, porque alguns projetos abrem
várias issues com o mesmo modelo de texto. A consulta de exemplo virou "who
owns the copyright of code written by a machine", que traz frases sobre
licença, processo judicial e direito autoral de código gerado por IA. A
busca literal pelas mesmas palavras casaria com 4.887 frases, sem ordem.

**Lição.** A saída da demonstração também é resultado: precisa ser conferida
linha por linha, como um gráfico.

---

## 10. As frases de exemplo não mostravam o contraste prometido

**O que aconteceu.** A opção 4 compara o vetor estático com o contextual em
pares de frases. O par de paráfrases sobre função inventada saiu com 0,41 no
contextual, abaixo do par armadilha, que só divide a palavra "cursor"
(0,42). Na opção 7, o pedido "I would like to add support for Gemini
models." saiu 50% positivo e 49% neutro no transformer: um empate, em vez do
neutro que o exemplo devia mostrar.

**Diagnóstico.** As frases foram escritas antes de rodar os modelos, e
nenhuma foi medida antes de entrar no código. O que um modelo faz com uma
frase não dá para adivinhar: as duas frases da opção 7 têm "would like", e
uma saiu positiva e a outra, neutra.

**Correção.** Frases testadas antes de entrar no código:

| Opção | Frase nova | Resultado |
|---|---|---|
| 4 | "Put the mouse pointer after the last character." (paráfrase de "Move the cursor to the end of the line.") | 0,65 no contextual, contra 0,42 do par que divide "cursor"; no estático, o contrário (0,80 contra 0,86) |
| 7 | "We would like Copilot to support custom prompts." | VADER +0,637 (positivo, por "like" e "support"); transformer 75% neutro |

**Lição.** Exemplo de demonstração se escolhe medindo, como qualquer
parâmetro.
