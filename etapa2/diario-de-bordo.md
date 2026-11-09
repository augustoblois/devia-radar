# Diário de bordo · Etapa 2 (DevIA Radar)

Registro honesto dos erros que apareceram durante a implementação e de como
cada um foi resolvido. A professora pede esse conteúdo em **um slide**, então
o slide leva os 4 destacados com ⭐ e o resto fica de reserva para pergunta.

---

## ⭐ 1. O VADER contava a muleta "like" como elogio

**O que aconteceu.** A primeira comparação dizia que a fala dos podcasts era
mais positiva que o texto das issues: 40% de trechos positivos na fala,
contra 34% de frases positivas no texto.

**Diagnóstico.** Somando a nota de cada palavra do léxico do VADER em toda a
fala, "like" sozinho dava +2.486, quatro vezes a segunda colocada. O VADER dá
+1,5 a "like" pensando no verbo "gostar". Na fala, "like" aparece 16 vezes a
cada mil palavras (no texto das issues, 0,5), e o spaCy marca só 4% delas
como verbo. O resto é muleta ("it's like, you know") ou comparação
("like this").

**Correção.** A palavra sai do léxico nas duas mídias, para a regra continuar
igual dos dois lados (`config.SEM_SENTIMENTO`, aplicado em `conteudo._vader`):

```python
for palavra in config.SEM_SENTIMENTO:
    _VADER.lexicon.pop(palavra, None)
```

**Resultado.** A fala caiu de 40% para 31% de trechos positivos, e o texto
quase não mexeu (de 34,4% para 33,9%). A conclusão inverteu: a fala tem menos
frases positivas que o texto, e muito menos negativas (9% contra 25%).

**Lição.** Léxico de sentimento carrega o jeito de escrever da fonte em que
foi montado. O VADER veio de rede social escrita, e a muleta da fala virou
"opinião".

---

## ⭐ 2. O Whisper escrevia "cloud" onde o podcast dizia "Claude"

**O que aconteceu.** Nos 10 primeiros minutos do episódio do Y Combinator
sobre Claude Code, o Whisper sem ajuda escreveu "Claude" 13 vezes e "cloud"
3 vezes, e as 3 eram o Claude. No mesmo episódio, "Sonnet 3.5" virou
"Sonic 3.5" e "Claude Code" virou "FODCode".

**Primeira tentativa.** O parâmetro mais conhecido para isso é o
`initial_prompt`, um texto de contexto entregue ao modelo antes do áudio. Com
os nomes das ferramentas nele, piorou: 4 acertos e 11 erros ("Cloud"). No
faster-whisper, esse texto só pesa nas primeiras janelas de 30 s. Depois, o
contexto passa a ser o próprio texto já transcrito, com os erros junto.

**Correção.** O parâmetro `hotwords`, que o faster-whisper põe na frente de
**toda** janela de 30 s (`config.PROMPT_VOCABULARIO`):

| Vocabulário | "Claude" | "cloud" (erro) |
|---|---|---|
| nenhum | 13 | 3 |
| `initial_prompt` | 4 | 11 |
| `hotwords` | 16 | 0 |

"Sonnet 3.5" também voltou. A opção 5 do menu repete o teste ao vivo, num
trecho de 20 segundos.

**Efeito colateral.** A lista puxa som ambíguo para os nomes dela. Aos 141 s
do mesmo episódio, a transcrição completa escreveu "Claude Codex had just
come out", e o convidado disse "Claude Code" ("Codex" também está na lista).
A contagem não muda, porque continua sendo o Claude, mas o texto ganhou uma
palavra que ninguém disse.

**Lição.** Parâmetro de modelo se escolhe medindo. O mais conhecido piorou o
resultado, e só a contagem mostrou isso.

---

## ⭐ 3. A unidade de medida invertia a comparação

**O que aconteceu.** Comparando a fala com o texto da etapa 1, o texto saía
65% positivo e só 4% neutro, e a fala, metade neutra. Parecia que quem
escreve issue é muito mais emotivo do que quem fala em podcast.

**Diagnóstico.** A etapa 1 deu uma nota para cada issue inteira, com mediana
de 311 palavras. O `compound` do VADER soma a nota de todas as palavras e
depois espreme a soma entre −1 e +1, então texto longo vai para perto dos
extremos. O trecho de fala tinha 11 palavras em média: a comparação media
coisas de tamanho diferente.

O mesmo efeito apareceu dentro da fala. O Whisper às vezes só fecha o trecho
no fim da janela de 30 s. No episódio do Y Combinator, entre os minutos 19 e
28, saíram trechos de 80 a 135 palavras, quase todos perto de +1, e a linha do
tempo mostrava um paredão verde. No total, 3% dos trechos tinham 40 palavras
ou mais, e 74% deles saíam positivos.

**Correção.** A frase vira a unidade nas duas mídias. O texto das issues é
cortado em frases pela pontuação, e cada trecho do Whisper também, com o
tempo de cada frase estimado pela posição dos caracteres
(`conteudo._frases`). As ferramentas passam a ser contadas por frase dos dois
lados.

**Resultado.** Texto: 34% positivo, 41% neutro, 25% negativo. Fala: 30%, 61%
e 9%. O paredão sumiu da linha do tempo, e os minutos 19 a 28 mostram frases
positivas e negativas misturadas.

**Lição.** Antes de comparar duas fontes, iguale a unidade. No VADER, o
tamanho do texto mexe na nota.

---

## ⭐ 4. O dicionário da etapa 1 não enxergava o Cursor

**O que aconteceu.** Na fala, o dicionário de ferramentas achou o Cursor em
1 frase, em 10 horas de podcast. Ouvindo os episódios, ele aparece mais de
cem vezes.

**Diagnóstico.** A etapa 1 só aceitava "cursor ai", "cursor editor" e
"cursor ide", porque "cursor" sozinho também é o cursor da tela. Só que quase
ninguém fala nem escreve "Cursor AI": diz "Cursor". A pista que separa o
produto da palavra comum muda com a mídia:

- na escrita, a maiúscula é escolha de quem escreve. Numa amostra de 40
  "Cursor" com maiúscula nas issues, os 40 eram o produto;
- na fala, quem escolhe a maiúscula é o Whisper, que escreve o produto em
  minúscula uma vez em cada três. Dos 38 "cursor" minúsculos da fala, 37 eram
  o produto.

**Correção.** Uma regra por mídia, em `config.py`:

```python
CURSOR_ESCRITA = r"\bCursor\b"  # issues: só com maiúscula
CURSOR_FALA = r"(?<!blinking )(?<!mouse )(?<!text )\bcursor\b(?! position)"  # fala: qualquer caixa
```

A regra da fala deixa de fora o cursor da tela dito por extenso ("mouse
cursor", "cursor position").

**Resultado.** Na fala, de 1 para 105 frases, e o Cursor sobe para segundo
lugar, com 26% das menções. No texto, de 98 para 450 frases.

**Lição.** Palavra ambígua pede regra por mídia. A maiúscula resolve a dúvida
no texto escrito por gente e não ajuda em nada na transcrição automática.

---

## 5. Podcast feito por IA, falando de IA

Um dos episódios da primeira amostra, do "Production Engineering with AI",
trazia dois "apresentadores" lendo o capítulo de um livro, com "Right." e
"Exactly." o tempo todo: o formato do NotebookLM, ferramenta do Google que
gera podcast a partir de um documento. O podcast publicou 60 episódios em 5
dias. Voz sintética lendo roteiro gerado fica fora da pergunta de pesquisa,
que é sobre desenvolvedores falando: o mesmo motivo do filtro de robôs da
etapa 1.

Buscar palavras como "AI-generated" ou "ElevenLabs" no feed não resolvia: os
podcasts falam desses assuntos, e quase todos os 14 feeds tinham alguma.
`coleta._voz_sintetica` procura a **declaração** do próprio podcast, uma frase
sobre a narração ou a voz dele ("AI-generated narration"). Quem declara sai, e
o próximo episódio do mesmo termo de busca entra no lugar. O "Production
Engineering" declara, e o Eye On A.I. entrou. Na troca, o episódio que saiu
ainda ficava no cache da acústica, e `sinal.analisar` passou a limpar do
cache quem sai da amostra.

Dois suspeitos sem declaração continuam: o Voxstar AI Automation, de uma
empresa de voz por IA, com 0,1 pausa longa por minuto (os outros têm de 1,0 a
3,4), e o "The AI & Tech Society by Danar". Tirar episódio por palpite seria
escolher a amostra pelo resultado.

## 6. Anúncio brasileiro dentro de podcast americano

Três episódios vieram de 1,7 a 2,5 minutos mais longos que a duração
anunciada pelo iTunes. O link do áudio passa por um servidor de anúncios, que
costura propaganda no arquivo na hora do download, escolhida pelo país de
quem baixa. Baixado do Brasil, o episódio do Nick Saraev veio com Tokio Marine
e Nubank no começo e Ágora no fim, falados em português. Com o idioma fixado
em inglês, o Whisper "traduziu" os anúncios ("Almir de Salvador, who had his
Mal extraviado..."). Os outros dois ganharam um anúncio da Squarespace.

Medido e mantido: em toda a fala, 36 frases têm cara de anúncio, nenhuma
cita ferramenta de IA, e tirá-las muda o percentual de frases positivas em
0,1 ponto. É por isso que a opção 2 mostra 9,9 horas anunciadas e o painel
mostra 10,0 horas medidas.

## 7. Transcrição: da CPU para a placa de vídeo

Na CPU, o Whisper rodava a 1,2 vez o tempo real, e as 10 horas levariam mais
de 8. Na placa de vídeo (RTX 3050), falhava com
`Library cublas64_12.dll is not found`. O motor do faster-whisper precisa das
bibliotecas CUDA da NVIDIA, que o pip instala (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`) numa pasta que o Windows não procura sozinho.
`transcricao._bibliotecas_cuda` põe essa pasta no caminho de busca com
`os.add_dll_directory`.

A falta da DLL só aparece na primeira transcrição: o modelo carrega sem erro.
Por isso o programa transcreve 1 segundo de silêncio logo ao carregar e, se
falhar, cai para a CPU ali mesmo, antes da demo. Na placa, de 15 a 21 vezes o
tempo real.

## 8. O librosa carregava o MP3 inteiro na memória

`librosa.load` num episódio de uma hora lê o MP3 inteiro em 44.100 Hz estéreo
antes de reamostrar para 16 kHz, e passaria de 1 GB de memória. O episódio
inteiro passou a ser decodificado pelo PyAV (`decode_audio`, a função que o
próprio Whisper usa), que converte para 16 kHz mono enquanto lê: cerca de
230 MB por hora de áudio. Os trechos curtos das demos continuam com o
librosa, que pula direto para o segundo pedido.

## 9. Download falhou no servidor de anúncios

Na primeira coleta, um episódio falhou com erro de DNS em
`traffic.megaphone.fm`, o servidor de anúncios por onde o link passa (o mesmo
da entrada 6). Na segunda tentativa baixou normalmente. `coleta.baixar` agora
tenta 3 vezes, com espera crescente, e grava num arquivo `.parcial` que só
ganha o nome final quando o download termina. Se a internet cair no meio, o
arquivo pela metade não passa por completo.

## 10. O lematizador partia "vibe coding" em dois

A nuvem de palavras mostrava "cod" e "datum". O modelo pequeno do spaCy
lematiza o verbo "coding" como "cod" e "data" como "datum" (o singular em
latim), e "vibe coding", o segundo bigrama mais falado, aparecia partido. A
etapa 1 documentou o erro do "cod" sem corrigir. Aqui, um dicionário de
correção (`config.CORRECAO_LEMA`) troca os dois lemas depois do spaCy, e
"vibe coding" aparece inteiro, com 82 ocorrências.

## 11. A linha do tempo escondia a maior parte das menções

O gráfico 06 desenhava só as 18 primeiras menções a ferramentas, para os
nomes não se encavalarem, e parecia dizer que o episódio só falava de
ferramenta nos primeiros 5 minutos. Agora toda menção vira uma linha (56 no
episódio), e o nome só é escrito quando está a mais de 1,5 minuto do último
nome escrito.

## 12. Limites documentados em vez de corrigidos

| Limite | Exemplo | Por que fica |
|---|---|---|
| O VADER não conhece o domínio | "security" vale +1,4 no léxico, e a frase mais positiva da amostra fala de "security campaigns" | A palavra é técnica nas duas mídias, então não desequilibra a comparação. Sem ela, o podcast do GitHub continua o mais positivo (+0,25) |
| A lista de vocabulário puxa para os nomes dela | "Claude Codex" aos 141 s do episódio do Y Combinator | A contagem continua certa (entrada 2) |
| O tempo da frase é estimado | A frase herda o tempo do trecho, dividido pela posição dos caracteres | O Whisper só marca o início e o fim do trecho. A curva de voz tem um valor por segundo, então o erro fica na mesma escala |
| Idioma fixado em inglês | Anúncio em português "traduzido" | 36 frases de anúncio em 10.691, sem efeito medido (entrada 6) |
