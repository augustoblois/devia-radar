# Etapa 2 · Áudio (podcasts)

Análise do sinal com librosa, transcrição com Whisper e as técnicas de PLN da
etapa 1 aplicadas à fala transcrita.
Fonte: episódios de podcast de 2025 e 2026 em que desenvolvedores falam de
ferramentas de IA.

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
python sinal.py         # do som aos números num trecho de 20 segundos
python transcricao.py   # transcreve o mesmo trecho sem e com a lista de vocabulário
python conteudo.py      # roda só a análise da fala
```

### Antes da apresentação

O áudio (670 MB) e as transcrições ficam fora do Git, por tamanho e por
direito autoral. Na primeira execução, a opção 0 baixa os 14 episódios e
transcreve tudo. Depois disso o programa roda sem internet, tudo a partir do
cache em `dados/`.

O Whisper baixa o modelo `small` (cerca de 500 MB) na primeira transcrição.
Rode a opção 5 uma vez com internet, antes do dia, para o modelo ficar
guardado no computador.

### Placa de vídeo (opcional)

Com placa NVIDIA, a transcrição roda de 15 a 21 vezes mais rápido que o tempo
real. Na CPU, 1,2 vez: as 10 horas de áudio levam mais de 8 horas. Para usar
a placa no Windows, instale as bibliotecas CUDA pelo pip:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Sem elas, o programa avisa e usa a CPU sozinho.

---

## Opções do programa

| Opção | O que faz |
|---|---|
| 1 | Coleta os episódios pela iTunes Search API e baixa o áudio |
| 2 | Resumo da amostra (episódios, podcasts, duração, período) |
| 3 | Demonstra o sinal de um trecho de 20 s (onda, energia, espectrograma Mel, MFCC, altura da voz) e toca o trecho |
| 4 | Extrai as características acústicas de todos os episódios (librosa) |
| 5 | Transcreve o trecho ao vivo, sem e com a lista de vocabulário (Whisper) |
| 6 | Transcreve todos os episódios (Whisper) |
| 7 | Analisa a fala: ferramentas citadas, sentimento e voz |
| 8 | Gera os 8 gráficos e o dashboard |
| 9 | Busca por regex na fala e toca o momento encontrado (KWIC com áudio) |
| 0 | Roda o pipeline completo |
| r | Recoleta do zero, ignorando o cache |

---

## Estrutura

```
etapa2/
├── config.py          parâmetros centrais da etapa
├── coleta.py          iTunes Search API, filtros da amostra e download do áudio
├── sinal.py           librosa: energia, pausas, espectrograma, MFCC, altura da voz
├── transcricao.py     Whisper (faster-whisper): a fala vira texto com marca de tempo
├── conteudo.py        regex, spaCy e VADER na fala, e o cruzamento com a voz
├── visualizacao.py    matplotlib e wordcloud: 8 gráficos
├── main.py            menu e orquestração do pipeline
├── diario-de-bordo.md erros da implementação e as correções
├── dados/
│   ├── brutos/        áudio, transcrições e trechos .wav (fora do Git)
│   └── processados/   amostra, acústica e análise da fala
├── saida/             gráficos PNG prontos para os slides
└── apresentacao/      slides em PDF (o pptx e os roteiros de fala ficam só no computador da dupla)
```

## Pipeline

```
iTunes Search API
        ↓  coleta.py          7 termos de busca, 14 episódios de 14 podcasts
librosa
        ↓  sinal.py           10 horas de áudio, 100 quadros por segundo
Whisper (small)
        ↓  transcricao.py     104.503 palavras em 9.784 trechos com marca de tempo
regex + spaCy + VADER
        ↓  conteudo.py        10.691 frases: ferramentas, sentimento e voz
matplotlib + wordcloud
        ↓  visualizacao.py    8 gráficos, incluindo o dashboard consolidado
```

---

## O que cada técnica faz aqui

### iTunes Search API (`coleta.py`)

A busca pública de podcasts da Apple. Cada termo (`github copilot`,
`claude code`, `cursor ai`, `vibe coding`, `ai code review`,
`ai pair programming`, `ai generated code`) devolve até 200 episódios, e os
filtros deixam só o que serve para a pergunta: episódio de 2025 ou 2026,
entre 20 e 60 minutos, em inglês, falando de IA e de software ao mesmo tempo,
no máximo um episódio por podcast e sem voz gerada por IA declarada no feed.
Ficam os 2 primeiros de cada termo, na ordem de relevância da própria API.

### librosa (`sinal.py`)

Para o computador, som é uma lista de números: aqui, 16.000 por segundo. Em
quadros de 25 ms, um a cada 10 ms, o módulo mede:

- **energia** (RMS em decibéis) e **pausas longas** (silêncio de 0,7 s ou
  mais, 35 dB abaixo do pico do episódio);
- **espectrograma Mel** (64 bandas) e **MFCC** (13 coeficientes), o desenho do
  timbre da voz;
- **altura da voz (F0)**, pelo algoritmo YIN, entre 65 e 400 Hz;
- taxa de cruzamento por zero e centróide espectral.

Cada episódio vira um resumo (pausas por minuto, variação da altura da voz) e
uma curva com um valor por segundo, usada depois para cruzar a voz com o texto.

### Whisper (`transcricao.py`)

Modelo de reconhecimento de fala. Corta o áudio em janelas de 30 s,
transforma cada janela num espectrograma Mel e escreve o texto com a marca de
tempo de cada trecho. Parâmetros: modelo `small`, beam search de 5 (guarda as
5 melhores frases parciais em vez de só a mais provável), filtro de voz (VAD)
para não inventar texto no silêncio e uma lista de vocabulário (`hotwords`)
com os nomes das ferramentas.

### Regex, spaCy e VADER (`conteudo.py`)

As técnicas da etapa 1, agora na fala. Cada trecho do Whisper é cortado em
frases, com o tempo de cada frase estimado pela posição dos caracteres, e o
texto das issues da etapa 1 é medido na mesma unidade. Por frase:

- regex com o dicionário de ferramentas da etapa 1, mais a regra do Cursor
  (ver o diário);
- VADER para o sentimento, sem a palavra "like" no léxico (ver o diário);
- energia e altura da voz nos segundos que a frase ocupa, em **escore z**
  (quantos desvios-padrão acima da média do próprio episódio), e a
  **correlação de Spearman** entre a voz e o sentimento.

O spaCy extrai os lemas e os bigramas da fala inteira, com o filtro da etapa 1.

---

## Principais resultados

Amostra de **14 episódios** de **14 podcasts**, publicados entre setembro de
2025 e novembro de 2026: 10 horas de áudio e 104.503 palavras transcritas.

- **O Claude domina a fala.** 41% das menções a ferramentas nos podcasts são
  do Claude (167 frases, em 12 dos 14 episódios). No texto das issues ele tem
  28%, e quem lidera é o ChatGPT, com 30%. O Cursor fica em segundo na fala,
  com 26%.
- **A fala é mais neutra e quase não reclama.** Frase a frase, a fala tem 30%
  de frases positivas, 61% neutras e 9% negativas. O texto das issues tem
  34%, 41% e 25%. Issue é o lugar de relatar defeito; o podcast é conversa.
- **A unidade de medida muda a conclusão.** Medindo a issue inteira, como a
  etapa 1 fez, o texto sai 65% positivo e só 4% neutro: texto longo empurra o
  VADER para os extremos (ver o diário).
- **A voz não acompanha o sentimento.** Entre o sentimento da frase e a
  energia da voz, ρ = −0,025 (p = 0,02, 8.253 frases): passa no teste por
  causa do tamanho da amostra, mas explica menos de 0,1% da variação. Com a
  altura da voz, ρ = +0,003 (p = 0,78). As frases negativas não são ditas mais
  alto nem mais agudo que as outras.
- **Quem fala muda o tom.** O podcast oficial do GitHub é o mais positivo da
  amostra (média +0,28). O debate "Vibe Coding vs Real Engineering" é o mais
  neutro (+0,01).
- Bigramas mais falados: `claude code` (86), `vibe coding` (82),
  `pair programming` (69), `write code` (51), `code review` (50).
- Velocidade média de 180 palavras por minuto, de 128 (Voxstar) a 220
  (Y Combinator).

Os números saem do estado atual do cache em `dados/`. Recoletar com a opção
`r` muda a amostra, porque a busca da Apple devolve episódios novos todo dia.

---

## Onde aprender as bibliotecas usadas

| Biblioteca | Material |
|---|---|
| librosa | Tutorial oficial: <https://librosa.org/doc/latest/tutorial.html> |
| Whisper | Artigo e cartão do modelo: <https://github.com/openai/whisper> |
| faster-whisper | <https://github.com/SYSTRAN/faster-whisper> |
| iTunes Search API | <https://performance-partners.apple.com/search-api> |
| spaCy | Curso oficial gratuito: <https://course.spacy.io> |
| NLTK / VADER | <https://www.nltk.org/howto/sentiment.html> |
| matplotlib | <https://matplotlib.org/stable/gallery/index.html> |
