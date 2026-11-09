# Etapa 3 · Vídeo (palestras)

Análise dos quadros com PyAV e OpenCV, leitura da tela com OCR, transcrição
com Whisper e as técnicas de PLN das etapas 1 e 2 aplicadas ao que a palestra
mostra e ao que ela diz.
Fonte: palestras gravadas da FOSDEM 2025 e 2026 sobre IA no desenvolvimento de
software.

Slides: [`apresentacao/etapa3-devia-radar.pdf`](apresentacao/etapa3-devia-radar.pdf) ·
Diário de bordo: [`diario-de-bordo.md`](diario-de-bordo.md)

---

## Execução

```powershell
cd etapa3
python main.py
```

Abre o menu interativo, que é o modo usado na apresentação.
Para rodar tudo de uma vez, sem interação:

```powershell
python main.py --tudo
```

Cada módulo também roda sozinho, útil para demonstrar uma parte isolada:

```powershell
python quadros.py     # do vídeo aos números num quadro da palestra de demonstração
python leitura.py     # o mesmo trecho lido pelo OCR, pelo Whisper e pela legenda oficial
python conteudo.py    # roda só a análise
```

### Antes da apresentação

O vídeo (715 MB), as telas recortadas, as legendas e as transcrições ficam
fora do Git, por tamanho. Na primeira execução, a opção 0 baixa as 8
palestras, recorta as telas, lê e transcreve tudo. Depois disso o programa
roda sem internet, a partir do cache em `dados/`.

Os modelos:

- o Whisper usa o mesmo modelo `small` da etapa 2 (cerca de 500 MB). Se a
  etapa 2 já rodou neste computador, ele já está guardado. Se não, rode a
  opção 5 uma vez com internet, antes do dia;
- o RapidOCR traz as redes de leitura dentro do próprio pacote do pip;
- o detector de rosto YuNet (227 KB) vem no Git, em `modelos/`.

O painel das três etapas (gráfico 09) lê o resultado da etapa 2, em
`etapa2/dados/processados/analise_fala.json`, que também está no Git.

### Placa de vídeo (opcional)

A transcrição usa a placa NVIDIA quando ela existe, com as mesmas bibliotecas
da etapa 2:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Na placa, os 20 s da demonstração (opção 5) são transcritos em meio segundo.
Sem ela, o programa avisa e usa a CPU sozinho. Pela velocidade medida na
etapa 2, as 3,3 horas de fala levariam quase 3 horas.

---

## Opções do programa

| Opção | O que faz |
|---|---|
| 1 | Lê as agendas XML da FOSDEM 2025 e 2026, escolhe as palestras e baixa o vídeo (AV1) e a legenda oficial (.vtt) |
| 2 | Resumo da amostra (palestras, horário na agenda, anos, licença, trilhas) |
| 3 | Demonstra um quadro da palestra de testes com Claude: recorte do slide e da câmera, diferença para o segundo anterior, brilho, bordas (Canny) e rosto (Haar x YuNet). Gera e abre o gráfico 02 |
| 4 | Percorre todas as palestras, um quadro por segundo: troca de slide, moldura, bordas, rosto e um quadro guardado por tela |
| 5 | Lê o mesmo trecho de 20 s de três jeitos: o slide pelo OCR, a fala pelo Whisper ao vivo e a legenda oficial. Toca o trecho |
| 6 | Lê todas as telas (OCR) e transcreve todas as falas (Whisper) |
| 7 | Analisa: tipo de tela, ferramentas na tela, na fala e na legenda, velocidade e sentimento da fala por tipo de tela, e as três etapas lado a lado |
| 8 | Gera os 9 gráficos, com o dashboard e o painel das três etapas |
| 9 | Busca por regex na fala e na tela, abre o quadro do momento e toca o áudio (KWIC com vídeo) |
| 0 | Roda o pipeline completo |
| r | Recoleta do zero, ignorando o cache |

---

## Estrutura

```
etapa3/
├── config.py          parâmetros centrais da etapa
├── coleta.py          agenda XML da FOSDEM, regras da amostra e download de vídeo e legenda
├── quadros.py         PyAV e OpenCV: troca de slide, moldura, bordas, rosto e telas
├── leitura.py         RapidOCR (tela), Whisper (fala), legenda oficial e o tipo de cada tela
├── conteudo.py        tela, fala e legenda cruzadas pelo tempo; regex, spaCy e VADER
├── visualizacao.py    matplotlib: 9 gráficos
├── main.py            menu e orquestração do pipeline
├── diario-de-bordo.md erros da implementação e as correções
├── modelos/           detector de rosto YuNet (227 KB)
├── dados/
│   ├── brutos/        vídeo, legendas, transcrições, telas e trechos (fora do Git)
│   └── processados/   amostra, análise visual, leitura das telas e análise final
├── saida/             gráficos PNG prontos para os slides
└── apresentacao/      slides em PDF (o pptx e os roteiros de fala ficam só no computador da dupla)
```

## Pipeline

```
agenda XML da FOSDEM
        ↓  coleta.py          1.944 palestras com vídeo, 8 passam nas regras
PyAV + OpenCV
        ↓  quadros.py         3,3 horas, um quadro por segundo: 320 telas
RapidOCR + Whisper (small) + legenda oficial
        ↓  leitura.py         30.087 palavras escritas na tela, 26.499 faladas
regex + spaCy + VADER
        ↓  conteudo.py        3.170 frases, cada uma com o tipo de tela do mesmo segundo
matplotlib
        ↓  visualizacao.py    9 gráficos, com o dashboard e o painel das três etapas
```

---

## O que cada técnica faz aqui

### Agenda XML da FOSDEM (`coleta.py`)

A FOSDEM é a maior conferência europeia de software livre. Todo ano ela grava
mais de mil palestras e publica a agenda num XML, um formato
**semiestruturado**: as marcações (`<event>`, `<title>`, `<abstract>`)
organizam o arquivo, mas o conteúdo de dentro é texto livre. O
`xml.etree.ElementTree`, que vem com o Python, lê as marcações, e a regex lê
o texto de dentro, como na etapa 1.

As regras da amostra, em ordem:

1. termo de IA no **título**: 27 das 1.944 palestras com vídeo;
2. termo de desenvolvimento de software também no título, e nenhum termo de
   IA como produto (treino, dataset, inferência, servidor de modelo): sobram 8;
3. entre 15 e 50 minutos: as 8 passam.

A amostra é, portanto, a lista inteira das palestras que passam nas regras.
O vídeo vem em AV1, um formato de compressão que deixa o arquivo 10 vezes
menor que o MP4, com a mesma resolução. A licença é Creative Commons
Attribution: pode analisar e mostrar quadros, citando a fonte.

### PyAV e OpenCV (`quadros.py`)

Para o computador, vídeo é uma pilha de imagens (os **quadros**), e cada
quadro é uma matriz de pixels. O PyAV, o FFmpeg dentro do Python, decodifica
o vídeo, e o programa olha um quadro por segundo. A FOSDEM monta toda gravação
com o mesmo desenho, então cada quadro é recortado em duas **regiões de
interesse**: o slide, no canto superior esquerdo, e a câmera do palestrante,
no canto inferior direito. Em cada segundo:

- **diferença entre quadros**: quanto o slide mudou desde o segundo anterior,
  na média dos pixels (de 0 a 255). Acima de 6, o slide trocou. Uma máscara
  apaga o canto em que a câmera fica por cima do slide;
- **moldura**: a parte fixa da tela (logotipo e faixa da FOSDEM), comparada
  com a moldura típica da palestra. Segundo que foge dela é vinheta,
  patrocinador ou slide em tela cheia, e sai da análise;
- **brilho** médio do slide: abaixo de 90, a tela é escura (editor ou terminal);
- **bordas** pelo detector de Canny (limiares 100 e 200): a fração de pixels
  em que o brilho muda de repente, como no contorno de uma letra;
- **rosto** pelo YuNet, uma rede neural de 227 KB que roda pelo próprio
  OpenCV (confiança mínima de 0,6).

Cada trecho em que o slide fica parado 3 s ou mais vira uma **tela**, e um
quadro dela é guardado para o OCR. Trecho mais curto é animação, rolagem ou
vídeo passando, e conta como tela em **movimento**.

### RapidOCR, Whisper e a legenda oficial (`leitura.py`)

- **OCR** (reconhecimento óptico de caracteres): o RapidOCR usa duas redes
  neurais. A primeira acha onde há texto na imagem e a segunda lê cada
  pedaço, com uma confiança de 0 a 1 (abaixo de 0,6, a linha é descartada).
- **Tipo de tela**, pelo texto lido e pelo brilho: menos de 10 palavras é
  slide **visual**. Com 50 linhas ou mais (30, se a tela for escura), é
  captura de ferramenta: **código** se a tela é escura ou se 15% das linhas
  têm cara de código, **interface** no resto. O que sobra é slide de **texto**.
- **Whisper**: os mesmos parâmetros da etapa 2 (modelo `small`, beam search
  de 5, filtro de voz e a lista de vocabulário com os nomes das ferramentas).
  O áudio sai do próprio arquivo de vídeo.
- **Legenda oficial**: a FOSDEM publica um arquivo `.vtt` por palestra. Ela
  também sai de reconhecimento automático de fala, mas sem lista de
  vocabulário. Comparar as duas mede o efeito da lista em dados novos.

### Regex, spaCy e VADER (`conteudo.py`)

As técnicas das etapas 1 e 2, com as mesmas regras, para as três etapas serem
comparáveis. O módulo cruza tudo pelo relógio do vídeo: cada segundo recebe o
tipo da tela daquele momento, e cada frase falada recebe o tipo de tela do seu
segundo do meio. Por frase:

- regex com o dicionário de ferramentas das etapas 1 e 2. Na tela, o texto
  perde os espaços antes da comparação, porque o OCR cola palavras
  ("usingClaudeCode");
- VADER para o sentimento, sem a palavra "like" no léxico, em frases de 6
  palavras ou mais;
- velocidade da fala (palavras por minuto) e sentimento por tipo de tela.

O spaCy extrai os lemas do que foi escrito na tela e do que foi dito, para
comparar os dois vocabulários (gráfico 07).

---

## Principais resultados

Amostra de **8 palestras** da FOSDEM 2025 e 2026: 3,3 horas de vídeo, 320
telas, 30.087 palavras escritas na tela e 26.499 faladas.

- **A tela é quase toda slide.** Em 68% do tempo, a tela mostra slide de
  texto e, em 12%, slide visual. Captura de ferramenta ocupa 18% (10% de
  código e 8% de interface), e esse número é um piso (ver o diário). Três
  palestras concentram as capturas: a programação ao vivo em Swift (66% do
  tempo em código), a de Drag, Drop, and Deploy (48% em interface) e a de
  Self-Healing Rollouts (46% em ferramenta).
- **A velocidade da fala quase não muda com a tela**: 138 palavras por minuto
  no slide de texto, 142 no código e 136 na interface. Só a tela em movimento
  derruba a fala, para 85 palavras por minuto (47 frases, 2% do tempo).
- **O tom acompanha quem fala, mais do que a tela.** Diante de foto, meme ou
  diagrama, 33% das frases são positivas e 17% negativas; no slide de texto,
  25% e 10%. Só que 141 das 193 frases ditas com slide visual são da palestra
  do Rspamd, que é mais emotiva com qualquer tela: no slide de texto, ela tem
  35% de frases positivas, e as outras sete palestras, 24%. Sem o Rspamd,
  sobram 52 frases com slide visual, com menos negativas que o slide de texto
  (4% contra 9%). Ver o diário.
- **A lista de vocabulário ganha da legenda oficial.** 44 nomes de ferramenta
  aparecem só na nossa transcrição, e nenhum só na legenda. Na conferência
  manual, 33 são erros da legenda ("cloth" no lugar de Claude, 11 vezes numa
  palestra só), 6 são erros nossos causados pela própria lista ("OpenAPI"
  virou "OpenAI"; "CrowdStrike" virou "Claude striked"), 3 são diferença de
  grafia e 2 ficam em dúvida.
- **No vídeo, as ferramentas refletem o tema de cada palestra.** O ChatGPT tem
  51% das frases que citam ferramenta, mas 34 das 38 vêm da palestra que
  constrói um proxy do ChatGPT. Das 25 do Claude, 19 vêm da palestra de testes
  com Claude. Na tela, 31 das 34 telas com "Gemini" mostram o botão que o
  Chrome põe na janela.
- **As três etapas lado a lado**, frase a frase e com as mesmas regras:

  | Etapa | Amostra | Frases avaliadas | Positivas | Neutras | Negativas | Mais citada |
  |---|---|---|---|---|---|---|
  | 1 · texto | 979 issues do GitHub | 30.625 | 34% | 41% | 25% | ChatGPT (30%) |
  | 2 · fala | 14 episódios de podcast, 10 h | 8.253 | 30% | 61% | 9% | Claude (41%) |
  | 3 · vídeo | 8 palestras, 3,3 h | 2.379 | 25% | 66% | 9% | ChatGPT (51%) |

  A reclamação mora no texto: 25% das frases das issues são negativas, contra
  9% na fala dos podcasts e das palestras. Issue é o lugar de relatar defeito.
- Velocidade média de 140 palavras por minuto, de 119 (Firefox) a 166
  (Talking to Robots). Nos podcasts eram 180.
- O YuNet acha o rosto do palestrante em 86% do tempo, na média das
  palestras. A exceção é a do Firefox (1%), em que a câmera filma a sala de
  longe.

Os números saem do estado atual do cache em `dados/`. A agenda das duas
edições já está fechada, então recoletar com a opção `r` deve refazer a mesma
amostra.

---

## Onde aprender as bibliotecas usadas

| Biblioteca | Material |
|---|---|
| OpenCV | Tutoriais oficiais em Python: <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html> |
| PyAV | <https://github.com/PyAV-Org/PyAV> |
| YuNet | Modelo e exemplo de uso: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> |
| RapidOCR | <https://github.com/RapidAI/RapidOCR> |
| faster-whisper | <https://github.com/SYSTRAN/faster-whisper> |
| Agenda da FOSDEM | <https://fosdem.org/2026/schedule/> |
| spaCy | Curso oficial gratuito: <https://course.spacy.io> |
| NLTK / VADER | <https://www.nltk.org/howto/sentiment.html> |
| matplotlib | <https://matplotlib.org/stable/gallery/index.html> |
