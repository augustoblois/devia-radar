# Etapa 3 · Áudio e vídeo (palestras)

Análise da imagem com PyAV e OpenCV, do som com librosa, leitura da tela com
OCR, transcrição com Whisper e as técnicas de PLN da etapa 1 aplicadas ao que
a palestra mostra, ao jeito como ela soa e ao que ela diz.
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
python sinal.py       # do som aos números no mesmo momento, com a opção de ouvir
python leitura.py     # o mesmo trecho lido pelo OCR, pelo Whisper e pela legenda oficial
python conteudo.py    # roda só a análise
```

### Antes da apresentação

O vídeo (715 MB), as telas recortadas, as legendas e as transcrições ficam
fora do Git, por tamanho. Na primeira execução, a opção 0 baixa as 8
palestras, recorta as telas, mede o som, lê e transcreve tudo. Depois disso o
programa roda sem internet, a partir do cache em `dados/`.

Os modelos:

- o Whisper usa o modelo `small` (cerca de 500 MB), que o faster-whisper
  baixa na primeira transcrição. Rode a opção 7 uma vez com internet, antes
  do dia;
- o RapidOCR traz as redes de leitura dentro do próprio pacote do pip;
- o detector de rosto YuNet (227 KB) vem no Git, em `modelos/`.

O som (opção 6) é medido na CPU: as 8 palestras levam cerca de 3 minutos
(172 s no computador de desenvolvimento), e o resultado fica em
`dados/processados/acustica.json`.

O painel final (gráfico 11) lê o corpus da etapa 1, em
`etapa1/dados/processados/corpus_limpo.json`, que também está no Git.

### Placa de vídeo (opcional)

A transcrição usa a placa NVIDIA quando ela existe:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Na placa, os 20 s da demonstração (opção 7) são transcritos em meio segundo.
Sem ela, o programa avisa e usa a CPU sozinho.

---

## Opções do programa

| Opção | O que faz |
|---|---|
| 1 | Lê as agendas XML da FOSDEM 2025 e 2026, escolhe as palestras e baixa o vídeo (AV1) e a legenda oficial (.vtt) |
| 2 | Resumo da amostra (palestras, horário na agenda, anos, licença, trilhas) |
| 3 | Demonstra um quadro da palestra de testes com Claude: recorte do slide e da câmera, diferença para o segundo anterior, brilho, bordas (Canny) e rosto (Haar x YuNet). Gera e abre o gráfico 02 |
| 4 | Percorre todas as palestras, um quadro por segundo: troca de slide, moldura, bordas, rosto e um quadro guardado por tela |
| 5 | Demonstra o som do mesmo momento: 20 s como lista de amostras, energia, pausas, espectrograma Mel, MFCC e altura da voz (F0). Gera e abre o gráfico 03 e pergunta se toca o trecho |
| 6 | Mede o som de todas as palestras, um minuto por vez: silêncio, pausas longas, energia, F0, centróide, ZCR e MFCC |
| 7 | Lê o mesmo trecho de 20 s de três jeitos: o slide pelo OCR, a fala pelo Whisper ao vivo e a legenda oficial. Toca o trecho |
| 8 | Lê todas as telas (OCR) e transcreve todas as falas (Whisper) |
| 9 | Analisa: tipo de tela, voz por tipo de tela, ferramentas na tela, na fala e na legenda, velocidade e sentimento da fala, voz x sentimento e o painel final |
| 10 | Gera os 11 gráficos, com o dashboard e o painel final |
| 11 | Busca por regex na fala e na tela, abre o quadro do momento e toca o áudio (KWIC com vídeo) |
| 0 | Roda o pipeline completo: coleta, imagem, som, leitura, análise e gráficos |
| r | Recoleta do zero, ignorando o cache |
| s | Sai do programa |

---

## Estrutura

```
etapa3/
├── config.py          parâmetros centrais da etapa
├── coleta.py          agenda XML da FOSDEM, regras da amostra e download de vídeo e legenda
├── quadros.py         PyAV e OpenCV: troca de slide, moldura, bordas, rosto e telas
├── sinal.py           librosa: energia, silêncio, pausas, espectrograma Mel, MFCC, ZCR, centróide e F0
├── leitura.py         RapidOCR (tela), Whisper (fala), legenda oficial e o tipo de cada tela
├── conteudo.py        tela, voz, fala e legenda cruzadas pelo tempo; regex, spaCy, VADER e Spearman
├── visualizacao.py    matplotlib: 11 gráficos
├── main.py            menu e orquestração do pipeline
├── diario-de-bordo.md erros da implementação e as correções
├── modelos/           detector de rosto YuNet (227 KB)
├── dados/
│   ├── brutos/        vídeo, legendas, transcrições, telas e trechos (fora do Git)
│   └── processados/   amostra, análise visual, som, leitura das telas e análise final
├── saida/             gráficos PNG prontos para os slides
└── apresentacao/      slides em PDF (o pptx e os roteiros de fala ficam só no computador da dupla)
```

## Pipeline

```
agenda XML da FOSDEM
        ↓  coleta.py          1.944 palestras com vídeo, 8 passam nas regras
PyAV + OpenCV (imagem)
        ↓  quadros.py         3,3 horas, um quadro por segundo: 320 telas
librosa (som)
        ↓  sinal.py           100 quadros de áudio por segundo: silêncio, pausas, energia e F0
RapidOCR + Whisper (small) + legenda oficial
        ↓  leitura.py         30.087 palavras escritas na tela, 26.499 faladas
regex + spaCy + VADER + Spearman
        ↓  conteudo.py        3.170 frases, com o tipo de tela e a voz do mesmo momento
matplotlib
        ↓  visualizacao.py    11 gráficos, com o dashboard e o painel final
```

---

## O que cada técnica faz aqui

### Agenda XML da FOSDEM (`coleta.py`)

A FOSDEM é a maior conferência europeia de software livre. Todo ano ela grava
quase mil palestras e publica a agenda num XML, um formato
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
O vídeo vem em AV1, um formato de compressão que deixa o arquivo de 6 a 9
vezes menor que o MP4, com a mesma resolução. A licença é Creative Commons
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

### librosa (`sinal.py`)

O arquivo de vídeo traz duas trilhas: a de imagem, que o `quadros.py`
analisa, e a de som. Para o computador, som também é uma lista de números: o
microfone mede a pressão do ar 16 mil vezes por segundo (a **taxa de
amostragem**), e cada medida é uma **amostra**, entre −1 e +1. Pelo teorema
de Nyquist, 16 kHz representa frequências até 8 kHz, onde está quase toda a
voz humana.

O som é cortado em **quadros de áudio** de 25 ms, um a cada 10 ms (o
**salto**): 100 quadros por segundo. Dentro de 25 ms a voz parece parada, e
dá para medir. Em cada quadro:

- **energia RMS** (raiz da média dos quadrados das amostras), em decibéis:
  0 dB é o máximo possível, e cada −6 dB é metade da amplitude;
- **silêncio**: quadro 35 dB abaixo do mais alto da palestra (o critério do
  `librosa.effects.split`) ou a menos de 10 dB do ruído de fundo da sala, o
  que marcar mais. **Pausa longa** é silêncio de 0,7 s ou mais entre duas
  falas;
- **espectrograma**: a FFT (transformada rápida de Fourier) decompõe o
  quadro nas frequências que o formam. Na **escala Mel**, que imita o
  ouvido, são 64 bandas;
- **MFCC**: 13 números por quadro que resumem o formato do espectro Mel, a
  "impressão digital" do timbre;
- **ZCR** (taxa de cruzamento do zero): quantas vezes o sinal troca de sinal.
  Vogal cruza pouco, "s" e "f" cruzam muito;
- **centróide espectral**: o centro de gravidade das frequências. Voz
  abafada tem centróide baixo;
- **F0** (frequência fundamental), a altura da voz, pelo algoritmo **YIN**,
  que procura de quanto em quanto tempo o sinal se repete (entre 65 e 400
  Hz). Só entra na conta o quadro **sonoro**: com som, cruzando o zero
  poucas vezes e longe dos limites da faixa.

A palestra é medida um minuto por vez, porque o YIN de uma vez só pediria
perto de 7 GB na palestra de 50 minutos (ver o diário). O que fica guardado é
um resumo por palestra, a lista de pausas e duas curvas com um valor por
segundo (energia e F0), que o `conteudo.py` cruza com a tela e com cada
frase.

### RapidOCR, Whisper e a legenda oficial (`leitura.py`)

- **OCR** (reconhecimento óptico de caracteres): o RapidOCR usa duas redes
  neurais. A primeira acha onde há texto na imagem e a segunda lê cada
  pedaço, com uma confiança de 0 a 1 (abaixo de 0,6, a linha é descartada).
- **Tipo de tela**, pelo texto lido e pelo brilho: menos de 10 palavras é
  slide **visual**. Com 50 linhas ou mais (30, se a tela for escura), é
  captura de ferramenta: **código** se a tela é escura ou se 15% das linhas
  têm cara de código, **interface** no resto. O que sobra é slide de **texto**.
- **Whisper**: modelo `small`, beam search de 5, filtro de voz e a lista de
  vocabulário com os nomes das ferramentas. O áudio sai do próprio arquivo de
  vídeo, o mesmo que o `sinal.py` mede.
- **Legenda oficial**: a FOSDEM publica um arquivo `.vtt` por palestra. Ela
  também sai de reconhecimento automático de fala, mas sem lista de
  vocabulário. Comparar as duas mede o efeito da lista em dados novos.

### Regex, spaCy, VADER e Spearman (`conteudo.py`)

As técnicas de texto da etapa 1, com as mesmas regras na fala das palestras e
no texto das issues, para as duas fontes serem comparáveis no painel final. O
módulo cruza tudo pelo relógio do vídeo: cada segundo recebe o tipo da tela
daquele momento e a voz medida pelo `sinal.py`, e cada frase falada recebe o
tipo de tela do seu segundo do meio. Por frase:

- regex com o dicionário de ferramentas da etapa 1. Na tela, o texto perde
  os espaços antes da comparação, porque o OCR cola palavras
  ("usingClaudeCode"). No texto escrito, o Cursor sozinho só conta com a
  maiúscula (ver o diário, entrada 10);
- VADER para o sentimento, sem a palavra "like" no léxico, em frases de 6
  palavras ou mais;
- velocidade da fala (palavras por minuto) e sentimento por tipo de tela;
- a voz da frase: energia e F0 dos segundos que ela ocupa, em **escore z**
  dentro da palestra, ou seja, quantos desvios-padrão acima da média dela.
  Cada sala tem um microfone e cada pessoa tem uma altura de voz, então a
  voz é comparada com o normal da própria palestra.

A **correlação de Spearman** (de −1 a +1, e perto de 0 quer dizer sem
relação) mede se a voz sobe junto com o sentimento da frase. O spaCy extrai os
lemas do que foi escrito na tela e do que foi dito, para comparar os dois
vocabulários (gráfico 09).

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
- **O som em números.** Silêncio de 20,5% a 40,6% do tempo (média de 30,2%)
  e de 3,0 a 9,6 pausas longas por minuto (média de 5,8). A palestra de
  Swift, programação ao vivo, é a que mais pausa. A F0 mediana fica entre 106
  e 149 Hz em sete palestras e em 226 Hz na de testes com Claude (ver o
  diário, entrada 14).
- **A voz quase não muda com a tela.** Em escore z dentro de cada palestra,
  a energia da voz fica entre −0,11 e +0,04 desvio-padrão com qualquer tipo de
  tela, e a altura, entre −0,04 e +0,21. O +0,21 do slide visual vem dos
  primeiros 93 s da palestra do Firefox: o projetor está apagado (lido como
  visual) e outra pessoa anuncia o palestrante, com F0 de 224 Hz contra 112
  Hz da palestra. Sem esses segundos, o slide visual fica em +0,06.
- **As pausas misturam a tela com quem fala.** Com código na tela, são 8,3
  pausas longas por minuto, contra 5,1 no slide de texto. Só que 12 dos 20
  minutos de código são da palestra de Swift, que pausa muito com qualquer
  tela. Dentro de cada palestra, a captura de ferramenta tem mais pausas que o
  slide de texto em duas (Self-Healing Rollouts: 6,6 no código e 8,9 na
  interface, contra 4,6; Drag, Drop, and Deploy: 4,8 contra 3,3) e menos em
  uma (Swift: 9,5 contra 10,8). Ver o diário, entrada 16.
- **A voz não acompanha o sentimento da frase.** Nas 2.379 frases avaliadas,
  a correlação de Spearman entre o compound e a energia da frase é +0,005 (p
  = 0,79), e com a F0, −0,008 (p = 0,70). Separando por classe, as frases
  positivas (+0,09) e as negativas (+0,12) saem um pouco mais altas que as
  neutras (+0,02), uma diferença de um décimo de desvio-padrão ou menos.
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
- **O escrito e o falado lado a lado**, frase a frase e com as mesmas regras
  (gráfico 11):

  | Fonte | Amostra | Frases avaliadas | Positivas | Neutras | Negativas | Mais citada |
  |---|---|---|---|---|---|---|
  | escrito · issues do GitHub (etapa 1) | 979 issues | 30.625 | 34% | 41% | 25% | ChatGPT (30%) |
  | falado · palestras da FOSDEM | 8 palestras, 3,3 h | 2.379 | 25% | 66% | 9% | ChatGPT (51%) |

  A reclamação mora no texto: 25% das frases das issues são negativas, contra
  9% na fala das palestras. Issue é o lugar de relatar defeito. O Cursor tem
  19% das menções a ferramentas nas issues e nenhuma na fala das palestras:
  ali, ele só aparece na tela.
- Velocidade média de 140 palavras por minuto, de 119 (Firefox) a 166
  (Talking to Robots).
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
| librosa | Tutorial oficial: <https://librosa.org/doc/latest/tutorial.html> |
| RapidOCR | <https://github.com/RapidAI/RapidOCR> |
| faster-whisper | <https://github.com/SYSTRAN/faster-whisper> |
| Agenda da FOSDEM | <https://fosdem.org/2026/schedule/> |
| spaCy | Curso oficial gratuito: <https://course.spacy.io> |
| NLTK / VADER | <https://www.nltk.org/howto/sentiment.html> |
| SciPy (Spearman) | <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html> |
| matplotlib | <https://matplotlib.org/stable/gallery/index.html> |
