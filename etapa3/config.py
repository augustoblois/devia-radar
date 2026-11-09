"""
DevIA Radar - Configurações centrais da Etapa 3 (vídeo).

Etapa 3 da disciplina Dados Não Estruturados (UFPB).
O tema continua o das etapas 1 e 2: como desenvolvedores discutem o uso de
Inteligência Artificial no desenvolvimento de software. Agora a fonte são
palestras gravadas em vídeo, com slide, código na tela e fala.

Todo parâmetro que a gente pode querer ajustar durante a apresentação
mora aqui, para não precisar caçar valor mágico no meio da lógica.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
# Mesma correção das etapas 1 e 2 (UnicodeEncodeError no console do Windows).
for _saida in (sys.stdout, sys.stderr):
    try:
        _saida.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).parent
DIR_BRUTOS = RAIZ / "dados" / "brutos"         # fora do git: vídeo, legenda, transcrição
DIR_VIDEOS = DIR_BRUTOS / "videos"
DIR_LEGENDAS = DIR_BRUTOS / "legendas"
DIR_TRANSCRICOES = DIR_BRUTOS / "transcricoes"
DIR_TELAS = DIR_BRUTOS / "telas"               # um quadro-chave por slide (JPG)
DIR_TRECHOS = DIR_BRUTOS / "trechos"           # áudio e quadro tocados ou abertos nas demos
DIR_PROCESSADOS = RAIZ / "dados" / "processados"
DIR_SAIDA = RAIZ / "saida"

ARQ_PALESTRAS = DIR_PROCESSADOS / "palestras.json"
ARQ_VISUAL = DIR_PROCESSADOS / "analise_visual.json"
ARQ_LEITURA = DIR_PROCESSADOS / "leitura_tela.json"
ARQ_VIDEO = DIR_PROCESSADOS / "analise_video.json"

# Resultado da etapa 2, lido para o painel final das três etapas. Ele já
# traz o texto da etapa 1 recontado frase a frase (ver conteudo._etapa1).
ARQ_ETAPA2_FALA = RAIZ.parent / "etapa2" / "dados" / "processados" / "analise_fala.json"

for _pasta in (DIR_VIDEOS, DIR_LEGENDAS, DIR_TRANSCRICOES, DIR_TELAS, DIR_TRECHOS, DIR_PROCESSADOS, DIR_SAIDA):
    _pasta.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Coleta (agenda pública da FOSDEM em XML)
# ---------------------------------------------------------------------------
# FOSDEM é a maior conferência europeia de software livre. Toda palestra é
# gravada e publicada com licença Creative Commons Attribution, e a agenda
# completa sai num XML com título, resumo, trilha e os links dos vídeos.
AGENDAS = {
    2025: "https://fosdem.org/2025/schedule/xml",
    2026: "https://fosdem.org/2026/schedule/xml",
}
LICENCA = "CC BY 2.0 BE (vídeos da FOSDEM)"

N_PALESTRAS = 8
DURACAO_MIN = 15  # minutos: palestra relâmpago de 5 min não tem conteúdo
DURACAO_MAX = 50  # minutos: limita o tempo de processamento

# Formato baixado: AV1 em WebM. O mesmo vídeo em MP4 tem 9x o tamanho
# (1,1 GB contra 124 MB numa palestra de 50 min), com a mesma resolução.
SUFIXO_VIDEO = ".av1.webm"
SUFIXO_LEGENDA = ".vtt"

# ---------------------------------------------------------------------------
# Quadros (PyAV decodifica, OpenCV analisa)
# ---------------------------------------------------------------------------
AMOSTRAGEM = 1.0      # segundos entre dois quadros analisados (1 por segundo)

# A FOSDEM grava todas as salas com o mesmo layout: o slide no canto superior
# esquerdo e a câmera do palestrante num quadro menor no canto inferior
# direito. Cada região é (x0, y0, x1, y1) em FRAÇÃO da largura e da altura,
# então vale para qualquer resolução.
REGIAO_SLIDE = (0.008, 0.020, 0.812, 0.818)
REGIAO_CAMERA = (0.740, 0.730, 0.993, 0.990)

# Tamanho reduzido em que as diferenças entre quadros são medidas. Reduzir
# apaga o ruído de compressão do vídeo e deixa o cálculo instantâneo.
TAMANHO_SLIDE = (192, 108)
TAMANHO_CAMERA = (96, 54)

# Diferença média de brilho (0 a 255) entre dois quadros seguidos da região
# do slide a partir da qual consideramos que o slide mudou.
LIMIAR_TROCA = 6.0
TELA_MINIMA = 3       # segundos: troca mais rápida que isso conta como animação do mesmo slide

# Moldura: a parte fixa da tela, fora do slide e da câmera (logotipo e
# faixa da FOSDEM). Quando ela some, o quadro é vinheta de abertura, tela de
# patrocinador ou slide em tela cheia, e sai da análise.
TAMANHO_FUNDO = (96, 54)
LIMIAR_MOLDURA = 25.0  # diferença média para a moldura típica da palestra

# Bordas pelo detector de Canny, medidas no slide reduzido a 640 x 360.
TAMANHO_BORDAS = (640, 360)
CANNY_BAIXO = 100     # gradiente abaixo disso nunca é borda
CANNY_ALTO = 200      # gradiente acima disso sempre é borda (no meio, só se encostar numa)
LIMIAR_ESCURO = 90    # brilho médio do slide abaixo disso: tela escura (terminal, editor)

# Detector de rosto YuNet, uma rede neural pequena que roda pelo OpenCV.
# O Haar cascade, que vem dentro do OpenCV, achou rosto em 20 de 96 quadros
# de teste; o YuNet achou em 79 (ver diário de bordo).
MODELO_ROSTO = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
CONFIANCA_ROSTO = 0.6  # caixa com confiança menor que isso é descartada

# Quadro usado nas demonstrações (opções 3 e 5 do menu). De 0:16 a 0:36, o
# slide mostra "Claude", a nossa transcrição escreve "Claude" duas vezes e a
# legenda oficial escreve "cloth" nas duas.
DEMO_PALESTRA = "2026-8536"  # Testing on hardware with Claude AI
DEMO_SEGUNDO = 16

# ---------------------------------------------------------------------------
# Leitura da tela (OCR) e da fala (Whisper)
# ---------------------------------------------------------------------------
CONFIANCA_OCR = 0.6     # linha lida com confiança menor que isso é descartada

# Tipo de tela, decidido pelo texto que o OCR leu e pelo brilho (ver
# leitura.tipo_de_tela). Limiares escolhidos em 48 telas rotuladas à mão e
# conferidos em outras 32 sorteadas: 29 acertos. Os 3 erros foram capturas
# curtas lidas como slide, então o tempo de captura medido é um piso.
MIN_PALAVRAS_TELA = 10      # menos que isso: tela "visual" (foto, diagrama, meme)
LINHAS_CAPTURA = 50         # a partir disso: captura de ferramenta. Nenhum slide
                            # rotulado passou de 44 linhas de texto
LINHAS_CAPTURA_ESCURA = 30  # editor ou terminal em tema escuro, de letra maior
FRACAO_CODIGO = 0.15        # numa captura clara: a partir disso das linhas com
                            # cara de código, "código"; abaixo, "interface"

# Whisper: mesmos parâmetros da etapa 2, inclusive GPU quando existe.
TAXA_AMOSTRAGEM = 16_000
MODELO_WHISPER = "small"
FEIXE = 5
DISPOSITIVO = "auto"              # "auto", "cuda" ou "cpu"
TIPO_COMPUTO_GPU = "int8_float16"
TIPO_COMPUTO_CPU = "int8"
THREADS = max((os.cpu_count() or 4) // 2, 1)
TRECHO_DEMO = 20                  # segundos de fala transcritos ao vivo na opção 5
# Mesma lista de vocabulário da etapa 2, passada pelo parâmetro hotwords.
PROMPT_VOCABULARIO = (
    "A technical talk by a software developer about AI coding tools: "
    "Claude, Claude Code, Anthropic, ChatGPT, OpenAI, Codex, GitHub Copilot, "
    "Cursor, Gemini, Llama, Ollama, LLM, MCP, open source, code review, pull request."
)

# ---------------------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------------------
# O MESMO dicionário das etapas 1 e 2, para as três etapas serem comparáveis.
FERRAMENTAS_IA = {
    "GitHub Copilot": ["github copilot", "copilot"],
    "ChatGPT": ["chatgpt", "chat gpt", "gpt-4", "gpt-3", "gpt4", "openai"],
    "Claude": ["claude", "anthropic"],
    "Cursor": ["cursor ai", "cursor editor", "cursor ide"],
    "Gemini": ["gemini", "bard"],
    "Tabnine": ["tabnine", "tab nine"],
    "Codeium": ["codeium", "windsurf"],
    "Ollama": ["ollama", "llama"],
    "Sourcegraph Cody": ["cody", "sourcegraph"],
    "Amazon Q": ["amazon q", "codewhisperer"],
}

# "Cursor" sozinho é o produto e também palavra comum (o cursor da tela, o
# cursor de paginação). Por isso a etapa 1 só contava "cursor ai", "cursor
# editor" e "cursor ide", e perdia quase todas as menções (ver o diário de
# bordo da etapa 2). A pista que separa os dois usos depende da mídia:
# - ESCRITA (issue, slide): a maiúscula é escolha do autor. "Cursor" é o
#   produto; "cursor" minúsculo quase nunca é.
# - FALA (Whisper, legenda automática): a maiúscula é escolha do modelo, que
#   escreve o produto em minúscula uma vez em cada três. Ali vale qualquer
#   caixa, menos o cursor da tela dito por extenso.
CURSOR_ESCRITA = r"\bCursor\b"
CURSOR_FALA = r"(?<!blinking )(?<!mouse )(?<!text )\bcursor\b(?! position)"

LIMITE_POSITIVO = 0.05   # mesmas faixas do VADER das etapas 1 e 2
LIMITE_NEGATIVO = -0.05
MIN_PALAVRAS_FRASE = 6   # frase curta demais ("Yeah, right.") não sustenta sentimento

# Palavras que saem do léxico do VADER, como na etapa 2. O VADER dá +1,5 a
# "like", mas na fala quase todo "like" é muleta ("it's like, you know") ou
# comparação ("like this"), e deixava a fala mais positiva do que ela é
# (ver o diário de bordo da etapa 2).
SEM_SENTIMENTO = {"like"}

MODELO_SPACY = "en_core_web_sm"
POS_RELEVANTES = {"NOUN", "PROPN", "ADJ", "VERB"}
# Erros do lematizador do spaCy pequeno, os mesmos da etapa 2.
CORRECAO_LEMA = {"cod": "coding", "datum": "data"}
STOPWORDS_EXTRA = {
    "like", "know", "think", "thing", "things", "yeah", "kind", "lot", "way",
    "right", "mean", "gonna", "want", "need", "going", "get", "got", "say",
    "said", "actually", "really", "okay", "sort", "stuff", "people", "time",
    "good", "great", "little", "bit", "guy", "talk", "look", "come", "make",
    "use", "work", "try", "let", "tell", "maybe", "pretty", "sure", "able",
    "different", "start", "point", "feel", "happen", "year", "day", "today",
    "slide", "thank", "question",
}
TOP_N = 20

# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
PALETA = {
    "positivo": "#2E9E5B",
    "neutro": "#8A8F98",
    "negativo": "#D1495B",
    "destaque": "#3D5A80",
    "apoio": "#EE9B00",
    "texto": "#3D5A80",   # etapa 1
    "fala": "#EE9B00",    # etapa 2
    "video": "#2E9E5B",   # etapa 3
    "slide": "#3D5A80",
    "codigo": "#1C2A45",
}
# Uma cor por tipo de tela, iguais em todos os gráficos, e a ordem dos tipos
# em todo o projeto: primeiro os slides, depois as capturas de ferramenta.
# Código é escuro como um editor em tema escuro; movimento é cinza claro
# (tela sem parar).
PALETA_TELA = {
    "texto": "#5B8DB8",
    "visual": "#B07CC6",
    "código": "#1C2A45",
    "interface": "#2A9D8F",
    "movimento": "#BFC3C9",
}
DPI = 140
