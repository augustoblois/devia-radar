"""
DevIA Radar - Configurações centrais da Etapa 2 (áudio).

Etapa 2 da disciplina Dados Não Estruturados (UFPB).
O tema continua o da etapa 1: como desenvolvedores discutem o uso de
Inteligência Artificial no desenvolvimento de software. Agora a fonte é a
FALA, em episódios de podcast.

Todo parâmetro que a gente pode querer ajustar durante a apresentação
mora aqui, para não precisar caçar valor mágico no meio da lógica.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
# Mesma correção da etapa 1: o console do Windows usa cp1252 e quebra com
# UnicodeEncodeError ao imprimir emoji ou aspa curva dos títulos de podcast.
for _saida in (sys.stdout, sys.stderr):
    try:
        _saida.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# O Whisper baixa o modelo pelo huggingface_hub, que imprime um aviso longo
# sobre "symlinks" toda vez que roda no Windows. O aviso não muda nada no
# resultado e polui a tela da demonstração.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).parent
DIR_BRUTOS = RAIZ / "dados" / "brutos"        # fora do git: mídia e transcrição
DIR_AUDIO = DIR_BRUTOS / "audio"
DIR_TRANSCRICOES = DIR_BRUTOS / "transcricoes"
DIR_TRECHOS = DIR_BRUTOS / "trechos"          # recortes .wav tocados na demo
DIR_PROCESSADOS = RAIZ / "dados" / "processados"
DIR_SAIDA = RAIZ / "saida"

ARQ_EPISODIOS = DIR_PROCESSADOS / "episodios.json"
ARQ_ACUSTICA = DIR_PROCESSADOS / "acustica.json"
ARQ_FALA = DIR_PROCESSADOS / "analise_fala.json"

# Resultados da etapa 1, lidos para comparar o texto escrito com a fala.
DIR_ETAPA1 = RAIZ.parent / "etapa1" / "dados" / "processados"
ARQ_ETAPA1_CORPUS = DIR_ETAPA1 / "corpus_limpo.json"
ARQ_ETAPA1_PLN = DIR_ETAPA1 / "analise_pln.json"

for _pasta in (DIR_AUDIO, DIR_TRANSCRICOES, DIR_TRECHOS, DIR_PROCESSADOS, DIR_SAIDA):
    _pasta.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Coleta (iTunes Search API, a busca pública de podcasts da Apple)
# ---------------------------------------------------------------------------
API_ITUNES = "https://itunes.apple.com/search"

# Cada termo vira uma consulta separada, como na etapa 1.
TERMOS_BUSCA = [
    "github copilot",
    "claude code",
    "cursor ai",
    "vibe coding",
    "ai code review",
    "ai pair programming",
    "ai generated code",
]

RESULTADOS_POR_TERMO = 200   # teto que a API aceita por consulta
EPISODIOS_POR_TERMO = 2      # 7 termos x 2 = 14 episódios
DURACAO_MIN = 20             # minutos: episódio curto demais é notícia rápida
DURACAO_MAX = 60             # minutos: limita o tempo de transcrição
ANOS_ACEITOS = ("2025", "2026")
PAUSA_ENTRE_CONSULTAS = 3.0  # segundos; a API da Apple aceita ~20 consultas/min

# ---------------------------------------------------------------------------
# Sinal de áudio (librosa)
# ---------------------------------------------------------------------------
# 16.000 amostras por segundo. Pelo teorema de Nyquist, isso representa
# frequências até 8.000 Hz, que cobrem a voz humana com folga. É também a
# taxa que o Whisper exige na entrada.
TAXA_AMOSTRAGEM = 16_000

JANELA = 400     # 400 amostras = 25 ms: trecho curto o bastante para a voz
                 # parecer "parada" dentro dele
SALTO = 160      # 160 amostras = 10 ms entre o início de uma janela e a outra
N_FFT = 512      # tamanho da FFT (potência de 2, maior que a janela)
N_MELS = 64      # bandas da escala Mel no espectrograma
N_MFCC = 13      # coeficientes MFCC, o padrão clássico em reconhecimento de fala

# Trecho mais de 35 dB abaixo do pico do episódio conta como silêncio.
TOP_DB_SILENCIO = 35
PAUSA_LONGA = 0.7        # segundos: abaixo disso é só o respiro entre palavras

# Faixa de frequência fundamental (F0) da voz falada. Abaixo de 65 Hz e
# acima de 400 Hz, o que o detector acha é ruído ou harmônico.
F0_MIN = 65
F0_MAX = 400
JANELA_F0 = 1024         # 64 ms: o YIN precisa de pelo menos 2 períodos de F0_MIN

TRECHO_DEMO = 20         # segundos de áudio nas demonstrações ao vivo
# Episódio e segundo de início do trecho das demos (opções 3 e 5): o episódio
# do Y Combinator sobre Claude Code, num trecho com cinco nomes de ferramenta.
# Sem a lista de vocabulário o Whisper escreve "Sonic 3.5" e "FODCode";
# com a lista, "Sonnet 3.5" e "Claude Code".
TRECHO_DEMO_EPISODIO = "1000748546511"
TRECHO_DEMO_INICIO = 128.0

# ---------------------------------------------------------------------------
# Transcrição (Whisper, via faster-whisper)
# ---------------------------------------------------------------------------
MODELO_WHISPER = "small"     # 244 milhões de parâmetros, roda bem em CPU
FEIXE = 5                    # beam search: mantém as 5 melhores hipóteses

# GPU quando existe, CPU quando não. A GPU precisa de placa NVIDIA e das
# bibliotecas CUDA instaladas pelo pip (ver README). Medido neste notebook
# (RTX 3050): 20x o tempo real na GPU; 1,2x na CPU, com dois processos.
DISPOSITIVO = "auto"              # "auto", "cuda" ou "cpu"
TIPO_COMPUTO_GPU = "int8_float16" # pesos em 8 bits, contas em 16 bits
TIPO_COMPUTO_CPU = "int8"         # pesos em 8 bits: ~4x mais leve, quase sem perda
THREADS = max((os.cpu_count() or 4) // 2, 1)

# Lista de vocabulário entregue ao Whisper (parâmetro `hotwords`) em TODAS as
# janelas de 30 s. O modelo passa a preferir essas grafias quando o som é
# ambíguo. Medido em 10 minutos de um episódio sobre Claude Code: sem a
# lista, 13 acertos de "Claude" e 3 erros ("cloud code"); com a lista via
# `hotwords`, 16 acertos e nenhum erro. Ver o diário de bordo.
PROMPT_VOCABULARIO = (
    "A conversation between software developers about AI coding tools: "
    "Claude, Claude Code, Anthropic, ChatGPT, OpenAI, Codex, GitHub Copilot, "
    "Cursor, Gemini, Windsurf, LLM, MCP, vibe coding, code review, pull request."
)

# ---------------------------------------------------------------------------
# Conteúdo da fala
# ---------------------------------------------------------------------------
# O MESMO dicionário da etapa 1, de propósito: comparar a fala com o texto só
# faz sentido se as duas contagens usam exatamente as mesmas regras.
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

# A única correção ao dicionário. "Cursor" sozinho é o produto e também
# palavra comum (o cursor da tela, o cursor de paginação), por isso a etapa
# 1 só contava "cursor ai", "cursor editor" e "cursor ide", e perdia quase
# todas as menções (ver o diário de bordo). A pista que separa os dois usos
# depende da mídia:
# - ESCRITA (as issues da etapa 1): a maiúscula é escolha do autor.
#   "Cursor" é o produto; "cursor" minúsculo quase nunca é.
# - FALA (Whisper): a maiúscula é escolha do modelo, que escreve o produto
#   em minúscula uma vez em cada três. Ali vale qualquer caixa, menos o
#   cursor da tela dito por extenso.
CURSOR_ESCRITA = r"\bCursor\b"
CURSOR_FALA = r"(?<!blinking )(?<!mouse )(?<!text )\bcursor\b(?! position)"

MODELO_SPACY = "en_core_web_sm"
POS_RELEVANTES = {"NOUN", "PROPN", "ADJ", "VERB"}

# Erros do lematizador do modelo pequeno do spaCy, achados na nuvem de
# palavras: o verbo "coding" vira "cod" e "data" vira "datum" (o singular em
# latim). Sem a correção, "vibe coding" aparecia partido em dois bigramas.
CORRECAO_LEMA = {"cod": "coding", "datum": "data"}

# Palavras que dominam qualquer conversa falada e não dizem nada do tema.
# Fala tem muito mais muleta ("kind of", "you know") do que texto escrito.
STOPWORDS_EXTRA = {
    "like", "know", "think", "thing", "things", "yeah", "kind", "lot", "way",
    "right", "mean", "gonna", "want", "need", "going", "get", "got", "say",
    "said", "actually", "really", "okay", "sort", "stuff", "people", "time",
    "good", "great", "little", "bit", "guy", "talk", "look", "come", "make",
    "use", "work", "try", "let", "tell", "maybe", "pretty", "sure", "able",
    "different", "start", "point", "feel", "happen", "year", "day", "today",
}

LIMITE_POSITIVO = 0.05   # mesmas faixas do VADER usadas na etapa 1
LIMITE_NEGATIVO = -0.05
MIN_PALAVRAS_FRASE = 6   # frase curta demais ("Yeah, right.") não sustenta sentimento

# Palavras que saem do léxico do VADER, nas duas mídias. O VADER foi feito
# para texto de rede social e dá +1,5 a "like". Na fala, "like" aparece 16
# vezes a cada mil palavras (0,5 no texto das issues), e o spaCy marca só 4%
# delas como o verbo "gostar": o resto é muleta ("it's like, you know") ou
# comparação ("like this"). Com o "like" no léxico, a fala parecia mais
# positiva que o texto só por causa da muleta (ver o diário de bordo).
SEM_SENTIMENTO = {"like"}

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
    "texto": "#3D5A80",   # etapa 1 (issues) nos gráficos de comparação
    "fala": "#EE9B00",    # etapa 2 (podcasts)
}
DPI = 140
