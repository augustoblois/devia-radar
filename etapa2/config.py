"""
DevIA Radar - Configurações centrais da Etapa 2 (semântica e sentimento).

Etapa 2 da disciplina Dados Não Estruturados (UFPB).
O tema continua o da etapa 1: como desenvolvedores discutem o uso de
Inteligência Artificial no desenvolvimento de software. A fonte também é a
mesma, as 979 issues do GitHub coletadas e limpas na etapa 1. Muda a
técnica: a etapa 1 CONTOU palavras (regex e léxico do VADER); a etapa 2
mede o SIGNIFICADO, com o pipeline do spaCy, embeddings e transformers.

Todo parâmetro que a gente pode querer ajustar durante a apresentação
mora aqui, para não precisar caçar valor mágico no meio da lógica.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
# Mesma correção das outras etapas: o console do Windows usa cp1252 e quebra
# com UnicodeEncodeError ao imprimir emoji ou aspa curva das issues.
for _saida in (sys.stdout, sys.stderr):
    try:
        _saida.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Os modelos vêm do Hugging Face Hub, que imprime um aviso longo sobre
# "symlinks" toda vez que roda no Windows. O aviso não muda nada no resultado
# e polui a tela da demonstração.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).parent
DIR_BRUTOS = RAIZ / "dados" / "brutos"          # fora do git: frases e vetores (refeitos pelo programa)
DIR_PROCESSADOS = RAIZ / "dados" / "processados"
DIR_SAIDA = RAIZ / "saida"

# A fonte: o corpus limpo da etapa 1, que já está no git.
ARQ_ETAPA1_CORPUS = RAIZ.parent / "etapa1" / "dados" / "processados" / "corpus_limpo.json"

# Cache das partes lentas. Ficam fora do git pelo tamanho: as frases com o
# texto ocupam ~10 MB e os vetores, ~50 MB. A opção 0 refaz tudo.
ARQ_FRASES = DIR_BRUTOS / "frases.json"
ARQ_VETORES_FRASES = DIR_BRUTOS / "vetores_frases.npy"
ARQ_VETORES_ISSUES = DIR_BRUTOS / "vetores_issues.npy"
ARQ_TRANSFORMER = DIR_BRUTOS / "sentimento_transformer.json"

# Resultados, pequenos e no git: os gráficos saem daqui.
ARQ_PIPELINE = DIR_PROCESSADOS / "analise_pipeline.json"
ARQ_TEMAS = DIR_PROCESSADOS / "temas.json"
ARQ_SENTIMENTO = DIR_PROCESSADOS / "analise_sentimento.json"

for _pasta in (DIR_BRUTOS, DIR_PROCESSADOS, DIR_SAIDA):
    _pasta.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline do spaCy
# ---------------------------------------------------------------------------
# O modelo médio, e não o pequeno da etapa 1: o "md" traz os mesmos
# componentes e, além deles, um vetor de 300 números para cada palavra do
# vocabulário (os embeddings estáticos da opção 4). Instala com:
#     python -m spacy download en_core_web_md
MODELO_SPACY = "en_core_web_md"

# O MESMO dicionário de ferramentas da etapa 1. Na etapa 1 ele virava regex;
# aqui ele vira um componente do pipeline (entity_ruler), e cada variação
# passa a ser uma entidade com o rótulo FERRAMENTA.
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
ROTULO_FERRAMENTA = "FERRAMENTA"

# "Cursor" sozinho é o produto e também palavra comum (o cursor da tela, o
# cursor de paginação de uma API). Na escrita, a maiúscula é escolha de quem
# escreve: "Cursor" é o produto, "cursor" quase nunca é. Por isso a regra do
# Cursor sozinho compara o texto exato (TEXT), e as outras, a forma
# minúscula (LOWER).
CURSOR_SOZINHO = "Cursor"

# Frase com menos palavras que isso não sustenta sentimento ("Thanks!",
# "Steps to reproduce:"). É a mesma regra das etapas de fala.
MIN_PALAVRAS_FRASE = 6

POS_RELEVANTES = {"NOUN", "PROPN", "ADJ", "VERB"}
# Palavras que passam pelo filtro do spaCy e não dizem nada do tema (as
# mesmas da etapa 1).
STOPWORDS_EXTRA = {
    "issue", "pr", "bug", "fix", "use", "using", "used", "like", "get",
    "would", "could", "also", "one", "thing", "way", "need", "want",
    "make", "work", "try", "see", "know", "think", "good", "new", "add",
    "https", "http", "com", "www", "github", "org", "file", "line",
}
# Erros do lematizador do spaCy, os mesmos das outras etapas: o verbo
# "coding" vira "cod" e "data" vira "datum" (o singular em latim).
CORRECAO_LEMA = {"cod": "coding", "datum": "data"}

TOP_N = 20

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Palavras cujos vizinhos a opção 4 mostra nos vetores estáticos do spaCy.
PALAVRAS_DEMO = ["copilot", "claude", "cursor", "gemini", "bug", "prompt"]
VIZINHOS = 8

# Modelo de embeddings de FRASE (sentence-transformers). O MiniLM tem 6
# camadas de transformer, 22 milhões de parâmetros e devolve 384 números por
# frase. É o modelo de entrada mais usado da biblioteca: roda em CPU e foi
# treinado em 1 bilhão de pares de frases parecidas.
MODELO_FRASES = "sentence-transformers/all-MiniLM-L6-v2"
LOTE_FRASES = 64

# Temas: as issues são agrupadas pelo KMeans nos vetores de frase. O número
# de grupos é testado de MIN_TEMAS a MAX_TEMAS, e fica o de maior
# silhueta (ver embeddings.agrupar).
MIN_TEMAS = 4
MAX_TEMAS = 12
SEMENTE = 42      # o KMeans e o t-SNE começam de um sorteio; a semente fixa o sorteio
TERMOS_TEMA = 6   # termos que dão nome a cada tema

# ---------------------------------------------------------------------------
# Sentimento
# ---------------------------------------------------------------------------
# VADER: as mesmas faixas da etapa 1.
LIMITE_POSITIVO = 0.05
LIMITE_NEGATIVO = -0.05

# Transformer de sentimento: RoBERTa (125 milhões de parâmetros) treinado em
# 124 milhões de tweets e ajustado para três classes: negativo, neutro e
# positivo. As mesmas três classes do VADER, e o mesmo tipo de texto curto e
# informal para o qual o VADER foi feito: a comparação é justa.
MODELO_SENTIMENTO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LOTE_SENTIMENTO = 32
MAX_TOKENS = 512  # o teto do modelo; frase mais longa que isso é cortada

# O modelo devolve rótulos em inglês; o programa fala português.
ROTULOS_TRANSFORMER = {"negative": "negativo", "neutral": "neutro", "positive": "positivo"}

# A apresentação não pode depender da internet da sala. Se os dois modelos
# já estão no cache do Hugging Face (depois da primeira execução), o
# programa nem tenta a rede. Sem esta linha, cada carga fazia consultas ao
# Hub e imprimia avisos no meio da demo.
_CACHE_HF = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
if all((_CACHE_HF / f"models--{modelo.replace('/', '--')}").exists() for modelo in (MODELO_FRASES, MODELO_SENTIMENTO)):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
PALETA = {
    "positivo": "#2E9E5B",
    "neutro": "#8A8F98",
    "negativo": "#D1495B",
    "destaque": "#3D5A80",
    "apoio": "#EE9B00",
    "vader": "#EE9B00",        # o medidor da etapa 1
    "transformer": "#3D5A80",  # o medidor da etapa 2
}
DPI = 140
