"""
DevIA Radar - Configurações centrais do projeto.

Etapa 1 da disciplina Dados Não Estruturados (UFPB).
Tema: como desenvolvedores discutem o uso de Inteligência Artificial
no desenvolvimento de software.

Todo parâmetro que a gente pode querer ajustar durante a apresentação
mora aqui, para não precisar caçar valor mágico no meio da lógica.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
# O console do Windows usa cp1252 por padrão, que não consegue escrever
# travessão, aspa curva nem emoji. Issue do GitHub vem cheia disso, e o
# print quebraria com UnicodeEncodeError no meio da apresentação.
# `errors="replace"` troca o caractere impossível por "?" em vez de derrubar.
for _saida in (sys.stdout, sys.stderr):
    try:
        _saida.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).parent
DIR_BRUTOS = RAIZ / "dados" / "brutos"
DIR_PROCESSADOS = RAIZ / "dados" / "processados"
DIR_SAIDA = RAIZ / "saida"

ARQ_ISSUES = DIR_BRUTOS / "issues_github.json"
ARQ_LIMPO = DIR_PROCESSADOS / "corpus_limpo.json"
ARQ_PLN = DIR_PROCESSADOS / "analise_pln.json"
ARQ_SENTIMENTO = DIR_PROCESSADOS / "analise_sentimento.json"

for _pasta in (DIR_BRUTOS, DIR_PROCESSADOS, DIR_SAIDA):
    _pasta.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Coleta (GitHub Search API)
# ---------------------------------------------------------------------------
API_BUSCA_ISSUES = "https://api.github.com/search/issues"

# Cada termo vira uma consulta separada na API.
# Escolhemos termos que aparecem quando dev fala de IA escrevendo código.
TERMOS_BUSCA = [
    "github copilot",
    "chatgpt generated code",
    "ai generated code",
    "llm integration",
    "cursor ai editor",
    "ai code review",
]

RESULTADOS_POR_PAGINA = 100  # teto que a API do GitHub aceita
PAGINAS_POR_TERMO = 2  # 2 x 100 = até 200 issues por termo

# A Search API limita 10 consultas/minuto sem token e 30/minuto com token.
# Esperar entre as chamadas evita o erro HTTP 403 (rate limit).
PAUSA_SEM_TOKEN = 7.0
PAUSA_COM_TOKEN = 2.5

# ---------------------------------------------------------------------------
# Domínio: ferramentas de IA que queremos rastrear
# ---------------------------------------------------------------------------
# chave = nome canônico exibido nos gráficos
# valor = lista de variações que a expressão regular vai aceitar
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

LINGUAGENS = [
    "python", "javascript", "typescript", "java", "c#", "c\\+\\+", "go",
    "rust", "ruby", "php", "kotlin", "swift", "scala", "dart", "sql",
]

# ---------------------------------------------------------------------------
# PLN
# ---------------------------------------------------------------------------
MODELO_SPACY = "en_core_web_sm"  # as issues do GitHub são em inglês

# Palavras que passam pelo filtro do spaCy mas não dizem nada sobre o tema.
STOPWORDS_EXTRA = {
    "issue", "pr", "bug", "fix", "use", "using", "used", "like", "get",
    "would", "could", "also", "one", "thing", "way", "need", "want",
    "make", "work", "try", "see", "know", "think", "good", "new", "add",
    "https", "http", "com", "www", "github", "org", "file", "line",
}

# Classes gramaticais que carregam significado para nuvem de palavras.
POS_RELEVANTES = {"NOUN", "PROPN", "ADJ", "VERB"}

# Tipos de entidade nomeada que interessam ao tema.
# Ficaram de fora CARDINAL, ORDINAL, MONEY, PERCENT, TIME e QUANTITY: em
# texto técnico eles só devolvem número solto e símbolo de markdown.
ENTIDADES_RELEVANTES = {
    "ORG",          # organizacao, empresa
    "PRODUCT",      # produto
    "PERSON",       # pessoa
    "GPE",          # pais, cidade, estado
    "NORP",         # nacionalidade, grupo
    "LANGUAGE",     # idioma
    "EVENT",        # evento
    "WORK_OF_ART",  # obra, título
    "DATE",         # data
}

TOP_N = 20  # quantos itens mostrar nos rankings e gráficos

# ---------------------------------------------------------------------------
# Sentimento
# ---------------------------------------------------------------------------
# Faixas do VADER (compound score), conforme a documentação oficial da lib.
LIMITE_POSITIVO = 0.05
LIMITE_NEGATIVO = -0.05

# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------
PALETA = {
    "positivo": "#2E9E5B",
    "neutro": "#8A8F98",
    "negativo": "#D1495B",
    "destaque": "#3D5A80",
    "apoio": "#EE9B00",
}
DPI = 140
