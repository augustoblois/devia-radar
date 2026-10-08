"""
Etapa 1 - LIMPEZA E EXTRAÇÃO COM EXPRESSÕES REGULARES (RE).

Expressão regular é uma mini linguagem para descrever padrões de texto.
Em vez de perguntar "esse texto é igual a tal palavra", a gente pergunta
"esse texto tem a FORMA de uma URL / de uma versão / de um bloco de código".

O módulo tem duas responsabilidades:

1. LIMPAR: issue do GitHub vem cheia de coisa que não é linguagem natural
   (bloco de código, stack trace, HTML, link). Isso envenena a análise de
   PLN, porque `import`, `def` e `https` virariam as palavras mais frequentes
   do corpus. Regex remove esse ruído.

2. EXTRAIR: a mesma regex que localiza o ruído serve para colher informação
   estruturada. Quais ferramentas de IA a issue cita? Quais linguagens?
   Ela mostra código? Tem stack trace (sinal de bug relatado)?

Tabela rápida dos símbolos usados abaixo:
    \\b      fronteira de palavra, impede casar "go" dentro de "google"
    \\d      um dígito
    \\s      espaço, tab ou quebra de linha
    +        uma ou mais repetições
    *?       zero ou mais, modo preguiçoso (para o mais cedo possível)
    [^x]     qualquer caractere que não seja x
    (?:...)  agrupa sem capturar
    |        alternativa (ou)
"""

from __future__ import annotations

import json
import re
from typing import Any

import config

# ---------------------------------------------------------------------------
# 1. Padrões de RUÍDO (o que será removido do texto)
# ---------------------------------------------------------------------------

# Bloco de código cercado por três crases. [\s\S] casa qualquer caractere,
# inclusive quebra de linha, coisa que o ponto sozinho não faz.
RE_BLOCO_CODIGO = re.compile(r"```[\s\S]*?```")

# Código curto no meio da frase, entre uma crase.
RE_CODIGO_INLINE = re.compile(r"`[^`\n]+`")

# Imagem em markdown: ![alt](url). Precisa vir antes do link comum.
RE_IMAGEM_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Link em markdown [texto](url): guardamos só o texto visível.
RE_LINK_MD = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Tag HTML solta, comum em issue com <details> ou <img>.
RE_HTML = re.compile(r"<[^>]{1,200}>")

# Citação de resposta anterior (linha começando com >).
RE_CITACAO = re.compile(r"^\s*>.*$", re.MULTILINE)

# --- Sintaxe de markdown -----------------------------------------------
# Motivo de existir: sem isso o spaCy lia "###" como entidade do tipo MONEY,
# porque "#" é símbolo de moeda em alguns países. Os títulos de template de
# issue apareciam entre as entidades mais frequentes do corpus.
RE_TITULO_MD = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
RE_ENFASE = re.compile(r"(\*{1,3}|_{1,3})(\S(?:[^*_]*\S)?)\1")
RE_CHECKBOX = re.compile(r"^\s*[-*+]\s*\[[ xX]\]\s*", re.MULTILINE)
RE_MARCADOR = re.compile(r"^\s*(?:[-*+]|\d{1,2}[.)])\s+", re.MULTILINE)
RE_SEPARADOR = re.compile(r"^\s*[-=*_]{3,}\s*$", re.MULTILINE)
RE_TABELA_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$", re.MULTILINE)

# Sobra de símbolo, emoji e pontuação decorativa. Mantemos letra, número e
# a pontuação que o spaCy usa para separar sentença.
RE_SIMBOLO = re.compile(r"[^\w\s.,;:!?'\"()/-]", re.UNICODE)

# Espaços repetidos viram um só.
RE_ESPACOS = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# 2. Padrões de EXTRAÇÃO (o que será colhido antes de remover)
# ---------------------------------------------------------------------------

RE_URL = re.compile(r"https?://[^\s)\]>\"']+")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_MENCAO = re.compile(r"@([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)\b")
RE_REFERENCIA = re.compile(r"#(\d{1,6})\b")

# Versão semântica: v1.2.3, 2.0, 18.3.1
RE_VERSAO = re.compile(r"\bv?(\d{1,3}\.\d{1,3}(?:\.\d{1,4})?)\b")

# Stack trace em várias linguagens ao mesmo tempo.
RE_STACKTRACE = re.compile(
    r'(?:File\s+"[^"]+",\s+line\s+\d+'  # Python
    r"|\bat\s+[\w.$]+\([^)]*\.(?:java|js|ts|kt):\d+\)"  # Java / JS
    r"|Traceback \(most recent call last\)"  # Python
    r"|\bException in thread\b)",
    re.IGNORECASE,
)


def _regex_de_vocabulario(termos: list[str]) -> re.Pattern[str]:
    """
    Monta uma regex única a partir de uma lista de palavras.

    Ordenamos do termo mais longo para o mais curto porque a alternativa `|`
    para na PRIMEIRA que casar. Sem isso, "copilot" casaria antes de
    "github copilot" e a gente perderia o rótulo mais específico.
    """
    ordenados = sorted(termos, key=len, reverse=True)
    alternativas = "|".join(re.escape(t) if " " in t else t for t in ordenados)
    return re.compile(rf"\b(?:{alternativas})\b", re.IGNORECASE)


# Uma regex por ferramenta de IA, montada a partir do config.
REGEX_FERRAMENTAS = {
    nome: _regex_de_vocabulario(variacoes)
    for nome, variacoes in config.FERRAMENTAS_IA.items()
}

RE_LINGUAGENS = re.compile(
    rf"\b(?:{'|'.join(config.LINGUAGENS)})\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# 3. Funções públicas
# ---------------------------------------------------------------------------


def extrair_padroes(texto: str) -> dict[str, Any]:
    """
    Colhe a informação estruturada ANTES de o texto ser limpo.

    Duas fontes diferentes de propósito. Os sinais estruturais (tem código,
    tem stack trace, URLs, versões) saem do texto bruto, porque é justamente
    no trecho técnico que eles vivem. Já ferramenta de IA e linguagem de
    programação saem do texto SEM os blocos de código.

    O motivo veio de um falso positivo real: `User::class.java` dentro de um
    trecho em Kotlin fazia a issue ser contada como issue sobre Java.
    """
    prosa = RE_BLOCO_CODIGO.sub(" ", texto)

    ferramentas = [
        nome for nome, padrao in REGEX_FERRAMENTAS.items() if padrao.search(prosa)
    ]
    linguagens = sorted({m.lower() for m in RE_LINGUAGENS.findall(prosa)})

    return {
        "ferramentas_ia": ferramentas,
        "linguagens": linguagens,
        "urls": RE_URL.findall(texto)[:10],
        "versoes": sorted(set(RE_VERSAO.findall(texto)))[:10],
        "mencoes": sorted(set(RE_MENCAO.findall(texto)))[:10],
        "referencias": sorted(set(RE_REFERENCIA.findall(texto)))[:10],
        "tem_codigo": bool(RE_BLOCO_CODIGO.search(texto)),
        "tem_stacktrace": bool(RE_STACKTRACE.search(texto)),
        "qtd_blocos_codigo": len(RE_BLOCO_CODIGO.findall(texto)),
    }


def limpar_texto(texto: str) -> str:
    """
    Remove o ruído técnico e devolve só a linguagem natural.

    A ordem importa. Bloco de código sai primeiro porque pode conter URL,
    tag HTML e crase dentro dele. Se a gente limpasse na ordem inversa,
    sobrariam pedaços soltos de código no texto final.
    """
    texto = RE_BLOCO_CODIGO.sub(" ", texto)
    texto = RE_STACKTRACE.sub(" ", texto)
    texto = RE_IMAGEM_MD.sub(" ", texto)
    texto = RE_LINK_MD.sub(r"\1", texto)  # mantém o texto, descarta o endereço
    texto = RE_URL.sub(" ", texto)
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_HTML.sub(" ", texto)
    texto = RE_CITACAO.sub(" ", texto)
    texto = RE_CODIGO_INLINE.sub(" ", texto)
    texto = RE_MENCAO.sub(" ", texto)
    texto = RE_REFERENCIA.sub(" ", texto)

    # Sintaxe de markdown vira espaço ou perde a marcação, mantendo o texto.
    texto = RE_TABELA_SEP.sub(" ", texto)
    texto = RE_SEPARADOR.sub(" ", texto)
    texto = RE_TITULO_MD.sub(" ", texto)
    texto = RE_CHECKBOX.sub(" ", texto)
    texto = RE_MARCADOR.sub(" ", texto)
    texto = RE_ENFASE.sub(r"\2", texto)
    texto = RE_SIMBOLO.sub(" ", texto)

    texto = RE_ESPACOS.sub(" ", texto)
    return texto.strip()


# Conta de robô termina em [bot] no GitHub. Alguns times também usam sufixo
# -bot ou -ci. Issue de robô é relatório automático, não opinião de pessoa.
RE_AUTOR_BOT = re.compile(r"(\[bot\]$|^dependabot|-(?:bot|ci)$)", re.IGNORECASE)

# Título gerado por automação de pipeline.
RE_TITULO_AUTOMATICO = re.compile(
    r"^\s*(?:\[?(?:ci|build|chore|automated|nightly)\]?\b"
    r"|workflow failure"
    r"|.*\bfailed\s+(?:on|in|for)\s+(?:main|master|ci)\b)",
    re.IGNORECASE,
)


def processar(issues: list[dict[str, Any]], salvar: bool = True) -> list[dict[str, Any]]:
    """
    Aplica extração e limpeza em todo o corpus.

    Além de limpar, a função descarta três tipos de documento que sujam a
    análise. Os três foram descobertos olhando o resultado, não antes dele:

    1. Issue de robô. O bigrama mais frequente do corpus era "workflow
       failure", vindo de bot de CI, e não de pessoa discutindo IA.
    2. Texto curto demais depois da limpeza, que era só código ou log.
    3. Texto duplicado. Template de issue faz o mesmo bloco aparecer dezenas
       de vezes e inflar artificialmente a frequência dos termos dele.
    """
    corpus: list[dict[str, Any]] = []
    vistos: set[int] = set()
    descartes = {"bot": 0, "curto": 0, "duplicado": 0}

    for issue in issues:
        autor = issue.get("autor", "")
        if RE_AUTOR_BOT.search(autor) or RE_TITULO_AUTOMATICO.match(issue["titulo"]):
            descartes["bot"] += 1
            continue

        bruto = f"{issue['titulo']}\n{issue['corpo']}"
        padroes = extrair_padroes(bruto)
        limpo = limpar_texto(bruto)

        if len(limpo) < 40:
            descartes["curto"] += 1
            continue

        assinatura = hash(limpo.lower())
        if assinatura in vistos:
            descartes["duplicado"] += 1
            continue
        vistos.add(assinatura)

        corpus.append(
            {
                "id": issue["id"],
                "repositorio": issue["repositorio"],
                "titulo": issue["titulo"],
                "texto_limpo": limpo,
                "criado_em": issue["criado_em"],
                "qtd_comentarios": issue["qtd_comentarios"],
                "url": issue["url"],
                "caracteres_antes": len(bruto),
                "caracteres_depois": len(limpo),
                **padroes,
            }
        )

    if salvar:
        with config.ARQ_LIMPO.open("w", encoding="utf-8") as arquivo:
            json.dump(corpus, arquivo, ensure_ascii=False, indent=2)

    antes = sum(d["caracteres_antes"] for d in corpus) or 1
    depois = sum(d["caracteres_depois"] for d in corpus)
    reducao = (1 - depois / antes) * 100

    print(f"  documentos úteis .... {len(corpus)} de {len(issues)}")
    print(
        f"  descartados ......... {descartes['bot']} de robô, "
        f"{descartes['curto']} curtos, {descartes['duplicado']} duplicados"
    )
    print(f"  ruído removido ...... {reducao:.1f}% dos caracteres")
    print(f"  com bloco de código . {sum(d['tem_codigo'] for d in corpus)}")
    print(f"  com stack trace ..... {sum(d['tem_stacktrace'] for d in corpus)}")
    if salvar:
        print(f"  salvo em ............ {config.ARQ_LIMPO.name}")

    return corpus


def carregar_corpus() -> list[dict[str, Any]]:
    """Le o corpus já limpo do disco."""
    if not config.ARQ_LIMPO.exists():
        return []
    with config.ARQ_LIMPO.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def demonstrar(issues: list[dict[str, Any]] | None = None) -> None:
    """
    Mostra na tela o antes e o depois de uma issue real.

    Serve para a apresentação: a professora pediu para explicar os comandos
    usados, e ver a regex agindo em texto de verdade explica melhor que slide.
    """
    exemplo = (
        "## Bug with Copilot suggestions\n"
        "@octocat reported in #1234 that GitHub Copilot v1.2.3 breaks on "
        "Python 3.11. See https://github.com/org/repo/issues/1234 for context.\n"
        "```python\n"
        "def quebrado():\n"
        "    raise ValueError('isso aqui é código, não linguagem natural')\n"
        "```\n"
        "<details>Honestly ChatGPT handled it better.</details>\n"
        "> quoting a previous comment here\n"
    )

    if issues:
        candidatos = [
            i for i in issues if RE_BLOCO_CODIGO.search(i.get("corpo", ""))
        ]
        if candidatos:
            alvo = min(candidatos, key=lambda i: len(i["corpo"]))
            exemplo = f"{alvo['titulo']}\n{alvo['corpo']}"[:900]

    print("=" * 72)
    print("TEXTO BRUTO")
    print("=" * 72)
    print(exemplo)

    print("=" * 72)
    print("PADRÕES EXTRAÍDOS PELA REGEX")
    print("=" * 72)
    for chave, valor in extrair_padroes(exemplo).items():
        print(f"  {chave:.<24} {valor}")

    print("=" * 72)
    print("TEXTO LIMPO (o que vai para o spaCy)")
    print("=" * 72)
    print(limpar_texto(exemplo))
    print()


if __name__ == "__main__":
    import coleta

    demonstrar(coleta.coletar())
