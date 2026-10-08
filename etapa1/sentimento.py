"""
Etapa 1 - ANÁLISE DE SENTIMENTO COM VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) vem dentro do NLTK.
Ele é baseado em léxico: existe uma lista onde cada palavra já tem uma nota
de valência dada por avaliadores humanos ("awesome" +3.1, "broken" -1.9).

Por que VADER e não um modelo treinado:
- não precisa de dados rotulados nem de treino, roda na hora;
- entende reforço de pontuação ("great!!!" pesa mais que "great");
- entende maiúscula ("GREAT" pesa mais que "great");
- entende negação ("not great" vira negativo);
- entende atenuador ("kind of great" pesa menos).
Isso encaixa com texto de fórum e issue, que é o nosso caso.

A saída principal é o `compound`: soma normalizada entre -1 e +1.
Convenção da documentação oficial, replicada no config:
    compound >= +0.05  positivo
    compound <= -0.05  negativo
    entre os dois       neutro

Limitação honesta, boa de citar na apresentação: VADER não entende ironia
nem jargão técnico. Em issue de bug, a palavra "critical" é vocabulário
técnico, mas o léxico lê como carga negativa.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

import config

RE_SENTENCA = re.compile(r"(?<=[.!?])\s+")

_ANALISADOR = None


def carregar_analisador():
    """Carrega o VADER, baixando o léxico na primeira vez."""
    global _ANALISADOR
    if _ANALISADOR is not None:
        return _ANALISADOR

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        _ANALISADOR = SentimentIntensityAnalyzer()
    except LookupError:
        print("  baixando léxico do VADER (só na primeira vez)...")
        nltk.download("vader_lexicon", quiet=True)
        _ANALISADOR = SentimentIntensityAnalyzer()

    return _ANALISADOR


def classificar(compound: float) -> str:
    if compound >= config.LIMITE_POSITIVO:
        return "positivo"
    if compound <= config.LIMITE_NEGATIVO:
        return "negativo"
    return "neutro"


def analisar(corpus: list[dict[str, Any]], salvar: bool = True) -> dict[str, Any]:
    """Pontua cada documento e agrega por ferramenta e por mês."""
    vader = carregar_analisador()

    distribuicao: Counter[str] = Counter()
    por_ferramenta: dict[str, list[float]] = defaultdict(list)
    por_mes: dict[str, list[float]] = defaultdict(list)
    notas: list[float] = []
    exemplos: list[tuple[float, str, str]] = []
    sentencas_vistas: set[str] = set()

    for doc in corpus:
        # Truncar em 2000 caracteres: opinião em issue aparece no começo, e
        # texto muito longo dilui o compound para perto de zero.
        texto = doc["texto_limpo"][:2000]
        compound = vader.polarity_scores(texto)["compound"]
        rotulo = classificar(compound)

        doc["sentimento"] = rotulo
        doc["compound"] = round(compound, 4)

        notas.append(compound)
        distribuicao[rotulo] += 1

        for ferramenta in doc.get("ferramentas_ia", []):
            por_ferramenta[ferramenta].append(compound)

        if doc.get("criado_em"):
            por_mes[doc["criado_em"][:7]].append(compound)

        # Guardamos as sentenças mais carregadas para mostrar exemplos reais
        # na apresentação. O conjunto `sentencas_vistas` evita a mesma frase
        # de template aparecer três vezes seguidas no ranking.
        for sentenca in RE_SENTENCA.split(texto):
            sentenca = sentenca.strip()
            if not 40 <= len(sentenca) <= 220:
                continue
            chave = sentenca.lower()[:80]
            if chave in sentencas_vistas:
                continue
            nota = vader.polarity_scores(sentenca)["compound"]
            if abs(nota) > 0.75:
                sentencas_vistas.add(chave)
                exemplos.append((nota, sentenca, doc["repositorio"]))

    exemplos.sort(key=lambda t: t[0])

    resumo_ferramentas = {
        nome: {
            "mencoes": len(valores),
            "media": round(sum(valores) / len(valores), 4),
            "positivos": sum(1 for v in valores if classificar(v) == "positivo"),
            "negativos": sum(1 for v in valores if classificar(v) == "negativo"),
            "neutros": sum(1 for v in valores if classificar(v) == "neutro"),
        }
        for nome, valores in sorted(
            por_ferramenta.items(), key=lambda par: -len(par[1])
        )
        if len(valores) >= 5  # abaixo disso a média não significa nada
    }

    linha_tempo = {
        mes: {"media": round(sum(v) / len(v), 4), "volume": len(v)}
        for mes, v in sorted(por_mes.items())
        if len(v) >= 3
    }

    resultado = {
        "distribuicao": dict(distribuicao),
        "media_geral": round(sum(notas) / max(len(notas), 1), 4),
        "por_ferramenta": resumo_ferramentas,
        "linha_tempo": linha_tempo,
        "exemplos_negativos": [
            {"nota": round(n, 3), "texto": t, "repositorio": r}
            for n, t, r in exemplos[:5]
        ],
        "exemplos_positivos": [
            {"nota": round(n, 3), "texto": t, "repositorio": r}
            for n, t, r in exemplos[-5:][::-1]
        ],
    }

    if salvar:
        with config.ARQ_SENTIMENTO.open("w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, ensure_ascii=False, indent=2)
        # O corpus agora carrega o rótulo de sentimento em cada documento.
        with config.ARQ_LIMPO.open("w", encoding="utf-8") as arquivo:
            json.dump(corpus, arquivo, ensure_ascii=False, indent=2)

    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_SENTIMENTO.exists():
        return {}
    with config.ARQ_SENTIMENTO.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def imprimir(analise: dict[str, Any]) -> None:
    if not analise:
        print("  Nenhuma análise de sentimento salva. Rode a opção antes.")
        return

    total = sum(analise["distribuicao"].values()) or 1

    print("\n  DISTRIBUIÇÃO GERAL")
    print("  " + "-" * 40)
    for rotulo in ("positivo", "neutro", "negativo"):
        qtd = analise["distribuicao"].get(rotulo, 0)
        pct = qtd / total * 100
        barra = "#" * int(pct / 2)
        print(f"    {rotulo:<10} {qtd:>5} ({pct:>5.1f}%) {barra}")
    print(f"    compound médio do corpus: {analise['media_geral']:+.4f}")

    print("\n  SENTIMENTO POR FERRAMENTA DE IA")
    print("  " + "-" * 62)
    print(f"    {'ferramenta':<20}{'menções':>9}{'média':>9}{'pos':>7}{'neu':>7}{'neg':>7}")
    for nome, dados in analise["por_ferramenta"].items():
        print(
            f"    {nome:<20}{dados['mencoes']:>9}{dados['media']:>+9.3f}"
            f"{dados['positivos']:>7}{dados['neutros']:>7}{dados['negativos']:>7}"
        )

    print("\n  FRASES MAIS NEGATIVAS DO CORPUS")
    print("  " + "-" * 62)
    for item in analise["exemplos_negativos"][:3]:
        print(f"    [{item['nota']:+.2f}] {item['texto'][:150]}")
        print(f"           fonte: {item['repositorio']}")

    print("\n  FRASES MAIS POSITIVAS DO CORPUS")
    print("  " + "-" * 62)
    for item in analise["exemplos_positivos"][:3]:
        print(f"    [{item['nota']:+.2f}] {item['texto'][:150]}")
        print(f"           fonte: {item['repositorio']}")
    print()


if __name__ == "__main__":
    import limpeza

    imprimir(analisar(limpeza.carregar_corpus()))
