"""
Etapa 2 - ANÁLISE DO CONTEÚDO DA FALA.

Depois da transcrição, a fala vira texto, e as técnicas da etapa 1 voltam:
regex para achar ferramentas de IA, spaCy para o vocabulário e VADER para o
sentimento. A novidade é que agora cada frase tem TEMPO (o segundo em que
começa e termina) e VOZ (energia e altura medidas no sinal.py).

Três perguntas:
1. Quais ferramentas de IA a fala cita, e o ranking bate com o do texto
   escrito nas issues da etapa 1?
2. O tom da fala é o mesmo tom do texto?
3. A voz muda quando a frase é negativa? Aqui o sentimento das PALAVRAS
   (VADER) é cruzado com a energia e a altura da VOZ (librosa).

Vocabulário novo deste módulo:
- Palavras por minuto (WPM): velocidade da fala. Conversa em inglês fica
  perto de 150 WPM.
- Escore z: quantos desvios-padrão um valor está acima da média do próprio
  episódio. Cada podcast grava num volume diferente e cada voz tem uma
  altura própria. Comparar a energia crua misturaria isso com a emoção.
  O escore z responde "esta frase está mais alta que o normal DESTE episódio?".
- Correlação de Spearman (rho): mede se duas grandezas sobem juntas, de -1
  a +1, usando a ordem dos valores. Perto de 0 quer dizer sem relação.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

import config

# ---------------------------------------------------------------------------
# Regex de ferramentas: a mesma construção da etapa 1
# ---------------------------------------------------------------------------


def _regex_de_vocabulario(termos: list[str], extra: str | None = None) -> re.Pattern[str]:
    """
    Uma regex por ferramenta, com as variações ordenadas da mais longa para a
    mais curta. A alternativa `|` para na primeira que casa, então a mais
    longa precisa vir antes (erro 4 do diário da etapa 1). `extra` é uma
    regex pronta somada às variações (a regra do Cursor sozinho).
    """
    ordenados = sorted(termos, key=len, reverse=True)
    alternativas = [re.escape(t) for t in ordenados] + ([extra] if extra else [])
    return re.compile(rf"\b(?:{'|'.join(alternativas)})\b", re.IGNORECASE)


# Na fala, o Cursor sozinho conta com qualquer caixa (config.CURSOR_FALA).
REGEX_FERRAMENTAS = {
    nome: _regex_de_vocabulario(variacoes, config.CURSOR_FALA if nome == "Cursor" else None)
    for nome, variacoes in config.FERRAMENTAS_IA.items()
}
RE_QUALQUER_FERRAMENTA = _regex_de_vocabulario(
    [v for variacoes in config.FERRAMENTAS_IA.values() for v in variacoes], config.CURSOR_FALA
)
# No texto escrito das issues, o Cursor sozinho só conta com a maiúscula
# (config.CURSOR_ESCRITA). O `(?-i:...)` desliga o IGNORECASE só ali dentro.
REGEX_ESCRITA = {
    nome: _regex_de_vocabulario(variacoes, f"(?-i:{config.CURSOR_ESCRITA})" if nome == "Cursor" else None)
    for nome, variacoes in config.FERRAMENTAS_IA.items()
}
RE_CURSOR_DICIONARIO = _regex_de_vocabulario(config.FERRAMENTAS_IA["Cursor"])  # só o da etapa 1

# Fim de frase: ponto, exclamação ou interrogação seguidos de espaço, ou
# quebra de linha (lista de tópicos numa issue).
RE_FRASE = re.compile(r"(?<=[.!?])\s+|\n+")


def ferramentas_em(texto: str) -> list[str]:
    return [nome for nome, padrao in REGEX_FERRAMENTAS.items() if padrao.search(texto)]


# ---------------------------------------------------------------------------
# Sentimento (VADER da etapa 1, sem as palavras de config.SEM_SENTIMENTO)
# ---------------------------------------------------------------------------
_VADER = None


def _vader():
    global _VADER
    if _VADER is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
        for palavra in config.SEM_SENTIMENTO:
            _VADER.lexicon.pop(palavra, None)
    return _VADER


def classificar(compound: float) -> str:
    if compound >= config.LIMITE_POSITIVO:
        return "positivo"
    if compound <= config.LIMITE_NEGATIVO:
        return "negativo"
    return "neutro"


# ---------------------------------------------------------------------------
# Cruzamento com a voz
# ---------------------------------------------------------------------------


def _voz_da_frase(curvas: dict[str, list], inicio: float, fim: float) -> tuple[float | None, float | None]:
    """Energia média e F0 mediana dos segundos que a frase ocupa."""
    primeiro, ultimo = int(inicio), max(int(np.ceil(fim)), int(inicio) + 1)
    energia = curvas["energia_db"][primeiro:ultimo]
    alturas = [f for f in curvas["f0_hz"][primeiro:ultimo] if f is not None]
    return (
        float(np.mean(energia)) if energia else None,
        float(np.median(alturas)) if alturas else None,
    )


def _escore_z(valores: list[float | None]) -> list[float | None]:
    presentes = np.array([v for v in valores if v is not None], dtype=float)
    if presentes.size < 2 or presentes.std() == 0:
        return [None] * len(valores)
    media, desvio = presentes.mean(), presentes.std()
    return [None if v is None else round((v - media) / desvio, 3) for v in valores]


def _spearman(x: list[float], y: list[float]) -> dict[str, float] | None:
    if len(x) < 10:
        return None
    from scipy.stats import spearmanr

    resultado = spearmanr(x, y)
    return {"rho": round(float(resultado.statistic), 3), "p": float(resultado.pvalue), "n": len(x)}


# ---------------------------------------------------------------------------
# Análise principal
# ---------------------------------------------------------------------------


def _frases(segmento: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Corta um trecho do Whisper em frases, cada uma com o seu tempo.

    O Whisper às vezes devolve um trecho de 30 s com mais de 100 palavras, e
    texto longo empurra o VADER para os extremos (o mesmo efeito da issue
    inteira na etapa 1). Por isso a unidade é a frase, nas duas mídias.
    O Whisper só dá o tempo do trecho inteiro, então o de cada frase é
    estimado pela posição dos caracteres: num trecho de 10 s e 100
    caracteres, a frase que vai do caractere 40 ao 80 fica com os segundos
    4 a 8.
    """
    texto = segmento["texto"].strip()
    if not texto:
        return []
    cortes = [0, *(m.end() for m in RE_FRASE.finditer(texto)), len(texto)]
    duracao = segmento["fim"] - segmento["inicio"]
    return [
        {
            "texto": texto[a:b].strip(),
            "inicio": round(segmento["inicio"] + duracao * a / len(texto), 2),
            "fim": round(segmento["inicio"] + duracao * b / len(texto), 2),
        }
        for a, b in itertools.pairwise(cortes)
        if texto[a:b].strip()
    ]


def _frases_do_episodio(transcricao: dict[str, Any], curvas: dict[str, list] | None) -> list[dict[str, Any]]:
    """Anota cada frase falada com palavras, sentimento, ferramentas e voz."""
    vader = _vader()
    anotadas = []
    for frase in (f for segmento in transcricao["segmentos"] for f in _frases(segmento)):
        palavras = len(frase["texto"].split())
        energia, f0 = _voz_da_frase(curvas, frase["inicio"], frase["fim"]) if curvas else (None, None)
        anotadas.append(
            {
                **frase,
                "palavras": palavras,
                "compound": round(vader.polarity_scores(frase["texto"])["compound"], 4),
                "avaliado": palavras >= config.MIN_PALAVRAS_FRASE,
                "ferramentas": ferramentas_em(frase["texto"]),
                "energia_db": None if energia is None else round(energia, 2),
                "f0_hz": None if f0 is None else round(f0, 1),
            }
        )
    for chave in ("energia_db", "f0_hz"):
        for frase, z in zip(anotadas, _escore_z([a[chave] for a in anotadas])):
            frase[f"{chave}_z"] = z
    return anotadas


def _vocabulario(textos: list[str]) -> dict[str, Any]:
    """Lemas e bigramas da fala com spaCy, com o mesmo filtro da etapa 1."""
    import spacy

    nlp = spacy.load(config.MODELO_SPACY, disable=["ner", "parser"])
    lemas: Counter[str] = Counter()
    bigramas: Counter[str] = Counter()
    for doc in nlp.pipe(textos, batch_size=20):
        uteis = [
            config.CORRECAO_LEMA.get(t.lemma_.lower(), t.lemma_.lower())
            for t in doc
            if not (t.is_stop or t.is_punct or t.like_num)
            and t.pos_ in config.POS_RELEVANTES
            and len(t.lemma_) >= 3
            and t.lemma_.isalpha()
            and t.lemma_.lower() not in config.STOPWORDS_EXTRA
        ]
        lemas.update(uteis)
        bigramas.update(f"{a} {b}" for a, b in itertools.pairwise(uteis))
    return {
        "top_lemas": lemas.most_common(config.TOP_N * 3),
        "top_bigramas": bigramas.most_common(config.TOP_N),
    }


def _etapa1() -> dict[str, Any]:
    """
    O texto das issues da etapa 1, FRASE A FRASE: a mesma unidade da fala.

    A etapa 1 mediu a issue inteira (mediana de 311 palavras), e texto longo
    empurra a nota do VADER para os extremos: só 4% das issues saíram
    neutras, contra 41% das frases. Comparar a issue inteira com uma frase
    falada invertia a conclusão (ver o diário de bordo). As ferramentas
    também são contadas por frase, com as regras da escrita.
    """
    if not config.ARQ_ETAPA1_CORPUS.exists():
        return {}
    with config.ARQ_ETAPA1_CORPUS.open(encoding="utf-8") as arquivo:
        corpus = json.load(arquivo)
    frases = [
        f.strip()
        for doc in corpus
        for f in RE_FRASE.split(f"{doc.get('titulo', '')}. {doc.get('texto_limpo', '')}")
        if f.strip()
    ]
    vader = _vader()
    # Mesma regra da fala: frase com menos de 6 palavras não sustenta sentimento.
    notas = [vader.polarity_scores(f)["compound"] for f in frases if len(f.split()) >= config.MIN_PALAVRAS_FRASE]
    ferramentas = Counter(nome for f in frases for nome, padrao in REGEX_ESCRITA.items() if padrao.search(f))
    por_issue = [doc["compound"] for doc in corpus if "compound" in doc]
    return {
        "documentos": len(corpus),
        "frases": len(frases),
        "frases_avaliadas": len(notas),
        "ferramentas": dict(ferramentas.most_common()),
        "cursor_sem_correcao": sum(bool(RE_CURSOR_DICIONARIO.search(f)) for f in frases),
        "distribuicao": dict(Counter(classificar(n) for n in notas)),
        "media": round(float(np.mean(notas)), 4) if notas else None,
        # Como a etapa 1 mediu (uma nota por issue), só para referência.
        "por_issue": {
            "distribuicao": dict(Counter(doc["sentimento"] for doc in corpus if doc.get("sentimento"))),
            "media": round(float(np.mean(por_issue)), 4) if por_issue else None,
        },
    }


def analisar(
    episodios: list[dict[str, Any]],
    transcricoes: dict[str, dict[str, Any]],
    acustica: dict[str, Any],
    salvar: bool = True,
) -> dict[str, Any]:
    """Junta transcrição, voz e sentimento de todos os episódios."""
    por_episodio: dict[str, Any] = {}
    frases_todas: list[dict[str, Any]] = []
    textos: list[str] = []
    mencoes: Counter[str] = Counter()
    episodios_que_citam: Counter[str] = Counter()

    for episodio in episodios:
        transcricao = transcricoes.get(episodio["id"])
        if not transcricao:
            continue
        curvas = acustica.get("curvas", {}).get(episodio["id"])
        frases = _frases_do_episodio(transcricao, curvas)
        avaliadas = [f for f in frases if f["avaliado"]]
        palavras = sum(f["palavras"] for f in frases)
        minutos_fala = sum(f["fim"] - f["inicio"] for f in frases) / 60
        citadas = Counter(nome for f in frases for nome in f["ferramentas"])

        mencoes.update(citadas)
        episodios_que_citam.update(citadas.keys())
        textos.append(" ".join(s["texto"] for s in transcricao["segmentos"]))
        for f in frases:
            f["episodio"] = episodio["id"]
        frases_todas.extend(frases)

        distribuicao = Counter(classificar(f["compound"]) for f in avaliadas)
        por_episodio[episodio["id"]] = {
            "podcast": episodio["podcast"],
            "titulo": episodio["titulo"],
            "palavras": palavras,
            "minutos_fala": round(minutos_fala, 1),
            "wpm": round(palavras / max(minutos_fala, 1e-9), 1),
            "trechos_whisper": len(transcricao["segmentos"]),
            "frases": len(frases),
            "ferramentas": dict(citadas.most_common()),
            "media_compound": round(float(np.mean([f["compound"] for f in avaliadas])), 4) if avaliadas else None,
            "distribuicao": dict(distribuicao),
            "acustica": acustica.get("episodios", {}).get(episodio["id"], {}),
        }
        print(
            f"  {episodio['podcast'][:32]:<32} {palavras:>6} palavras  "
            f"{por_episodio[episodio['id']]['wpm']:>5.0f} WPM  "
            f"ferramentas: {', '.join(citadas) or '-'}"
        )

    avaliadas = [f for f in frases_todas if f["avaliado"]]
    distribuicao = Counter(classificar(f["compound"]) for f in avaliadas)

    # Voz x sentimento: só as frases com sentimento avaliado e com voz medida.
    voz = {}
    for chave in ("energia_db_z", "f0_hz_z"):
        pares = [(f["compound"], f[chave]) for f in avaliadas if f.get(chave) is not None]
        por_classe = defaultdict(list)
        for compound, z in pares:
            por_classe[classificar(compound)].append(z)
        voz[chave] = {
            "spearman": _spearman([p[0] for p in pares], [p[1] for p in pares]),
            "media_por_classe": {c: round(float(np.mean(v)), 3) for c, v in por_classe.items()},
        }

    exemplos = sorted(
        (f["compound"], f["texto"], por_episodio[f["episodio"]]["podcast"])
        for f in avaliadas
        if 10 <= f["palavras"] <= 40 and abs(f["compound"]) >= 0.6
    )

    resultado = {
        "estatisticas": {
            "episodios": len(por_episodio),
            "podcasts": len({e["podcast"] for e in por_episodio.values()}),
            "horas_audio": round(sum(t["duracao_s"] for t in transcricoes.values()) / 3600, 2),
            "palavras": sum(e["palavras"] for e in por_episodio.values()),
            "trechos_whisper": sum(e["trechos_whisper"] for e in por_episodio.values()),
            "frases": len(frases_todas),
            "frases_avaliadas": len(avaliadas),
            "wpm_medio": round(float(np.mean([e["wpm"] for e in por_episodio.values()])), 1) if por_episodio else 0,
        },
        "ferramentas": {
            "mencoes": dict(mencoes.most_common()),
            "episodios_que_citam": dict(episodios_que_citam.most_common()),
        },
        "sentimento": {
            "distribuicao": dict(distribuicao),
            "media": round(float(np.mean([f["compound"] for f in avaliadas])), 4) if avaliadas else None,
        },
        "voz_x_sentimento": voz,
        "vocabulario": _vocabulario(textos) if textos else {},
        "etapa1": _etapa1(),
        "por_episodio": por_episodio,
        # Só números por frase: o texto completo da transcrição fica fora do
        # git (direito autoral), em dados/brutos/transcricoes.
        "frases": [
            {k: f[k] for k in ("episodio", "inicio", "fim", "compound", "avaliado", "ferramentas", "energia_db_z", "f0_hz_z")}
            for f in frases_todas
        ],
        "exemplos_negativos": [{"nota": round(n, 3), "texto": t, "podcast": p} for n, t, p in exemplos[:5]],
        "exemplos_positivos": [{"nota": round(n, 3), "texto": t, "podcast": p} for n, t, p in exemplos[-5:][::-1]],
    }

    if salvar:
        with config.ARQ_FALA.open("w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, ensure_ascii=False)
        print(f"  salvo em {config.ARQ_FALA.name}")
    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_FALA.exists():
        return {}
    with config.ARQ_FALA.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


# Como cada número aparece no terminal, em vez do nome da chave do JSON.
ROTULOS = {
    "episodios": "episódios",
    "podcasts": "podcasts",
    "horas_audio": "horas de áudio",
    "palavras": "palavras faladas",
    "trechos_whisper": "trechos do Whisper",
    "frases": "frases",
    "frases_avaliadas": "frases avaliadas pelo VADER",
    "wpm_medio": "palavras por minuto (média)",
}


def imprimir(analise: dict[str, Any]) -> None:
    """Mostra o resultado no terminal, formatado para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção de conteúdo primeiro.")
        return
    est = analise["estatisticas"]
    print("\n  A FALA EM NÚMEROS")
    print("  " + "-" * 50)
    for chave, valor in est.items():
        print(f"    {ROTULOS.get(chave, chave.replace('_', ' ')):.<30} {valor}")

    total_texto = sum(analise["etapa1"].get("ferramentas", {}).values()) or 1
    total_fala = sum(analise["ferramentas"]["mencoes"].values()) or 1
    print("\n  FERRAMENTAS: PARTICIPAÇÃO NAS MENÇÕES (fala x texto da etapa 1)")
    print("  " + "-" * 50)
    print(f"    {'ferramenta':<18}{'fala':>8}{'texto':>9}")
    for nome, qtd in analise["ferramentas"]["mencoes"].items():
        texto = analise["etapa1"].get("ferramentas", {}).get(nome, 0)
        print(f"    {nome:<18}{qtd / total_fala * 100:>7.1f}%{texto / total_texto * 100:>8.1f}%")
    if "cursor_sem_correcao" in analise["etapa1"]:
        print(
            f"    (Cursor no texto: {analise['etapa1']['cursor_sem_correcao']} frases com o dicionário da "
            f"etapa 1, {analise['etapa1']['ferramentas'].get('Cursor', 0)} contando o Cursor escrito sozinho)"
        )

    print(f"\n  SENTIMENTO (frase a frase nas duas mídias, {config.MIN_PALAVRAS_FRASE} palavras ou mais)")
    print("  " + "-" * 50)
    dist = analise["sentimento"]["distribuicao"]
    total = sum(dist.values()) or 1
    texto_dist = analise["etapa1"].get("distribuicao", {})
    total_t = sum(texto_dist.values()) or 1
    for rotulo in ("positivo", "neutro", "negativo"):
        print(
            f"    {rotulo:<10} fala {dist.get(rotulo, 0) / total * 100:>5.1f}%   "
            f"texto {texto_dist.get(rotulo, 0) / total_t * 100:>5.1f}%"
        )
    print(f"    compound médio: fala {analise['sentimento']['media']:+.3f}   texto {analise['etapa1'].get('media', 0):+.3f}")
    por_issue = analise["etapa1"].get("por_issue")
    if por_issue:
        dist_i = por_issue["distribuicao"]
        total_i = sum(dist_i.values()) or 1
        print(
            "    (issue inteira, como a etapa 1 mediu: "
            + ", ".join(f"{r} {dist_i.get(r, 0) / total_i * 100:.0f}%" for r in ("positivo", "neutro", "negativo"))
            + ". Texto longo empurra o VADER para os extremos.)"
        )

    print("\n  A VOZ MUDA COM O SENTIMENTO? (escore z dentro do episódio)")
    print("  " + "-" * 50)
    for chave, rotulo in (("energia_db_z", "energia"), ("f0_hz_z", "altura (F0)")):
        dados = analise["voz_x_sentimento"][chave]
        rho = dados["spearman"]
        medias = dados["media_por_classe"]
        print(
            f"    {rotulo:<12} rho = {rho['rho']:+.3f} (n={rho['n']})   "
            f"pos {medias.get('positivo', 0):+.2f}  neu {medias.get('neutro', 0):+.2f}  neg {medias.get('negativo', 0):+.2f}"
        )

    print("\n  FRASES MAIS NEGATIVAS")
    for item in analise["exemplos_negativos"][:3]:
        print(f"    [{item['nota']:+.2f}] {item['texto'][:140]}")
        print(f"           {item['podcast']}")
    print("\n  FRASES MAIS POSITIVAS")
    for item in analise["exemplos_positivos"][:3]:
        print(f"    [{item['nota']:+.2f}] {item['texto'][:140]}")
        print(f"           {item['podcast']}")
    print()


def buscar(padrao: re.Pattern[str], episodios: list[dict[str, Any]], transcricoes: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Busca por regex na fala transcrita, com a marca de tempo de cada achado.

    É a KWIC da etapa 1 com uma diferença: como cada segmento tem tempo, dá
    para voltar ao áudio e OUVIR o momento exato em que a palavra foi dita.
    """
    nomes = {e["id"]: e for e in episodios}
    achados = []
    for episodio_id, transcricao in transcricoes.items():
        for s in transcricao["segmentos"]:
            casamento = padrao.search(s["texto"])
            if casamento:
                achados.append(
                    {
                        "episodio": nomes[episodio_id],
                        "inicio": s["inicio"],
                        "fim": s["fim"],
                        "texto": s["texto"],
                        "casou": casamento.group(0),
                    }
                )
    return achados


if __name__ == "__main__":
    import coleta
    import sinal
    import transcricao

    eps = coleta.selecionar()
    imprimir(analisar(eps, transcricao.carregar_transcricoes(eps), sinal.carregar_analise()))
