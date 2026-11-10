"""
Etapa 3 - CONTEÚDO DA PALESTRA: tela, voz, fala e legenda cruzadas pelo tempo.

O quadros.py diz QUANDO a tela muda; o sinal.py diz COMO a voz soa; o
leitura.py diz O QUE a tela mostra e O QUE a palestra diz. Este módulo junta
tudo pelo relógio do vídeo e responde cinco perguntas:

1. Que tela as palestras sobre IA mostram: slide (de texto ou visual) ou
   captura de ferramenta (código ou interface)? Quanto tempo a tela fica em
   movimento (demo ao vivo, rolagem)?
2. As ferramentas de IA aparecem mais na tela ou na fala? A legenda oficial
   (sem lista de vocabulário) perde nomes que a nossa transcrição acha?
3. A fala e a voz mudam junto com a tela? Velocidade (WPM) e sentimento da
   fala, pausas, energia e altura da voz enquanto a tela mostra cada tipo.
4. A voz muda quando a frase é positiva ou negativa?
5. Painel final: o texto escrito nas issues do GitHub (etapa 1) e a fala
   das palestras lado a lado, contados com as mesmas regras.

As técnicas de texto são as da etapa 1: regex para ferramentas, spaCy para
o vocabulário e VADER para o sentimento, medido frase a frase.

Vocabulário novo deste módulo:
- Palavras por minuto (WPM): velocidade da fala. Conversa em inglês fica
  perto de 150 WPM.
- Escore z: quantos desvios-padrão um valor está acima da média da própria
  palestra. Cada sala tem um microfone e cada pessoa tem uma altura de voz.
  Comparar a energia crua misturaria isso com o momento da palestra. O
  escore z responde "a voz está mais alta que o normal DESTA palestra?".
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
import leitura

TIPOS = list(config.PALETA_TELA)  # texto, visual, código, interface, movimento

# ---------------------------------------------------------------------------
# Regex de ferramentas e sentimento: as mesmas da etapa 1
# ---------------------------------------------------------------------------


def _regex_de_vocabulario(termos: list[str], extra: str | None = None) -> re.Pattern[str]:
    """
    Variações da mais longa para a mais curta: a alternativa `|` para na
    primeira que casa. `extra` é uma regex pronta somada às variações.
    """
    alternativas = [re.escape(t) for t in sorted(termos, key=len, reverse=True)] + ([extra] if extra else [])
    return re.compile(rf"\b(?:{'|'.join(alternativas)})\b", re.IGNORECASE)


# Fala e legenda são reconhecimento de fala: o Cursor sozinho conta com
# qualquer caixa (config.CURSOR_FALA). Na tela, ver leitura.ferramentas_na_tela.
REGEX_FERRAMENTAS = {
    nome: _regex_de_vocabulario(v, config.CURSOR_FALA if nome == "Cursor" else None)
    for nome, v in config.FERRAMENTAS_IA.items()
}
RE_QUALQUER_FERRAMENTA = _regex_de_vocabulario(
    [v for vs in config.FERRAMENTAS_IA.values() for v in vs], config.CURSOR_FALA
)
# No texto escrito das issues, o Cursor sozinho só conta com a maiúscula
# (config.CURSOR_ESCRITA). O `(?-i:...)` desliga o IGNORECASE só ali dentro.
REGEX_ESCRITA = {
    nome: _regex_de_vocabulario(v, f"(?-i:{config.CURSOR_ESCRITA})" if nome == "Cursor" else None)
    for nome, v in config.FERRAMENTAS_IA.items()
}

# Fim de frase: ponto, exclamação ou interrogação seguidos de espaço, ou
# quebra de linha. A mesma regra corta a fala e o texto das issues.
RE_FRASE = re.compile(r"(?<=[.!?])\s+|\n+")


def ferramentas_em(texto: str) -> list[str]:
    return [nome for nome, padrao in REGEX_FERRAMENTAS.items() if padrao.search(texto)]


def ocorrencias(texto: str) -> Counter[str]:
    """Quantas vezes cada ferramenta foi escrita no texto. A ferramentas_em só diz quais apareceram."""
    return Counter({nome: len(p.findall(texto)) for nome, p in REGEX_FERRAMENTAS.items() if p.search(texto)})


_VADER = None


def _vader():
    """O VADER da etapa 1, sem as palavras de config.SEM_SENTIMENTO (diário de bordo, entrada 10)."""
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
# Voz: escore z e correlação
# ---------------------------------------------------------------------------


def _escore_z(valores: list[float | None]) -> list[float | None]:
    """Escore z de cada valor, com a média e o desvio dos valores presentes da lista."""
    presentes = np.array([v for v in valores if v is not None], dtype=float)
    if presentes.size < 2 or presentes.std() == 0:
        return [None] * len(valores)
    media, desvio = presentes.mean(), presentes.std()
    return [None if v is None else round(float((v - media) / desvio), 3) for v in valores]


def _spearman(x: list[float], y: list[float]) -> dict[str, float] | None:
    if len(x) < 10:
        return None
    from scipy.stats import spearmanr

    resultado = spearmanr(x, y)
    return {"rho": round(float(resultado.statistic), 3), "p": float(resultado.pvalue), "n": len(x)}


def _voz_da_frase(curvas: dict[str, list], inicio: float, fim: float) -> tuple[float | None, float | None]:
    """Energia média e F0 mediana dos segundos que a frase ocupa (só os segundos com medida)."""
    primeiro, ultimo = int(inicio), max(int(np.ceil(fim)), int(inicio) + 1)
    energia = [e for e in curvas["energia_db"][primeiro:ultimo] if e is not None]
    alturas = [f for f in curvas["f0_hz"][primeiro:ultimo] if f is not None]
    return (
        float(np.mean(energia)) if energia else None,
        float(np.median(alturas)) if alturas else None,
    )


# ---------------------------------------------------------------------------
# Cruzamento pelo tempo
# ---------------------------------------------------------------------------


def tipo_por_segundo(visual: dict[str, Any], lidas: list[dict[str, Any]]) -> list[str | None]:
    """
    O tipo de tela em cada segundo do vídeo.

    Segundo dentro de uma tela recebe o tipo que o OCR deu a ela. Segundo
    válido fora de qualquer tela é "movimento" (a tela não parou). Segundo
    fora da moldura padrão fica None e não entra em conta nenhuma.
    """
    tipos = {l["numero"]: l["tipo"] for l in lidas}
    por_segundo: list[str | None] = ["movimento" if v else None for v in visual["serie"]["valido"]]
    for tela in visual["telas"]:
        for segundo in range(int(tela["inicio"]), min(int(tela["fim"]), len(por_segundo))):
            por_segundo[segundo] = tipos.get(tela["numero"], "texto")
    return por_segundo


def voz_por_tipo(por_segundo: list[str | None], curvas: dict[str, list], pausas: list[list[float]]) -> dict[str, dict[str, Any]]:
    """
    A voz de uma palestra enquanto a tela mostra cada tipo (somas, para juntar depois).

    Energia e F0 de cada segundo viram escore z dentro da palestra, contra
    os segundos válidos dela (dentro da moldura padrão). Cada pausa longa
    entra no tipo de tela do seu segundo do meio, como a frase.
    """
    n = min(len(por_segundo), len(curvas["energia_db"]))
    tipos = por_segundo[:n]
    energia = _escore_z([e if t else None for e, t in zip(curvas["energia_db"][:n], tipos)])
    f0 = _escore_z([f if t else None for f, t in zip(curvas["f0_hz"][:n], tipos)])
    somas = {tipo: {"segundos": 0, "pausas": 0, "energia_z": [], "f0_z": []} for tipo in TIPOS}
    for tipo, e, f in zip(tipos, energia, f0):
        if tipo is None:
            continue
        somas[tipo]["segundos"] += 1
        if e is not None:
            somas[tipo]["energia_z"].append(e)
        if f is not None:
            somas[tipo]["f0_z"].append(f)
    for inicio, fim in pausas:
        meio = int((inicio + fim) / 2)
        if meio < n and tipos[meio]:
            somas[tipos[meio]]["pausas"] += 1
    return somas


def _resumo_voz(somas: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Pausas por minuto e escore z médio de cada tipo de tela.

    Tipo com menos de 1 minuto de tela fica sem taxa de pausas: 1 pausa em
    20 segundos daria "3 por minuto" e pareceria um achado.
    """
    return {
        tipo: {
            "minutos": round(s["segundos"] / 60, 1),
            "pausas": s["pausas"],
            "pausas_por_min": round(s["pausas"] / (s["segundos"] / 60), 2) if s["segundos"] >= 60 else None,
            "energia_z": round(float(np.mean(s["energia_z"])), 3) if s["energia_z"] else None,
            "f0_z": round(float(np.mean(s["f0_z"])), 3) if s["f0_z"] else None,
        }
        for tipo, s in somas.items()
    }


def _frases(segmento: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Corta um trecho do Whisper em frases, cada uma com o seu tempo.

    Nas palestras o trecho costuma ser curto, mas às vezes passa de 40
    palavras, e texto longo empurra o VADER para os extremos. O Whisper só
    dá o tempo do trecho inteiro, então o de cada frase é estimado pela
    posição dos caracteres: num trecho de 10 s e 100 caracteres, a frase
    que vai do caractere 40 ao 80 fica com os segundos 4 a 8.
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


def _frases_da_palestra(
    transcricao: dict[str, Any], por_segundo: list[str | None], curvas: dict[str, list] | None
) -> list[dict[str, Any]]:
    """
    Anota cada frase falada com palavras, sentimento, ferramentas, o tipo de
    tela no meio dela e a voz (energia e F0 em escore z dentro da palestra).
    """
    vader = _vader()
    anotadas = []
    for frase in (f for segmento in transcricao["segmentos"] for f in _frases(segmento)):
        palavras = len(frase["texto"].split())
        meio = int((frase["inicio"] + frase["fim"]) / 2)
        energia, f0 = _voz_da_frase(curvas, frase["inicio"], frase["fim"]) if curvas else (None, None)
        anotadas.append({
            **frase,
            "palavras": palavras,
            "compound": round(vader.polarity_scores(frase["texto"])["compound"], 4),
            "avaliado": palavras >= config.MIN_PALAVRAS_FRASE,
            "ferramentas": ferramentas_em(frase["texto"]),
            "tela": por_segundo[meio] if meio < len(por_segundo) else None,
            "energia_db": energia,
            "f0_hz": f0,
        })
    for chave in ("energia_db", "f0_hz"):
        for frase, z in zip(anotadas, _escore_z([a[chave] for a in anotadas])):
            frase[f"{chave}_z"] = z
    return anotadas


def _texto_na_janela(itens: list[dict[str, Any]], inicio: float, fim: float, folga: float = 2.0) -> str:
    return " ".join(i["texto"] for i in itens if i["fim"] > inicio - folga and i["inicio"] < fim + folga)


def _legenda_mais_proxima(legenda: list[dict[str, Any]], inicio: float, fim: float) -> str:
    """A linha da legenda que mais se sobrepõe ao intervalo: é a que aparece no exemplo."""
    if not legenda:
        return ""
    return max(legenda, key=lambda i: min(i["fim"], fim) - max(i["inicio"], inicio))["texto"]


def comparar_legenda(frases: list[dict[str, Any]], legenda: list[dict[str, Any]], titulo: str) -> dict[str, Any]:
    """
    Nossa transcrição x legenda oficial, ferramenta por ferramenta.

    Ganho: a nossa frase cita a ferramenta e a legenda, no mesmo intervalo
    de tempo (com 2 s de folga), não cita. Perda: o contrário. A perda
    existe porque a lista de vocabulário também pode inventar um nome.
    """
    ganhos: Counter[str] = Counter()
    perdas: Counter[str] = Counter()
    exemplos = []
    for f in frases:
        oficial = _texto_na_janela(legenda, f["inicio"], f["fim"])
        for nome in f["ferramentas"]:
            if nome not in ferramentas_em(oficial):
                ganhos[nome] += 1
                if len(exemplos) < 6:
                    exemplos.append({"palestra": titulo, "inicio": f["inicio"], "nosso": f["texto"],
                                     "oficial": _legenda_mais_proxima(legenda, f["inicio"], f["fim"])})
    for item in legenda:
        nosso = _texto_na_janela(frases, item["inicio"], item["fim"])
        for nome in ferramentas_em(item["texto"]):
            if nome not in ferramentas_em(nosso):
                perdas[nome] += 1
    return {"ganhos": dict(ganhos), "perdas": dict(perdas), "exemplos": exemplos}


# ---------------------------------------------------------------------------
# Vocabulário e o texto das issues
# ---------------------------------------------------------------------------


def _vocabulario(textos: list[str]) -> list[tuple[str, int]]:
    """Lemas com spaCy, com o filtro da etapa 1 e a correção de config.CORRECAO_LEMA."""
    import spacy

    nlp = spacy.load(config.MODELO_SPACY, disable=["ner", "parser"])
    lemas: Counter[str] = Counter()
    for doc in nlp.pipe(textos, batch_size=20):
        lemas.update(
            config.CORRECAO_LEMA.get(t.lemma_.lower(), t.lemma_.lower())
            for t in doc
            if not (t.is_stop or t.is_punct or t.like_num)
            and t.pos_ in config.POS_RELEVANTES
            and len(t.lemma_) >= 3
            and t.lemma_.isalpha()
            and t.lemma_.lower() not in config.STOPWORDS_EXTRA
        )
    return lemas.most_common(config.TOP_N * 3)


def _issues() -> dict[str, Any]:
    """
    O texto das issues do GitHub (etapa 1), FRASE A FRASE: a mesma unidade da fala.

    A etapa 1 deu uma nota para cada issue inteira (mediana de 311 palavras).
    O compound do VADER soma a nota de todas as palavras e espreme a soma
    entre -1 e +1, então texto longo vai para perto dos extremos: só 4% das
    issues saíram neutras. Para comparar com a fala, o texto é cortado em
    frases pela mesma regra (RE_FRASE), com o mesmo mínimo de palavras, e as
    ferramentas são contadas por frase, com a regra da escrita para o Cursor.
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
    notas = [vader.polarity_scores(f)["compound"] for f in frases if len(f.split()) >= config.MIN_PALAVRAS_FRASE]
    ferramentas = Counter(nome for f in frases for nome, padrao in REGEX_ESCRITA.items() if padrao.search(f))
    return {
        "volume": f"{len(corpus)} issues",
        "frases": len(frases),
        "frases_avaliadas": len(notas),
        "ferramentas": dict(ferramentas.most_common()),
        "distribuicao": dict(Counter(classificar(n) for n in notas)),
        "media": round(float(np.mean(notas)), 4) if notas else None,
    }


def _voz_x_sentimento(frases: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman entre o sentimento da frase e a voz, e a voz média de cada classe de sentimento."""
    avaliadas = [f for f in frases if f["avaliado"]]
    resultado = {}
    for chave in ("energia_db_z", "f0_hz_z"):
        pares = [(f["compound"], f[chave]) for f in avaliadas if f.get(chave) is not None]
        por_classe = defaultdict(list)
        for compound, z in pares:
            por_classe[classificar(compound)].append(z)
        resultado[chave] = {
            "spearman": _spearman([p[0] for p in pares], [p[1] for p in pares]),
            "media_por_classe": {c: round(float(np.mean(v)), 3) for c, v in por_classe.items()},
        }
    return resultado


# ---------------------------------------------------------------------------
# Análise principal
# ---------------------------------------------------------------------------


def _distribuicao(frases: list[dict[str, Any]]) -> dict[str, Any]:
    avaliadas = [f for f in frases if f["avaliado"]]
    return {
        "distribuicao": dict(Counter(classificar(f["compound"]) for f in avaliadas)),
        "media": round(float(np.mean([f["compound"] for f in avaliadas])), 4) if avaliadas else None,
        "n": len(avaliadas),
    }


def _wpm(frases: list[dict[str, Any]]) -> float | None:
    minutos = sum(f["fim"] - f["inicio"] for f in frases) / 60
    return round(sum(f["palavras"] for f in frases) / minutos, 1) if minutos > 0 else None


def analisar(
    palestras: list[dict[str, Any]],
    visual: dict[str, Any],
    lidas: dict[str, Any],
    transcricoes: dict[str, Any],
    acustica: dict[str, Any],
    salvar: bool = True,
) -> dict[str, Any]:
    """Cruza tela, voz, fala e legenda de todas as palestras já processadas."""
    por_palestra: dict[str, Any] = {}
    todas: list[dict[str, Any]] = []
    tempo_por_tipo: Counter[str] = Counter()
    na_tela: Counter[str] = Counter()
    na_fala: Counter[str] = Counter()
    na_legenda: Counter[str] = Counter()
    ganhos: Counter[str] = Counter()
    perdas: Counter[str] = Counter()
    exemplos: list[dict[str, Any]] = []
    textos_fala: list[str] = []
    textos_tela: list[str] = []
    voz_total = {tipo: {"segundos": 0, "pausas": 0, "energia_z": [], "f0_z": []} for tipo in TIPOS}

    for palestra in palestras:
        pid = palestra["id"]
        v, l, t = visual.get("palestras", {}).get(pid), lidas.get(pid), transcricoes.get(pid)
        if not (v and l and t):
            continue
        som = acustica.get("palestras", {}).get(pid)
        por_segundo = tipo_por_segundo(v, l)
        frases = _frases_da_palestra(t, por_segundo, som["curvas"] if som else None)
        voz = voz_por_tipo(por_segundo, som["curvas"], som["pausas"]) if som else None
        if voz:
            for tipo, s in voz.items():
                for chave in ("segundos", "pausas"):
                    voz_total[tipo][chave] += s[chave]
                for chave in ("energia_z", "f0_z"):
                    voz_total[tipo][chave].extend(s[chave])
        legenda = leitura.ler_legenda(palestra)
        comparacao = comparar_legenda(frases, legenda, palestra["titulo"])

        texto_fala = " ".join(f["texto"] for f in frases)
        texto_legenda = " ".join(i["texto"] for i in legenda)
        tela = Counter(f for tela in l for f in tela["ferramentas"])
        tipos = Counter(s for s in por_segundo if s)
        validos = sum(tipos.values()) or 1

        tempo_por_tipo.update(tipos)
        na_tela.update(tela)
        na_fala.update(ocorrencias(texto_fala))
        na_legenda.update(ocorrencias(texto_legenda))
        ganhos.update(comparacao["ganhos"])
        perdas.update(comparacao["perdas"])
        exemplos.extend(comparacao["exemplos"])
        textos_fala.append(texto_fala)
        textos_tela.append(" ".join(linha for tela in l for linha, _ in tela["linhas"]))
        for f in frases:
            f["palestra"] = pid
        todas.extend(frases)

        por_palestra[pid] = {
            "titulo": palestra["titulo"],
            "ano": palestra["ano"],
            "trilha": palestra["trilha"],
            "visual": v["resumo"],
            "tempo_por_tipo": {tipo: round(tipos.get(tipo, 0) / validos * 100, 1) for tipo in TIPOS},
            "palavras": sum(f["palavras"] for f in frases),
            "wpm": _wpm(frases),
            "wpm_por_tipo": {tipo: _wpm([f for f in frases if f["tela"] == tipo]) for tipo in TIPOS},
            "sentimento": _distribuicao(frases),
            "ferramentas": {
                "tela": dict(tela),
                "fala": dict(ocorrencias(texto_fala)),
                "legenda": dict(ocorrencias(texto_legenda)),
            },
            "acustica": som["resumo"] if som else {},
            "voz_por_tipo": _resumo_voz(voz) if voz else {},
        }
        tempo = por_palestra[pid]["tempo_por_tipo"]
        print(
            f"  {palestra['titulo'][:36]:<36} {por_palestra[pid]['wpm'] or 0:>4.0f} WPM  "
            f"código {tempo['código']:>4.1f}%  interface {tempo['interface']:>4.1f}%  "
            f"tela: {', '.join(tela) or '-'}  fala: {', '.join(ocorrencias(texto_fala)) or '-'}"
        )

    validos = sum(tempo_por_tipo.values()) or 1
    # Os exemplos da palestra da demo vêm primeiro: são os que a turma vê ao vivo na opção 7.
    titulo_demo = por_palestra.get(config.DEMO_PALESTRA, {}).get("titulo")
    exemplos.sort(key=lambda e: e["palestra"] != titulo_demo)
    extremas = sorted(
        (f["compound"], f["texto"], por_palestra[f["palestra"]]["titulo"])
        for f in todas
        if 10 <= f["palavras"] <= 40 and abs(f["compound"]) >= 0.6
    )
    nomes = sorted(set(na_tela) | set(na_fala) | set(na_legenda), key=lambda n: -(na_tela[n] + na_fala[n] + na_legenda[n]))
    com_som = [p["acustica"] for p in por_palestra.values() if p["acustica"]]

    resultado = {
        "estatisticas": {
            "palestras": len(por_palestra),
            "horas_video": round(sum(p["visual"]["duracao_s"] for p in por_palestra.values()) / 3600, 2),
            "telas": sum(p["visual"]["telas"] for p in por_palestra.values()),
            "palavras_fala": sum(p["palavras"] for p in por_palestra.values()),
            "palavras_tela": sum(len(t.split()) for t in textos_tela),
            "trechos_whisper": sum(len(transcricoes[pid]["segmentos"]) for pid in por_palestra),
            "frases": len(todas),
            "frases_avaliadas": sum(f["avaliado"] for f in todas),
            "wpm_medio": round(float(np.mean([p["wpm"] for p in por_palestra.values() if p["wpm"]])), 1) if por_palestra else 0,
            "rosto_pct_medio": round(float(np.mean([p["visual"]["rosto_pct"] for p in por_palestra.values()])), 1) if por_palestra else 0,
            "silencio_pct_medio": round(float(np.mean([a["silencio_pct"] for a in com_som])), 1) if com_som else None,
            "pausas_por_min_medio": round(float(np.mean([a["pausas_por_min"] for a in com_som])), 2) if com_som else None,
        },
        "tipos": {
            "tempo_pct": {tipo: round(tempo_por_tipo.get(tipo, 0) / validos * 100, 1) for tipo in TIPOS},
            "wpm": {tipo: _wpm([f for f in todas if f["tela"] == tipo]) for tipo in TIPOS},
            "sentimento": {tipo: _distribuicao([f for f in todas if f["tela"] == tipo]) for tipo in TIPOS},
            "voz": _resumo_voz(voz_total) if com_som else {},
        },
        "voz_x_sentimento": _voz_x_sentimento(todas) if com_som else {},
        "ferramentas": {
            "por_fonte": {n: {"tela": na_tela[n], "fala": na_fala[n], "legenda": na_legenda[n]} for n in nomes},
            # Frases que citam cada ferramenta: a mesma unidade do texto das issues no painel final.
            "mencoes_fala": dict(Counter(nome for f in todas for nome in f["ferramentas"]).most_common()),
            "legenda": {"ganhos": dict(ganhos), "perdas": dict(perdas), "exemplos": exemplos[:12]},
        },
        "sentimento": _distribuicao(todas),
        "vocabulario": {
            "fala": _vocabulario(textos_fala) if textos_fala else [],
            "tela": _vocabulario(textos_tela) if textos_tela else [],
        },
        "painel_final": {},
        "por_palestra": por_palestra,
        # Só números por frase: o texto completo fica em dados/brutos.
        "frases": [
            {k: f[k] for k in ("palestra", "inicio", "fim", "palavras", "compound", "avaliado", "ferramentas", "tela",
                               "energia_db_z", "f0_hz_z")}
            for f in todas
        ],
        "exemplos_negativos": [{"nota": round(n, 3), "texto": t, "palestra": p} for n, t, p in extremas[:5]],
        "exemplos_positivos": [{"nota": round(n, 3), "texto": t, "palestra": p} for n, t, p in extremas[-5:][::-1]],
    }
    resultado["painel_final"] = {
        "issues": _issues(),
        "palestras": {
            "volume": f"{resultado['estatisticas']['palestras']} palestras, {resultado['estatisticas']['horas_video']:.1f} h".replace(".", ","),
            "frases_avaliadas": resultado["estatisticas"]["frases_avaliadas"],
            "ferramentas": resultado["ferramentas"]["mencoes_fala"],
            "distribuicao": resultado["sentimento"]["distribuicao"],
            "media": resultado["sentimento"]["media"],
        },
    }

    if salvar:
        with config.ARQ_VIDEO.open("w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, ensure_ascii=False)
        print(f"  salvo em {config.ARQ_VIDEO.name}")
    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_VIDEO.exists():
        return {}
    with config.ARQ_VIDEO.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _pct(contagem: dict[str, int], chave: str) -> float:
    return contagem.get(chave, 0) / (sum(contagem.values()) or 1) * 100


# Como cada número aparece no terminal, em vez do nome da chave do JSON.
ROTULOS = {
    "palestras": "palestras",
    "horas_video": "horas de vídeo",
    "telas": "telas (slides e capturas)",
    "palavras_fala": "palavras faladas",
    "palavras_tela": "palavras escritas na tela",
    "trechos_whisper": "trechos do Whisper",
    "frases": "frases faladas",
    "frases_avaliadas": "frases avaliadas pelo VADER",
    "wpm_medio": "palavras por minuto (média)",
    "rosto_pct_medio": "rosto na câmera (% do tempo)",
    "silencio_pct_medio": "silêncio (% do tempo, média)",
    "pausas_por_min_medio": "pausas longas por min (média)",
}


def _z(valor: float | None) -> str:
    return f"{valor:+.2f}" if valor is not None else "-"


def imprimir(analise: dict[str, Any]) -> None:
    """Mostra o resultado no terminal, formatado para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção 9 primeiro.")
        return
    print("\n  ÁUDIO E VÍDEO EM NÚMEROS")
    print("  " + "-" * 56)
    for chave, valor in analise["estatisticas"].items():
        print(f"    {ROTULOS.get(chave, chave.replace('_', ' ')):.<32} {valor}")

    print("\n  O QUE A TELA MOSTRA (tempo de vídeo) E COMO A FALA ACOMPANHA")
    print("  " + "-" * 56)
    print(f"    {'tela':<11}{'tempo':>8}{'WPM':>8}{'compound':>11}")
    for tipo in TIPOS:
        sent = analise["tipos"]["sentimento"][tipo]
        wpm = analise["tipos"]["wpm"][tipo]
        media = f"{sent['media']:+.3f}" if sent["media"] is not None else "-"
        print(f"    {tipo:<11}{analise['tipos']['tempo_pct'][tipo]:>7.1f}%{wpm or 0:>8.0f}{media:>11}")

    voz = analise["tipos"].get("voz") or {}
    if voz:
        print("\n  A VOZ POR TIPO DE TELA (escore z dentro de cada palestra)")
        print("  " + "-" * 56)
        print(f"    {'tela':<11}{'minutos':>9}{'pausas/min':>12}{'energia z':>11}{'F0 z':>8}")
        for tipo in TIPOS:
            v = voz[tipo]
            taxa = f"{v['pausas_por_min']:.2f}" if v["pausas_por_min"] is not None else "-"
            print(f"    {tipo:<11}{v['minutos']:>9.1f}{taxa:>12}{_z(v['energia_z']):>11}{_z(v['f0_z']):>8}")
        print("\n    sentimento da frase x voz (Spearman, frases avaliadas):")
        for chave, rotulo in (("energia_db_z", "energia"), ("f0_hz_z", "F0")):
            item = analise["voz_x_sentimento"].get(chave) or {}
            s = item.get("spearman")
            if not s:
                continue
            classes = item["media_por_classe"]
            print(
                f"      {rotulo:<8} rho {s['rho']:+.3f} (p = {s['p']:#.2g}, {s['n']} frases)  z médio: "
                + "  ".join(f"{c} {_z(classes.get(c))}" for c in ("positivo", "neutro", "negativo"))
            )

    print("\n  FERRAMENTAS: TELA (OCR) x FALA (nosso Whisper) x LEGENDA OFICIAL")
    print("  " + "-" * 56)
    print(f"    {'ferramenta':<18}{'telas':>7}{'fala':>7}{'legenda':>9}")
    for nome, fontes in analise["ferramentas"]["por_fonte"].items():
        print(f"    {nome:<18}{fontes['tela']:>7}{fontes['fala']:>7}{fontes['legenda']:>9}")
    leg = analise["ferramentas"]["legenda"]
    print(f"\n    nomes que só a nossa transcrição escreveu: {sum(leg['ganhos'].values())} {leg['ganhos']}")
    print(f"    nomes que só a legenda oficial escreveu:   {sum(leg['perdas'].values())} {leg['perdas']}")
    for item in leg["exemplos"][:2]:
        minuto, segundo = divmod(int(item["inicio"]), 60)
        print(f"\n    [{item['palestra'][:40]}, {minuto}:{segundo:02d}]")
        print(f"      nosso:   {item['nosso'][:110]}")
        print(f"      oficial: {item['oficial'][:110]}")

    print(f"\n  O ESCRITO E O FALADO (sentimento frase a frase, {config.MIN_PALAVRAS_FRASE} palavras ou mais)")
    print("  " + "-" * 56)
    for chave, rotulo in (("issues", "issues (escrito)"), ("palestras", "palestras (falado)")):
        fonte = analise["painel_final"].get(chave) or {}
        if not fonte:
            print(f"    {rotulo:<19} (sem dados)")
            continue
        dist = fonte["distribuicao"]
        topo = max(fonte["ferramentas"], key=fonte["ferramentas"].get) if fonte["ferramentas"] else "-"
        print(
            f"    {rotulo:<19} {fonte['volume']:<22} positivo {_pct(dist, 'positivo'):>4.0f}%  "
            f"negativo {_pct(dist, 'negativo'):>4.0f}%  mais citada: {topo}"
        )
    print()


def buscar(
    padrao: re.Pattern[str],
    palestras: list[dict[str, Any]],
    transcricoes: dict[str, Any],
    lidas: dict[str, Any],
    visual: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Busca por regex na fala e no texto lido da tela, com o segundo de cada achado.

    É a KWIC da etapa 1 com duas fontes: o que foi DITO (Whisper) e o que
    estava ESCRITO no slide (OCR).
    """
    nomes = {p["id"]: p for p in palestras}
    achados = []
    for pid, transcricao in transcricoes.items():
        for s in transcricao["segmentos"]:
            casamento = padrao.search(s["texto"])
            if casamento and pid in nomes:
                achados.append({"palestra": nomes[pid], "fonte": "fala", "inicio": s["inicio"],
                                "fim": s["fim"], "texto": s["texto"], "casou": casamento.group(0)})
    for pid, telas in lidas.items():
        inicio_de = {t["numero"]: t for t in visual.get("palestras", {}).get(pid, {}).get("telas", [])}
        for tela in telas:
            texto = " ".join(linha for linha, _ in tela["linhas"])
            casamento = padrao.search(texto)
            if casamento and pid in nomes and tela["numero"] in inicio_de:
                info = inicio_de[tela["numero"]]
                achados.append({"palestra": nomes[pid], "fonte": "tela", "inicio": info["inicio"],
                                "fim": info["fim"], "texto": texto, "casou": casamento.group(0)})
    return achados


if __name__ == "__main__":
    import coleta
    import quadros
    import sinal

    eps = coleta.selecionar()
    imprimir(analisar(eps, quadros.carregar_analise(), leitura.carregar_leitura(),
                      leitura.carregar_transcricoes(eps), sinal.carregar_analise()))
