"""
Etapa 3 - CONTEÚDO DA PALESTRA: tela, fala e legenda cruzadas pelo tempo.

O quadros.py diz QUANDO a tela muda; o leitura.py diz O QUE a tela mostra e
O QUE a palestra diz. Este módulo junta as três coisas pelo relógio do
vídeo e responde quatro perguntas:

1. Que tela as palestras sobre IA mostram: slide (de texto ou visual) ou
   captura de ferramenta (código ou interface)? Quanto tempo a tela fica em
   movimento (demo ao vivo, rolagem)?
2. As ferramentas de IA aparecem mais na tela ou na fala? A legenda oficial
   (sem lista de vocabulário) perde nomes que a nossa transcrição acha?
3. A fala muda junto com a tela? Velocidade (WPM) e sentimento da fala
   enquanto a tela mostra cada tipo.
4. Painel final: texto (etapa 1), fala (etapa 2) e vídeo (etapa 3) lado a
   lado, contados com as mesmas regras.

As técnicas de texto são as das etapas 1 e 2: regex para ferramentas,
spaCy para o vocabulário e VADER para o sentimento, medido frase a frase
como na etapa 2.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from typing import Any

import numpy as np

import config
import leitura

TIPOS = list(config.PALETA_TELA)  # texto, visual, código, interface, movimento

# ---------------------------------------------------------------------------
# Regex de ferramentas e sentimento: as mesmas das etapas 1 e 2
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

# Fim de frase: ponto, exclamação ou interrogação seguidos de espaço, ou
# quebra de linha. A mesma regra da etapa 2.
RE_FRASE = re.compile(r"(?<=[.!?])\s+|\n+")


def ferramentas_em(texto: str) -> list[str]:
    return [nome for nome, padrao in REGEX_FERRAMENTAS.items() if padrao.search(texto)]


def ocorrencias(texto: str) -> Counter[str]:
    """Quantas vezes cada ferramenta foi escrita no texto. A ferramentas_em só diz quais apareceram."""
    return Counter({nome: len(p.findall(texto)) for nome, p in REGEX_FERRAMENTAS.items() if p.search(texto)})


_VADER = None


def _vader():
    """O VADER da etapa 1, sem as palavras de config.SEM_SENTIMENTO (como na etapa 2)."""
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


def _frases(segmento: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Corta um trecho do Whisper em frases, cada uma com o seu tempo (igual à etapa 2).

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


def _frases_da_palestra(transcricao: dict[str, Any], por_segundo: list[str | None]) -> list[dict[str, Any]]:
    """Anota cada frase falada com palavras, sentimento, ferramentas e o tipo de tela no meio dela."""
    vader = _vader()
    anotadas = []
    for frase in (f for segmento in transcricao["segmentos"] for f in _frases(segmento)):
        palavras = len(frase["texto"].split())
        meio = int((frase["inicio"] + frase["fim"]) / 2)
        anotadas.append({
            **frase,
            "palavras": palavras,
            "compound": round(vader.polarity_scores(frase["texto"])["compound"], 4),
            "avaliado": palavras >= config.MIN_PALAVRAS_FRASE,
            "ferramentas": ferramentas_em(frase["texto"]),
            "tela": por_segundo[meio] if meio < len(por_segundo) else None,
        })
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
# Vocabulário e etapas anteriores
# ---------------------------------------------------------------------------


def _vocabulario(textos: list[str]) -> list[tuple[str, int]]:
    """Lemas com spaCy, com o mesmo filtro das etapas 1 e 2."""
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


def _analise_etapa2() -> dict[str, Any]:
    if not config.ARQ_ETAPA2_FALA.exists():
        return {}
    with config.ARQ_ETAPA2_FALA.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _etapa1() -> dict[str, Any]:
    """
    O texto das issues do GitHub (etapa 1), frase a frase, como a etapa 2
    recontou: mesma unidade da fala e o Cursor escrito sozinho incluído
    (ver o diário de bordo da etapa 2). Lido do resultado da etapa 2, para a
    regra existir num lugar só.
    """
    texto = _analise_etapa2().get("etapa1")
    if not texto:
        return {}
    return {
        "volume": f"{texto['documentos']} issues",
        "frases_avaliadas": texto["frases_avaliadas"],
        "ferramentas": texto["ferramentas"],
        "distribuicao": texto["distribuicao"],
        "media": texto["media"],
    }


def _etapa2() -> dict[str, Any]:
    """A fala dos podcasts (etapa 2)."""
    fala = _analise_etapa2()
    if not fala:
        return {}
    est = fala["estatisticas"]
    return {
        "volume": f"{est['episodios']} episódios, {est['horas_audio']:.1f} h".replace(".", ","),
        "frases_avaliadas": est["frases_avaliadas"],
        "ferramentas": fala["ferramentas"]["mencoes"],
        "distribuicao": fala["sentimento"]["distribuicao"],
        "media": fala["sentimento"]["media"],
    }


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
    salvar: bool = True,
) -> dict[str, Any]:
    """Cruza tela, fala e legenda de todas as palestras já processadas."""
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

    for palestra in palestras:
        pid = palestra["id"]
        v, l, t = visual.get("palestras", {}).get(pid), lidas.get(pid), transcricoes.get(pid)
        if not (v and l and t):
            continue
        por_segundo = tipo_por_segundo(v, l)
        frases = _frases_da_palestra(t, por_segundo)
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
        }
        tempo = por_palestra[pid]["tempo_por_tipo"]
        print(
            f"  {palestra['titulo'][:36]:<36} {por_palestra[pid]['wpm'] or 0:>4.0f} WPM  "
            f"código {tempo['código']:>4.1f}%  interface {tempo['interface']:>4.1f}%  "
            f"tela: {', '.join(tela) or '-'}  fala: {', '.join(ocorrencias(texto_fala)) or '-'}"
        )

    validos = sum(tempo_por_tipo.values()) or 1
    # Os exemplos da palestra da demo vêm primeiro: são os que a turma vê ao vivo na opção 5.
    titulo_demo = por_palestra.get(config.DEMO_PALESTRA, {}).get("titulo")
    exemplos.sort(key=lambda e: e["palestra"] != titulo_demo)
    extremas = sorted(
        (f["compound"], f["texto"], por_palestra[f["palestra"]]["titulo"])
        for f in todas
        if 10 <= f["palavras"] <= 40 and abs(f["compound"]) >= 0.6
    )
    nomes = sorted(set(na_tela) | set(na_fala) | set(na_legenda), key=lambda n: -(na_tela[n] + na_fala[n] + na_legenda[n]))

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
        },
        "tipos": {
            "tempo_pct": {tipo: round(tempo_por_tipo.get(tipo, 0) / validos * 100, 1) for tipo in TIPOS},
            "wpm": {tipo: _wpm([f for f in todas if f["tela"] == tipo]) for tipo in TIPOS},
            "sentimento": {tipo: _distribuicao([f for f in todas if f["tela"] == tipo]) for tipo in TIPOS},
        },
        "ferramentas": {
            "por_fonte": {n: {"tela": na_tela[n], "fala": na_fala[n], "legenda": na_legenda[n]} for n in nomes},
            # Frases que citam cada ferramenta: a unidade da etapa 2, para o painel das três etapas.
            "mencoes_fala": dict(Counter(nome for f in todas for nome in f["ferramentas"]).most_common()),
            "legenda": {"ganhos": dict(ganhos), "perdas": dict(perdas), "exemplos": exemplos[:12]},
        },
        "sentimento": _distribuicao(todas),
        "vocabulario": {
            "fala": _vocabulario(textos_fala) if textos_fala else [],
            "tela": _vocabulario(textos_tela) if textos_tela else [],
        },
        "tres_etapas": {},
        "por_palestra": por_palestra,
        # Só números por frase: o texto completo fica em dados/brutos.
        "frases": [
            {k: f[k] for k in ("palestra", "inicio", "fim", "palavras", "compound", "avaliado", "ferramentas", "tela")}
            for f in todas
        ],
        "exemplos_negativos": [{"nota": round(n, 3), "texto": t, "palestra": p} for n, t, p in extremas[:5]],
        "exemplos_positivos": [{"nota": round(n, 3), "texto": t, "palestra": p} for n, t, p in extremas[-5:][::-1]],
    }
    resultado["tres_etapas"] = {
        "texto": _etapa1(),
        "fala": _etapa2(),
        "video": {
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
}


def imprimir(analise: dict[str, Any]) -> None:
    """Mostra o resultado no terminal, formatado para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção 7 primeiro.")
        return
    print("\n  O VÍDEO EM NÚMEROS")
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

    print(f"\n  AS TRÊS ETAPAS LADO A LADO (sentimento frase a frase, {config.MIN_PALAVRAS_FRASE} palavras ou mais)")
    print("  " + "-" * 56)
    for chave, rotulo in (("texto", "texto (etapa 1)"), ("fala", "fala (etapa 2)"), ("video", "vídeo (etapa 3)")):
        etapa = analise["tres_etapas"].get(chave) or {}
        if not etapa:
            print(f"    {rotulo:<17} (sem dados)")
            continue
        dist = etapa["distribuicao"]
        topo = max(etapa["ferramentas"], key=etapa["ferramentas"].get) if etapa["ferramentas"] else "-"
        print(
            f"    {rotulo:<17} {etapa['volume']:<24} positivo {_pct(dist, 'positivo'):>4.0f}%  "
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

    É a KWIC das etapas anteriores com duas fontes: o que foi DITO (Whisper)
    e o que estava ESCRITO no slide (OCR).
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

    eps = coleta.selecionar()
    imprimir(analisar(eps, quadros.carregar_analise(), leitura.carregar_leitura(), leitura.carregar_transcricoes(eps)))
