"""
Etapa 2 - EMBEDDINGS: o significado como lista de números.

Embedding é um vetor (uma lista de números) que representa o significado de
uma palavra ou de uma frase. Coisas de significado parecido ganham vetores
parecidos, e "parecido" vira conta: a similaridade do cosseno, que mede o
ângulo entre dois vetores (1 = mesma direção; perto de 0 = sem relação).

Dois tipos, e a diferença entre eles é o assunto da opção 4:

1. ESTÁTICO (vetores de palavra do spaCy, en_core_web_md): um vetor fixo de
   300 números por palavra, aprendido antes de as ferramentas de IA existirem,
   a partir de quais palavras aparecem perto de quais em texto da web. O
   "cursor" tem um vetor só, seja o do mouse, seja o editor. O vetor de uma
   frase é a média dos vetores das palavras dela.
2. CONTEXTUAL (sentence-transformers, modelo MiniLM): um transformer lê a
   frase inteira e devolve 384 números para ela. A mesma palavra muda de
   vetor conforme as vizinhas.

Com os vetores de frase, duas aplicações:

- BUSCA SEMÂNTICA (opção 6): acha frases pelo significado, sem exigir as
  mesmas palavras. A busca da etapa 1 era por regex, palavra exata.
- TEMAS (opção 5): o KMeans junta as issues de vetores parecidos em grupos,
  e o t-SNE desenha os 384 números num plano para o gráfico.

Vocabulário:
- KMeans: escolhe k centros e põe cada issue no grupo do centro mais
  próximo; move os centros para o meio dos grupos e repete até parar.
- Silhueta: nota de -1 a +1 para um agrupamento. Alta quando cada issue está
  bem mais perto do próprio grupo do que do grupo vizinho.
- t-SNE: técnica que desenha pontos de muitas dimensões num plano tentando
  manter perto quem estava perto. Distância entre grupos distantes no
  desenho não quer dizer nada; só a vizinhança.
- c-TF-IDF: nota de cada palavra num grupo. Alta para a palavra frequente
  NAQUELE grupo e rara nos outros. As palavras de nota maior dão nome ao tema.

Onde aprender: https://sbert.net e https://spacy.io/usage/linguistic-features#vectors-similarity
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from typing import Any

import numpy as np

import config
import pipeline

_VOCABULARIO = None  # spaCy só com o vocabulário e os vetores: carrega em 1 s
_MODELO = None       # o MiniLM, carregado uma vez só

# Cinco frases para comparar os dois tipos de vetor: duas paráfrases (1 e 2;
# 3 e 4), uma armadilha (2 e 3 dividem a palavra "cursor", com sentidos
# diferentes) e uma frase sem relação com as outras (5).
FRASES_DEMO = [
    "Copilot generated the entire file for me.",
    "Cursor wrote the whole module for me.",
    "Move the cursor to the end of the line.",
    "Put the mouse pointer after the last character.",
    "The AI keeps inventing functions that do not exist.",
]
# Os pares comparados na tela (índices em FRASES_DEMO) e o que liga cada par.
PARES_DEMO = [
    (0, 1, "mesmo sentido (a IA escreveu o código)"),
    (2, 3, "mesmo sentido (o ponteiro do mouse)"),
    (1, 2, "mesma palavra (cursor), outro sentido"),
    (0, 4, "sem relação"),
]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def _vocabulario():
    """O en_core_web_md sem nenhum componente: só o tokenizer e a tabela de vetores."""
    global _VOCABULARIO
    if _VOCABULARIO is None:
        import spacy

        _VOCABULARIO = spacy.load(config.MODELO_SPACY, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    return _VOCABULARIO


def modelo_frases():
    """
    O MiniLM do sentence-transformers.

    Tenta primeiro o modelo já guardado no computador, sem internet, para a
    demo não depender da rede da sala. Na primeira vez, baixa (~90 MB).
    """
    global _MODELO
    if _MODELO is None:
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging

        logging.disable_progress_bar()
        logging.set_verbosity_error()
        try:
            _MODELO = SentenceTransformer(config.MODELO_FRASES, local_files_only=True)
        except (OSError, ValueError):
            print("  baixando o modelo de frases (~90 MB, só na primeira vez)...")
            _MODELO = SentenceTransformer(config.MODELO_FRASES)
    return _MODELO


def codificar(textos: list[str], progresso: bool = False) -> np.ndarray:
    """
    Vetores de frase normalizados (comprimento 1). Com vetor de comprimento 1,
    o cosseno entre dois vetores é só o produto escalar: a @ b.
    """
    return modelo_frases().encode(
        textos, batch_size=config.LOTE_FRASES, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=progresso,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Opção 4: estático x contextual
# ---------------------------------------------------------------------------


def mesmo_vetor(palavra: str) -> list[str] | None:
    """
    As palavras que usam EXATAMENTE o mesmo vetor que `palavra`.

    O en_core_web_md guarda 20 mil vetores para quase 700 mil palavras. Para
    caber, cada palavra rara foi apontada para o vetor da palavra frequente
    mais parecida, e passou a dividir esse vetor com outras. None quando a
    palavra nem vetor tem.
    """
    vocab = _vocabulario().vocab
    linha = vocab.vectors.key2row.get(vocab.strings[palavra])
    if linha is None:
        return None
    outras = {
        vocab.strings[chave].lower()
        for chave, r in vocab.vectors.key2row.items()
        if r == linha and vocab.strings[chave].isalpha()
    }
    outras.discard(palavra)
    return sorted(outras)


def similaridades(frases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de similaridade das frases pelos dois tipos de vetor."""
    docs = [_vocabulario()(f) for f in frases]
    estatico = np.array([[a.similarity(b) for b in docs] for a in docs])
    vetores = codificar(frases)
    return estatico, vetores @ vetores.T


def pares_sorteados(frases: list[str], n: int = 200) -> dict[str, Any]:
    """
    Similaridade de pares de frases sorteadas do corpus, sem relação entre si.

    Mede o quanto cada tipo de vetor DIFERENCIA as frases: se dois textos
    quaisquer já saem parecidos, a busca por significado não funciona.
    """
    random.seed(config.SEMENTE)
    curtas = [f for f in frases if f.count(" ") < 40]
    amostra = random.sample(curtas, 2 * n)
    docs = [_vocabulario()(f) for f in amostra]
    vetores = codificar(amostra)
    estatico = [docs[i].similarity(docs[i + 1]) for i in range(0, 2 * n, 2) if docs[i].vector_norm and docs[i + 1].vector_norm]
    contextual = [float(vetores[i] @ vetores[i + 1]) for i in range(0, 2 * n, 2)]
    def resumo(valores: list[float]) -> dict[str, Any]:
        return {"media": round(float(np.mean(valores)), 3), "p10": round(float(np.percentile(valores, 10)), 3),
                "p90": round(float(np.percentile(valores, 90)), 3), "valores": [round(float(v), 3) for v in valores]}

    return {"n": n, "estatico": resumo(estatico), "contextual": resumo(contextual)}


def demonstrar_palavras(frases_corpus: list[str] | None = None) -> dict[str, Any]:
    """Os vetores estáticos não conhecem as ferramentas; os contextuais leem a frase."""
    vetores = _vocabulario().vocab.vectors
    print(f"\n  1. VETORES ESTÁTICOS DO spaCy ({config.MODELO_SPACY})")
    print(f"     {vetores.shape[0]:,} vetores de {vetores.shape[1]} números para {vetores.n_keys:,} palavras".replace(",", "."))
    print("     Palavra rara não tem vetor próprio: divide o de outras.\n")
    compartilhados = {}
    for palavra in config.PALAVRAS_DEMO + ["chatgpt", "ollama"]:
        outras = mesmo_vetor(palavra)
        compartilhados[palavra] = outras
        if outras is None:
            print(f"     {palavra:<9} sem vetor: a palavra é mais nova que o modelo")
        else:
            print(f"     {palavra:<9} mesmo vetor que {len(outras)} palavras, como: {', '.join(outras[:9])}")

    print("\n  2. SIMILARIDADE ENTRE FRASES (cosseno): estático x contextual (MiniLM)")
    estatico, contextual = similaridades(FRASES_DEMO)
    for indice, frase in enumerate(FRASES_DEMO, start=1):
        print(f"     [{indice}] {frase}")
    print(f"\n     {'par':<7}{'o que liga as duas frases':<39}{'estático':>9}{'contextual':>11}")
    for a, b, relacao in PARES_DEMO:
        print(f"     {a + 1} e {b + 1}  {relacao:<39}{estatico[a, b]:>9.2f}{contextual[a, b]:>11.2f}")
    print("\n     O estático acha 2 e 3 (mesma palavra) mais parecidas que 3 e 4")
    print("     (mesmo sentido). O contextual faz o contrário.")

    sorteio = None
    if frases_corpus:
        sorteio = pares_sorteados(frases_corpus)
        e, c = sorteio["estatico"], sorteio["contextual"]
        print(f"\n  3. {sorteio['n']} PARES DE FRASES SORTEADAS DO CORPUS (sem relação entre si)")
        print(f"     estático:   média {e['media']:.2f}  (de {e['p10']:.2f} a {e['p90']:.2f} em 80% dos pares)")
        print(f"     contextual: média {c['media']:.2f}  (de {c['p10']:.2f} a {c['p90']:.2f} em 80% dos pares)")
        print("     Na média de vetores de palavra, duas frases quaisquer já saem parecidas.")
    print()
    return {"compartilhados": {p: (o[:12] if o else None) for p, o in compartilhados.items()},
            "quantos": {p: (len(o) if o is not None else None) for p, o in compartilhados.items()},
            "frases_demo": FRASES_DEMO, "estatico": estatico.round(3).tolist(), "contextual": contextual.round(3).tolist(),
            "sorteio": sorteio}


# ---------------------------------------------------------------------------
# Opção 5: vetores do corpus e temas
# ---------------------------------------------------------------------------


def frases_avaliadas(frases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """As frases que entram nas análises: 6 palavras ou mais. A ordem é a do cache."""
    return [f for f in frases if f["palavras"] >= config.MIN_PALAVRAS_FRASE]


def vetorizar(corpus: list[dict[str, Any]], frases: list[dict[str, Any]], forcar: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Vetor de cada frase avaliada e de cada issue (título e texto).

    A issue é longa, e o MiniLM só lê os primeiros 256 pedaços de palavra
    (cerca de 200 palavras): o título e o começo, onde o assunto aparece.
    """
    avaliadas = frases_avaliadas(frases)
    if not forcar and config.ARQ_VETORES_FRASES.exists() and config.ARQ_VETORES_ISSUES.exists():
        vf, vi = np.load(config.ARQ_VETORES_FRASES), np.load(config.ARQ_VETORES_ISSUES)
        if len(vf) == len(avaliadas) and len(vi) == len(corpus):
            print(f"  vetores lidos do cache ({len(vf):,} frases e {len(vi)} issues)".replace(",", "."))
            return vf, vi
    relogio = time.time()
    print(f"  calculando o vetor de {len(corpus)} issues...")
    vi = codificar([pipeline.texto_da_issue(issue) for issue in corpus], progresso=True)
    print(f"  calculando o vetor de {len(avaliadas):,} frases...".replace(",", "."))
    vf = codificar([f["texto"] for f in avaliadas], progresso=True)
    np.save(config.ARQ_VETORES_FRASES, vf)
    np.save(config.ARQ_VETORES_ISSUES, vi)
    print(f"  vetores calculados em {time.time() - relogio:.0f} s e guardados em dados/brutos/")
    return vf, vi


def _nomes_dos_temas(rotulos: np.ndarray, lemas_por_issue: list[dict[str, int]], k: int) -> list[list[tuple[str, float]]]:
    """
    c-TF-IDF: cada tema vira um documento só, com todos os lemas das suas issues.

    tf = frequência do lema no tema / total de lemas do tema
    idf = log(1 + média de lemas por tema / frequência do lema em todos os temas)
    """
    por_tema = [Counter() for _ in range(k)]
    for rotulo, lemas in zip(rotulos, lemas_por_issue):
        por_tema[int(rotulo)].update(lemas)
    total = Counter()
    for contagem in por_tema:
        total.update(contagem)
    media = sum(total.values()) / k
    nomes = []
    for contagem in por_tema:
        soma = sum(contagem.values()) or 1
        notas = {lema: (qtd / soma) * math.log(1 + media / total[lema]) for lema, qtd in contagem.items() if qtd >= 5}
        nomes.append([(lema, round(nota, 4)) for lema, nota in sorted(notas.items(), key=lambda par: -par[1])[: config.TERMOS_TEMA]])
    return nomes


def agrupar(corpus: list[dict[str, Any]], vetores_issues: np.ndarray, lemas_por_issue: list[dict[str, int]]) -> dict[str, Any]:
    """Testa de MIN_TEMAS a MAX_TEMAS grupos, fica com o de maior silhueta e desenha o mapa."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    print(f"  KMeans de {config.MIN_TEMAS} a {config.MAX_TEMAS} grupos (silhueta pelo cosseno):")
    silhuetas = {}
    modelos = {}
    for k in range(config.MIN_TEMAS, config.MAX_TEMAS + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=config.SEMENTE).fit(vetores_issues)
        silhuetas[k] = round(float(silhouette_score(vetores_issues, modelo.labels_, metric="cosine")), 4)
        modelos[k] = modelo
        print(f"    k={k:>2}  silhueta {silhuetas[k]:.4f}")
    k = max(silhuetas, key=silhuetas.get)
    modelo = modelos[k]
    rotulos = modelo.labels_
    print(f"  escolhido: k={k}")

    print("  desenhando o mapa com o t-SNE...")
    plano = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=config.SEMENTE).fit_transform(vetores_issues)

    nomes = _nomes_dos_temas(rotulos, lemas_por_issue, k)
    centros = modelo.cluster_centers_ / np.linalg.norm(modelo.cluster_centers_, axis=1, keepdims=True)
    temas = []
    for t in range(k):
        membros = np.where(rotulos == t)[0]
        perto = membros[np.argsort(-(vetores_issues[membros] @ centros[t]))[:4]]
        temas.append({
            "tema": t,
            "issues": len(membros),
            "termos": nomes[t],
            "exemplos": [corpus[i]["titulo"][:90] for i in perto],
        })
    # Do maior para o menor: o número do tema passa a ser a posição no ranking.
    ordem = sorted(range(k), key=lambda t: -temas[t]["issues"])
    novo = {antigo: novo for novo, antigo in enumerate(ordem)}
    temas = [{**temas[antigo], "tema": novo[antigo]} for antigo in ordem]

    resultado = {
        "k": k,
        "silhuetas": silhuetas,
        "temas": temas,
        "tema_da_issue": [novo[int(r)] for r in rotulos],
        "plano": np.round(plano, 2).tolist(),
    }
    with config.ARQ_TEMAS.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False)
    print(f"  salvo em {config.ARQ_TEMAS.name}")
    return resultado


def carregar_temas() -> dict[str, Any]:
    if not config.ARQ_TEMAS.exists():
        return {}
    with config.ARQ_TEMAS.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def nome_curto(tema: dict[str, Any], termos: int = 3) -> str:
    return " · ".join(lema for lema, _ in tema["termos"][:termos])


def imprimir_temas(temas: dict[str, Any]) -> None:
    if not temas:
        print("  Nenhum tema salvo. Rode a opção 5 primeiro.")
        return
    print(f"\n  OS {temas['k']} TEMAS DAS ISSUES (KMeans nos vetores do MiniLM)")
    print("  " + "-" * 64)
    for tema in temas["temas"]:
        print(f"\n  Tema {tema['tema'] + 1}: {tema['issues']} issues  |  {', '.join(l for l, _ in tema['termos'])}")
        for titulo in tema["exemplos"][:3]:
            print(f"      - {titulo}")
    print()


# ---------------------------------------------------------------------------
# Opção 6: busca semântica
# ---------------------------------------------------------------------------

RE_PALAVRA = re.compile(r"[a-z]{2,}")
_PORTER = None  # o stemmer de Porter, criado na primeira busca


def palavras_de(texto: str) -> dict[str, str]:
    """
    As palavras que uma busca literal procuraria (fora as stopwords do spaCy),
    cada uma com o seu radical. O radical (stemmer de Porter, do NLTK) junta
    as variações da palavra: "copyrighted" e "copyright" dão "copyright".
    """
    global _PORTER
    from spacy.lang.en.stop_words import STOP_WORDS

    if _PORTER is None:
        from nltk.stem import PorterStemmer

        _PORTER = PorterStemmer()
    return {p: _PORTER.stem(p) for p in RE_PALAVRA.findall(texto.lower()) if p not in STOP_WORDS}


def buscar(consulta: str, avaliadas: list[dict[str, Any]], vetores_frases: np.ndarray, n: int = 8) -> list[tuple[float, dict[str, Any]]]:
    """
    As n frases de vetor mais parecido com o da consulta (cosseno).

    Frase repetida entra uma vez só: alguns projetos abrem várias issues com o
    mesmo modelo de texto.
    """
    notas = vetores_frases @ codificar([consulta])[0]
    achados: list[tuple[float, dict[str, Any]]] = []
    vistas: set[str] = set()
    for i in np.argsort(-notas):
        if avaliadas[i]["texto"] in vistas:
            continue
        vistas.add(avaliadas[i]["texto"])
        achados.append((float(notas[i]), avaliadas[i]))
        if len(achados) == n:
            break
    return achados


def demonstrar_busca(consulta: str, avaliadas: list[dict[str, Any]], vetores_frases: np.ndarray, corpus: list[dict[str, Any]]) -> None:
    """Busca por significado, ao lado do que a busca literal (regex) acharia."""
    relogio = time.time()
    achados = buscar(consulta, avaliadas, vetores_frases)
    gasto = time.time() - relogio
    procuradas = palavras_de(consulta)  # palavra da consulta -> radical
    # A busca literal procura o radical no começo da palavra: copyright acha copyrighted.
    literal = re.compile(r"\b(?:" + "|".join(sorted(set(procuradas.values()))) + r")\w*", re.IGNORECASE) if procuradas else None
    com_alguma = sum(bool(literal and literal.search(f["texto"])) for f in avaliadas)
    print(f"\n  CONSULTA: {consulta}")
    print(f"  {len(avaliadas):,} frases comparadas em {gasto:.2f} s\n".replace(",", "."))
    sem_comum = 0
    for posicao, (nota, frase) in enumerate(achados, start=1):
        radicais = set(palavras_de(frase["texto"]).values())
        comuns = sorted(p for p, radical in procuradas.items() if radical in radicais)
        sem_comum += not comuns
        marca = f"palavras da consulta: {', '.join(comuns)}" if comuns else "nenhuma palavra da consulta"
        print(f"  ({posicao}) {nota:.2f}  {frase['texto'][:150]}")
        print(f"            issue: {corpus[frase['issue']]['titulo'][:70]}  |  {marca}")
    print(f"\n  {sem_comum} dos {len(achados)} achados não têm nenhuma palavra da consulta (nem variação dela).")
    if literal:
        quantas = f"{com_alguma:,}".replace(",", ".")
        print(f"  A busca literal por {', '.join(sorted(procuradas))} (e variações) casaria com {quantas} frases,")
        print("  sem ordem de relevância.")
    print()


if __name__ == "__main__":
    dados = pipeline.carregar_frases()
    textos = [f["texto"] for f in dados.get("frases", [])]
    demonstrar_palavras(textos or None)
