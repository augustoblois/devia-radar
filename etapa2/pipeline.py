"""
Etapa 2 - O PIPELINE DO spaCy, com dois componentes nossos.

O spaCy processa o texto numa linha de montagem, o pipeline. O tokenizer
pica o texto em tokens e cria o Doc; depois, cada componente recebe o Doc,
acrescenta uma anotação e passa adiante. O modelo en_core_web_md vem com seis:

    tok2vec -> tagger -> parser -> attribute_ruler -> lemmatizer -> ner

- tok2vec: transforma cada token num vetor de números que os componentes
  seguintes usam para decidir (é a "leitura" compartilhada do texto);
- tagger: a classe gramatical fina (NN = substantivo singular, VBZ = verbo
  na 3ª pessoa...);
- parser: a árvore de dependências (quem é sujeito de quem) e, junto, onde
  cada frase termina;
- attribute_ruler: regras que convertem a classe fina na universal (NOUN, VERB);
- lemmatizer: a forma de dicionário ("running" vira "run");
- ner: entidades nomeadas (ORG, PRODUCT, PERSON...), aprendidas de exemplos.

Esta etapa acrescenta dois componentes:

1. entity_ruler (fábrica que vem com o spaCy), ANTES do ner: o dicionário de
   ferramentas da etapa 1 vira regra. Cada variação ("github copilot",
   "copilot") vira uma entidade FERRAMENTA, com o nome canônico guardado no
   id. Como vem antes, o ner respeita o que a regra marcou.
2. vader (componente nosso, feito com @Language.component), no fim: dá a
   cada FRASE a nota do VADER, o medidor de sentimento da etapa 1, e guarda
   num atributo novo, frase._.vader.

Vocabulário:
- Doc, Token, Span: o texto inteiro, uma palavra, um pedaço contínuo (uma
  frase, uma entidade).
- Atributo de extensão (o "._"): campo novo pendurado num Doc, Token ou
  Span. O spaCy reserva o "._" para o campo nosso não colidir com os dele.

Onde aprender: https://spacy.io/usage/processing-pipelines
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

import config

_NLP = None
_VADER = None


# ---------------------------------------------------------------------------
# Corpus da etapa 1
# ---------------------------------------------------------------------------


def carregar_corpus() -> list[dict[str, Any]]:
    """As 979 issues limpas pela etapa 1 (título, texto, ferramentas pela regex, nota do VADER)."""
    if not config.ARQ_ETAPA1_CORPUS.exists():
        raise SystemExit(f"Corpus da etapa 1 não encontrado: {config.ARQ_ETAPA1_CORPUS}")
    with config.ARQ_ETAPA1_CORPUS.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def texto_da_issue(issue: dict[str, Any]) -> str:
    """Título e corpo juntos, como a etapa 1 mediu. O ponto separa o título da primeira frase."""
    return f"{issue.get('titulo', '').strip()}. {issue.get('texto_limpo', '').strip()}"


def resumo(corpus: list[dict[str, Any]]) -> None:
    datas = sorted(issue["criado_em"][:10] for issue in corpus)
    palavras = sorted(len(texto_da_issue(issue).split()) for issue in corpus)
    regex = Counter(nome for issue in corpus for nome in issue.get("ferramentas_ia", []))
    print(f"\n  issues ......................... {len(corpus)}")
    print(f"  repositórios ................... {len({issue['repositorio'] for issue in corpus})}")
    print(f"  período ........................ {datas[0]} a {datas[-1]}")
    print(f"  palavras ....................... {sum(palavras):,}".replace(",", "."))
    print(f"  palavras por issue (mediana) ... {palavras[len(palavras) // 2]}")
    print("\n  Ferramentas citadas, pela regex da etapa 1 (issues):")
    for nome, qtd in regex.most_common(7):
        print(f"    {nome:.<28} {qtd}")


# ---------------------------------------------------------------------------
# Os dois componentes nossos
# ---------------------------------------------------------------------------


def _vader():
    """O VADER da etapa 1, carregado uma vez só."""
    global _VADER
    if _VADER is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
    return _VADER


def padroes_ferramentas() -> list[dict[str, Any]]:
    """
    O dicionário da etapa 1 no formato do entity_ruler.

    Padrão escrito como texto ("github copilot") é comparado pela forma
    minúscula de cada token (phrase_matcher_attr="LOWER"), e "Copilot",
    "copilot" e "COPILOT" casam igual. O Cursor sozinho é um padrão de
    token com TEXT, que compara a grafia exata: só "Cursor" com maiúscula.
    O id guarda o nome canônico, e "gpt-4" e "openai" viram os dois "ChatGPT".
    """
    padroes = [
        {"label": config.ROTULO_FERRAMENTA, "pattern": variacao, "id": nome}
        for nome, variacoes in config.FERRAMENTAS_IA.items()
        for variacao in variacoes
    ]
    padroes.append({"label": config.ROTULO_FERRAMENTA, "pattern": [{"TEXT": config.CURSOR_SOZINHO}], "id": "Cursor"})
    return padroes


def _registrar_componentes() -> None:
    """Registra no spaCy o atributo frase._.vader e o componente que o preenche."""
    from spacy.language import Language
    from spacy.tokens import Span

    if not Span.has_extension("vader"):
        Span.set_extension("vader", default=None)
    if not Span.has_extension("ferramentas"):
        # Atributo calculado na hora (getter): as ferramentas marcadas dentro do Span.
        Span.set_extension(
            "ferramentas",
            getter=lambda span: sorted({e.ent_id_ for e in span.ents if e.label_ == config.ROTULO_FERRAMENTA}),
        )

    if "vader" not in Language.factories:

        @Language.component("vader")
        def componente_vader(doc):
            """Dá a cada frase do Doc o compound do VADER (de -1 a +1)."""
            analisador = _vader()
            for frase in doc.sents:
                frase._.vader = analisador.polarity_scores(frase.text)["compound"]
            return doc


def carregar_modelo():
    """
    O en_core_web_md com os dois componentes nossos.

    O modelo não vem com a biblioteca. Instala com:
        python -m spacy download en_core_web_md
    """
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy

    _registrar_componentes()
    try:
        nlp = spacy.load(config.MODELO_SPACY)
    except OSError as erro:
        raise SystemExit(
            f"Modelo '{config.MODELO_SPACY}' ausente.\nRode: python -m spacy download {config.MODELO_SPACY}"
        ) from erro
    regras = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
    regras.add_patterns(padroes_ferramentas())
    nlp.add_pipe("vader", last=True)
    _NLP = nlp
    return nlp


# ---------------------------------------------------------------------------
# Demonstração: o pipeline por dentro, componente a componente
# ---------------------------------------------------------------------------

FRASE_DEMO = (
    "Claude Code keeps rewriting my tests without asking. "
    "I switched back to GitHub Copilot and the build passes again."
)


def _linha(nome: str, texto: str) -> None:
    print(f"  {nome:.<17} {texto}")


def demonstrar(frase: str | None = None) -> None:
    """
    Roda o pipeline um componente por vez e mostra o que cada um acrescentou.

    `nlp(texto)` faz tudo de uma vez. Aqui o laço faz o mesmo à mão: o
    tokenizer cria o Doc (nlp.make_doc) e cada componente de nlp.pipeline é
    chamado em ordem sobre ele.
    """
    import spacy

    nlp = carregar_modelo()
    frase = frase or FRASE_DEMO
    print(f"\n  FRASE: {frase}")
    print(f"\n  PIPELINE: tokenizer -> {' -> '.join(nlp.pipe_names)}\n")

    doc = nlp.make_doc(frase)
    _linha("tokenizer", f"{len(doc)} tokens: {' | '.join(t.text for t in doc[:12])}{' ...' if len(doc) > 12 else ''}")
    for nome, componente in nlp.pipeline:
        doc = componente(doc)
        if nome == "tok2vec":
            _linha(nome, f"cada token virou um vetor de {doc.tensor.shape[1]} números (doc.tensor: {doc.tensor.shape[0]} x {doc.tensor.shape[1]})")
        elif nome == "tagger":
            _linha(nome, " ".join(f"{t.text}/{t.tag_}" for t in doc[:8]) + " ...")
        elif nome == "parser":
            frases = list(doc.sents)
            raiz = frases[0].root
            sujeito = next((t.text for t in raiz.children if t.dep_ == "nsubj"), "-")
            _linha(nome, f"{len(frases)} frases; na 1ª, o verbo principal é '{raiz.text}' e o sujeito, '{sujeito}'")
        elif nome == "attribute_ruler":
            _linha(nome, " ".join(f"{t.text}/{t.pos_}" for t in doc[:8]) + " ...")
        elif nome == "lemmatizer":
            trocas = [f"{t.text}->{t.lemma_}" for t in doc if t.lemma_.lower() != t.text.lower() and t.is_alpha]
            _linha(nome, ", ".join(trocas[:6]))
        elif nome in ("entity_ruler", "ner"):
            ents = [f"'{e.text}' {e.label_}" for e in doc.ents]
            _linha(nome, "; ".join(ents) or "nenhuma entidade")
        elif nome == "vader":
            _linha(nome, "   ".join(f"frase {i}: {f._.vader:+.3f}" for i, f in enumerate(doc.sents, start=1)))

    print(f"\n  {'TOKEN':<12}{'LEMA':<12}{'POS':<7}{'TAG':<6}{'DEP':<10}{'ENTIDADE'}")
    print("  " + "-" * 62)
    for token in doc:
        entidade = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else ""
        print(f"  {token.text:<12}{token.lemma_:<12}{token.pos_:<7}{token.tag_:<6}{token.dep_:<10}{entidade}")

    print("\n  O MESMO TEXTO SEM O NOSSO entity_ruler (só o ner que vem no modelo):")
    with nlp.select_pipes(disable=["entity_ruler"]):
        sem_regra = nlp(frase)
    for ent in sem_regra.ents:
        print(f"    '{ent.text}' -> {ent.label_} ({spacy.explain(ent.label_)})")
    if not sem_regra.ents:
        print("    nenhuma entidade")
    print("\n  COM O entity_ruler:")
    for ent in doc.ents:
        nome = f", nome canônico '{ent.ent_id_}'" if ent.label_ == config.ROTULO_FERRAMENTA else ""
        print(f"    '{ent.text}' -> {ent.label_}{nome}")
    print()


# ---------------------------------------------------------------------------
# O corpus inteiro pelo pipeline
# ---------------------------------------------------------------------------


def _lema_util(token) -> str | None:
    """
    O lema do token, se ele entra na contagem de vocabulário; senão None.

    Mesmas regras da etapa 1: fora stopword, pontuação, número e símbolo;
    só classes que carregam significado; 3 letras ou mais.
    """
    if token.is_stop or token.is_punct or token.is_space or token.like_num:
        return None
    if token.pos_ not in config.POS_RELEVANTES:
        return None
    lema = token.lemma_.lower().strip()
    lema = config.CORRECAO_LEMA.get(lema, lema)
    if len(lema) < 3 or not lema.isalpha() or lema in config.STOPWORDS_EXTRA:
        return None
    return lema


def _regex_etapa1() -> dict[str, re.Pattern[str]]:
    """A regex de ferramentas da etapa 1, variações da mais longa para a mais curta."""
    return {
        nome: re.compile(rf"\b(?:{'|'.join(re.escape(v) for v in sorted(variacoes, key=len, reverse=True))})\b", re.IGNORECASE)
        for nome, variacoes in config.FERRAMENTAS_IA.items()
    }


def _rotulo_sem_regra(nlp, frases: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """
    Como o ner que vem no modelo rotularia cada ferramenta, sem a nossa regra.

    Roda o pipeline de novo, sem o entity_ruler, só nas frases que citam
    ferramenta, e olha o rótulo que o ner deu ao mesmo pedaço de texto.
    """
    rotulos: dict[str, Counter[str]] = {}
    with nlp.select_pipes(disable=["entity_ruler", "vader"]):
        docs = nlp.pipe((f["texto"] for f in frases), batch_size=64)
        for frase, doc in zip(frases, docs):
            for inicio, fim, nome in frase["spans"]:
                rotulo = next((e.label_ for e in doc.ents if e.start_char < fim and e.end_char > inicio), "nenhum")
                rotulos.setdefault(nome, Counter())[rotulo] += 1
    return rotulos


def processar(corpus: list[dict[str, Any]], forcar: bool = False) -> dict[str, Any]:
    """
    Passa as 979 issues pelo pipeline e guarda cada frase com as anotações.

    O parser corta as frases. A regex das etapas de fala só corta na
    pontuação, e a limpeza da etapa 1 apagou as quebras de linha: um item de
    lista emendava no outro (ver o diário de bordo).

    Leva uns 3 minutos na CPU. Depois disso, a opção lê o cache em
    dados/brutos/frases.json.
    """
    if not forcar and config.ARQ_FRASES.exists() and config.ARQ_PIPELINE.exists():
        dados = carregar_frases()
        if dados.get("issues") == len(corpus):
            print(f"  frases lidas do cache ({config.ARQ_FRASES.name})")
            return carregar_analise()

    nlp = carregar_modelo()
    textos = [texto_da_issue(issue) for issue in corpus]
    frases: list[dict[str, Any]] = []
    lemas_por_issue: list[dict[str, int]] = []
    rotulos_ner: Counter[str] = Counter()
    issues_por_ferramenta: Counter[str] = Counter()
    tokens = 0
    relogio = time.time()

    print(f"  processando {len(textos)} issues com o pipeline ({', '.join(nlp.pipe_names)})...")
    for indice, doc in enumerate(nlp.pipe(textos, batch_size=20)):
        if (indice + 1) % 100 == 0:
            print(f"    {indice + 1}/{len(textos)}  ({time.time() - relogio:.0f} s)")
        tokens += len(doc)
        lemas_por_issue.append(dict(Counter(lema for t in doc if (lema := _lema_util(t)))))
        rotulos_ner.update(e.label_ for e in doc.ents if e.label_ != config.ROTULO_FERRAMENTA)
        issues_por_ferramenta.update({e.ent_id_ for e in doc.ents if e.label_ == config.ROTULO_FERRAMENTA})
        for sent in doc.sents:
            texto = sent.text.strip()
            if not texto:
                continue
            ferramentas = [e for e in sent.ents if e.label_ == config.ROTULO_FERRAMENTA]
            frases.append({
                "issue": indice,
                "texto": texto,
                "palavras": len(texto.split()),
                "ferramentas": sent._.ferramentas,
                "vader": round(sent._.vader, 4),
                # Posição de cada ferramenta dentro da frase, para o teste sem a regra.
                "spans": [(e.start_char - sent.start_char, e.end_char - sent.start_char, e.ent_id_) for e in ferramentas],
            })
    print(f"  {len(textos)} issues em {time.time() - relogio:.0f} s")

    com_ferramenta = [f for f in frases if f["spans"]]
    print(f"  rodando o ner sem o entity_ruler em {len(com_ferramenta)} frases que citam ferramenta...")
    sem_regra = _rotulo_sem_regra(nlp, com_ferramenta)

    with config.ARQ_FRASES.open("w", encoding="utf-8") as arquivo:
        json.dump({"issues": len(corpus), "frases": frases, "lemas_por_issue": lemas_por_issue}, arquivo, ensure_ascii=False)

    avaliadas = [f for f in frases if f["palavras"] >= config.MIN_PALAVRAS_FRASE]
    # A regex da etapa 1 rodada no MESMO texto limpo. O campo ferramentas_ia
    # do corpus não serve para comparar: a etapa 1 extraiu as ferramentas do
    # texto bruto, antes de apagar código e link (ver o diário de bordo).
    regex = Counter(nome for texto in textos for nome, padrao in _regex_etapa1().items() if padrao.search(texto))
    analise = {
        "estatisticas": {
            "issues": len(corpus),
            "tokens": tokens,
            "frases": len(frases),
            "frases_avaliadas": len(avaliadas),
            "frases_com_ferramenta": len(com_ferramenta),
            "palavras_por_frase_mediana": sorted(f["palavras"] for f in frases)[len(frases) // 2],
            "frases_acima_300_palavras": sum(f["palavras"] > 300 for f in frases),
        },
        "pipeline": ["tokenizer", *nlp.pipe_names],
        "ferramentas": {
            "frases": dict(Counter(n for f in frases for n in f["ferramentas"]).most_common()),
            "issues_entity_ruler": dict(issues_por_ferramenta.most_common()),
            "issues_regex": dict(regex.most_common()),
            "rotulo_sem_regra": {n: dict(c.most_common()) for n, c in sem_regra.items()},
        },
        "entidades_ner": dict(rotulos_ner.most_common(10)),
    }
    with config.ARQ_PIPELINE.open("w", encoding="utf-8") as arquivo:
        json.dump(analise, arquivo, ensure_ascii=False, indent=1)
    print(f"  salvo em {config.ARQ_FRASES.name} (cache) e {config.ARQ_PIPELINE.name}")
    return analise


def carregar_frases() -> dict[str, Any]:
    """As frases anotadas (cache). Vazio se a opção 3 ainda não rodou."""
    if not config.ARQ_FRASES.exists():
        return {}
    with config.ARQ_FRASES.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_PIPELINE.exists():
        return {}
    with config.ARQ_PIPELINE.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


ROTULOS = {
    "issues": "issues",
    "tokens": "tokens",
    "frases": "frases (cortadas pelo parser)",
    "frases_avaliadas": f"frases com {config.MIN_PALAVRAS_FRASE} palavras ou mais",
    "frases_com_ferramenta": "frases que citam ferramenta",
    "palavras_por_frase_mediana": "palavras por frase (mediana)",
    "frases_acima_300_palavras": "frases com mais de 300 palavras",
}


def imprimir(analise: dict[str, Any]) -> None:
    """Mostra o resultado do pipeline no terminal, formatado para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção 3 primeiro.")
        return
    print("\n  O CORPUS PELO PIPELINE")
    print("  " + "-" * 60)
    for chave, valor in analise["estatisticas"].items():
        print(f"    {ROTULOS.get(chave, chave):.<44} {valor:,}".replace(",", "."))

    ferr = analise["ferramentas"]
    print("\n  FERRAMENTAS: regex da etapa 1 x entity_ruler, no mesmo texto limpo")
    print("  " + "-" * 60)
    print(f"    {'ferramenta':<18}{'issues (regex)':>15}{'issues (regra)':>16}{'frases':>8}")
    for nome, qtd in ferr["issues_entity_ruler"].items():
        print(f"    {nome:<18}{ferr['issues_regex'].get(nome, 0):>15}{qtd:>16}{ferr['frases'].get(nome, 0):>8}")
    print("    (o Cursor sobe porque a regra conta \"Cursor\" com maiúscula; ver o diário)")

    print("\n  SEM A NOSSA REGRA, O ner DO MODELO CHAMARIA CADA FERRAMENTA DE:")
    print("  " + "-" * 60)
    for nome, rotulos in sorted(ferr["rotulo_sem_regra"].items(), key=lambda par: -sum(par[1].values()))[:6]:
        total = sum(rotulos.values())
        partes = ", ".join(f"{r} {q / total:.0%}" for r, q in list(rotulos.items())[:4])
        print(f"    {nome:<18}{partes}")
    print()


if __name__ == "__main__":
    demonstrar()
    imprimir(processar(carregar_corpus()))
