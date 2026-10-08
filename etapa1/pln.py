"""
Etapa 1 - PROCESSAMENTO DE LINGUAGEM NATURAL COM spaCy.

spaCy é uma biblioteca que lê uma frase e devolve cada palavra já anotada:
classe gramatical, forma base, se e entidade nomeada, quem depende de quem.

Vocabulário mínimo (definido aqui porque aparece o tempo todo no código):

- Token: cada pedaço indivisível do texto. "don't" vira dois tokens: "do" e
  "n't". Pontuação também e token.
- Lema (lemma): a forma de dicionário da palavra. "running", "ran" e "runs"
  tem o mesmo lema "run". Contar lema em vez de palavra crua evita o mesmo
  conceito aparecer picado em três lugares do ranking.
- POS (part of speech): classe gramatical. NOUN substantivo, VERB verbo,
  ADJ adjetivo, PROPN nome próprio.
- Stopword: palavra de alta frequência e baixo significado ("the", "is",
  "of"). Filtrar é o que faz o ranking virar tema em vez de gramática.
- NER (named entity recognition): reconhecimento de entidade nomeada. O
  modelo marca trechos como ORG (organizacao), PRODUCT, PERSON, GPE (lugar).
- Noun chunk: frase nominal, o substantivo com seus acompanhantes.
  "the new AI assistant" é um noun chunk.

Pipeline do spaCy usado aqui:
    texto -> tokenizer -> tagger -> attribute_ruler -> lemmatizer -> parser -> ner

Onde aprender: https://spacy.io/usage/spacy-101 (curso gratuito e oficial).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import config

_NLP = None  # o modelo e pesado, entao carregamos uma vez só (cache de módulo)


def carregar_modelo():
    """
    Carrega o modelo de inglês do spaCy.

    O modelo não vem junto com a biblioteca. Instala com:
        python -m spacy download en_core_web_sm
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        import spacy
    except ImportError as erro:
        raise SystemExit("spaCy nao instalado. Rode: pip install spacy") from erro

    try:
        _NLP = spacy.load(config.MODELO_SPACY)
    except OSError as erro:
        raise SystemExit(
            f"Modelo '{config.MODELO_SPACY}' ausente.\n"
            f"Rode: python -m spacy download {config.MODELO_SPACY}"
        ) from erro

    # O spaCy recusa textos gigantes por segurança de memória.
    _NLP.max_length = 2_000_000
    return _NLP


def _token_util(token) -> bool:
    """
    Decide se um token entra na contagem de frequência.

    Regras, em ordem: não pode ser stopword, pontuação, espaço, número ou
    símbolo; precisa ter pelo menos 3 letras; a classe gramatical precisa
    carregar significado; e não pode estar na lista extra de ruído do domínio.
    """
    if token.is_stop or token.is_punct or token.is_space or token.like_num:
        return False
    if token.pos_ not in config.POS_RELEVANTES:
        return False
    lema = token.lemma_.lower().strip()
    if len(lema) < 3 or not lema.isalpha():
        return False
    return lema not in config.STOPWORDS_EXTRA


def analisar(corpus: list[dict[str, Any]], salvar: bool = True) -> dict[str, Any]:
    """
    Roda o pipeline do spaCy em todo o corpus e agrega os resultados.

    `nlp.pipe` processa em lote. E bem mais rápido que chamar `nlp(texto)`
    num laco, porque aproveita vetorizacao interna da biblioteca.
    """
    nlp = carregar_modelo()

    # Truncar protege o tempo de execução: issue de 40 mil caracteres e quase
    # sempre log colado, e as primeiras linhas já trazem o assunto.
    textos = [doc["texto_limpo"][:3000] for doc in corpus]
    print(f"  processando {len(textos)} documentos com spaCy...")

    freq_lemas: Counter[str] = Counter()
    freq_por_pos: dict[str, Counter[str]] = {pos: Counter() for pos in config.POS_RELEVANTES}
    freq_entidades: dict[str, Counter[str]] = {}
    grafias: dict[str, Counter[str]] = {}  # forma minúscula -> grafias vistas
    freq_chunks: Counter[str] = Counter()
    bigramas: Counter[str] = Counter()

    total_tokens = 0
    total_sentencas = 0
    vocabulario: set[str] = set()

    for indice, doc in enumerate(nlp.pipe(textos, batch_size=50), start=1):
        if indice % 100 == 0:
            print(f"    {indice}/{len(textos)}")

        total_tokens += len(doc)
        total_sentencas += sum(1 for _ in doc.sents)

        uteis = [t for t in doc if _token_util(t)]
        lemas = [t.lemma_.lower() for t in uteis]

        freq_lemas.update(lemas)
        vocabulario.update(lemas)

        for token in uteis:
            freq_por_pos[token.pos_][token.lemma_.lower()] += 1

        # Bigrama: par de lemas vizinhos. Revela expressão composta
        # ("code quality", "pull request") que token isolado esconde.
        bigramas.update(f"{a} {b}" for a, b in zip(lemas, lemas[1:]))

        for ent in doc.ents:
            # Dois filtros, e os dois vieram de erro observado na prática:
            # 1. só os tipos de entidade que dizem algo sobre o tema;
            # 2. a entidade precisa conter letra, senao "2.0" e "###" entram.
            if ent.label_ not in config.ENTIDADES_RELEVANTES:
                continue
            texto_ent = " ".join(ent.text.split())
            if not 2 <= len(texto_ent) <= 40:
                continue
            if not any(caractere.isalpha() for caractere in texto_ent):
                continue
            # Agrupamos pela forma minúscula para "Cursor" e "cursor" não
            # virarem duas barras concorrentes no mesmo gráfico. A grafia
            # exibida depois e a mais frequente das duas.
            chave = texto_ent.lower()
            freq_entidades.setdefault(ent.label_, Counter())[chave] += 1
            grafias.setdefault(chave, Counter())[texto_ent] += 1

        for chunk in doc.noun_chunks:
            limpo = " ".join(
                t.lemma_.lower() for t in chunk if _token_util(t)
            ).strip()
            if " " in limpo:  # só frases com duas palavras ou mais
                freq_chunks[limpo] += 1

    # Type-Token Ratio: vocabulário único dividido pelo total de palavras.
    # Perto de 1 significa texto variado, perto de 0 significa repetitivo.
    ttr = len(vocabulario) / max(sum(freq_lemas.values()), 1)

    resultado = {
        "estatisticas": {
            "documentos": len(textos),
            "tokens_totais": total_tokens,
            "sentencas": total_sentencas,
            "vocabulario_unico": len(vocabulario),
            "riqueza_lexical_ttr": round(ttr, 4),
            "media_tokens_por_doc": round(total_tokens / max(len(textos), 1), 1),
        },
        "top_lemas": freq_lemas.most_common(config.TOP_N * 3),
        "top_por_pos": {
            pos: contador.most_common(config.TOP_N)
            for pos, contador in freq_por_pos.items()
        },
        "top_bigramas": bigramas.most_common(config.TOP_N),
        "top_noun_chunks": freq_chunks.most_common(config.TOP_N),
        "top_entidades": {
            rotulo: [
                (grafias[chave].most_common(1)[0][0], qtd)
                for chave, qtd in contador.most_common(config.TOP_N)
            ]
            for rotulo, contador in sorted(
                freq_entidades.items(), key=lambda par: -sum(par[1].values())
            )[:6]
        },
    }

    if salvar:
        with config.ARQ_PLN.open("w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, ensure_ascii=False, indent=2)

    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_PLN.exists():
        return {}
    with config.ARQ_PLN.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _linha(titulo: str) -> None:
    print(f"\n  {titulo}")
    print("  " + "-" * len(titulo))


def imprimir(analise: dict[str, Any]) -> None:
    """Mostra o resultado no terminal, formatado para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção de PLN primeiro.")
        return

    est = analise["estatisticas"]
    _linha("ESTATÍSTICAS DO CORPUS")
    for chave, valor in est.items():
        rotulo = chave.replace("_", " ")
        print(f"    {rotulo:.<30} {valor}")

    _linha(f"TOP {config.TOP_N} LEMAS (palavra na forma de dicionário)")
    for pos, (lema, qtd) in enumerate(analise["top_lemas"][: config.TOP_N], 1):
        print(f"    {pos:>2}. {lema:.<28} {qtd}")

    nomes_pos = {
        "NOUN": "substantivos",
        "VERB": "verbos",
        "ADJ": "adjetivos",
        "PROPN": "nomes próprios",
    }
    for pos, itens in analise["top_por_pos"].items():
        _linha(f"TOP 10 {nomes_pos.get(pos, pos).upper()} ({pos})")
        for lema, qtd in itens[:10]:
            print(f"    {lema:.<30} {qtd}")

    _linha("TOP 10 BIGRAMAS (pares de palavras vizinhas)")
    for termo, qtd in analise["top_bigramas"][:10]:
        print(f"    {termo:.<30} {qtd}")

    _linha("TOP 10 FRASES NOMINAIS (noun chunks)")
    for termo, qtd in analise["top_noun_chunks"][:10]:
        print(f"    {termo:.<30} {qtd}")

    descricao_ent = {
        "ORG": "organizações e empresas",
        "PRODUCT": "produtos",
        "PERSON": "pessoas",
        "GPE": "países e cidades",
        "DATE": "datas",
        "CARDINAL": "números",
        "NORP": "nacionalidades e grupos",
        "WORK_OF_ART": "obras",
        "LANGUAGE": "idiomas",
    }
    for rotulo, itens in analise["top_entidades"].items():
        legenda = descricao_ent.get(rotulo, rotulo)
        _linha(f"ENTIDADES NOMEADAS - {rotulo} ({legenda})")
        for texto, qtd in itens[:8]:
            print(f"    {texto:.<30} {qtd}")
    print()


def demonstrar(frase: str | None = None) -> None:
    """
    Anota uma frase token a token na tela.

    E a forma mais direta de mostrar na apresentação o que o spaCy faz por
    baixo do capô, sem precisar de slide teorico.
    """
    nlp = carregar_modelo()
    frase = frase or (
        "Our team adopted GitHub Copilot last year, but reviewers keep "
        "rejecting the AI generated code because it hides subtle bugs."
    )
    doc = nlp(frase)

    print(f"\nFRASE: {frase}\n")
    print(f"  {'TOKEN':<16}{'LEMA':<16}{'POS':<8}{'DEP':<12}{'STOP':<6}")
    print("  " + "-" * 60)
    for token in doc:
        if token.is_space:
            continue
        print(
            f"  {token.text:<16}{token.lemma_:<16}{token.pos_:<8}"
            f"{token.dep_:<12}{'sim' if token.is_stop else 'nao':<6}"
        )

    print("\n  ENTIDADES NOMEADAS ENCONTRADAS:")
    if doc.ents:
        import spacy

        for ent in doc.ents:
            explicacao = spacy.explain(ent.label_) or ""
            print(f"    {ent.text:<24} {ent.label_:<10} {explicacao}")
    else:
        print("    nenhuma")

    print("\n  FRASES NOMINAIS (noun chunks):")
    for chunk in doc.noun_chunks:
        print(f"    {chunk.text}")

    print("\n  TOKENS QUE SOBREVIVEM AO FILTRO:")
    print("    " + ", ".join(t.lemma_.lower() for t in doc if _token_util(t)))
    print()


if __name__ == "__main__":
    import limpeza

    demonstrar()
    imprimir(analisar(limpeza.carregar_corpus()))
