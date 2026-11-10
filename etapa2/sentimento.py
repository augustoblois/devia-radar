"""
Etapa 2 - SENTIMENTO COM TRANSFORMER, lado a lado com o VADER da etapa 1.

O VADER é um LÉXICO: uma tabela de 7.500 palavras com nota de -4 a +4,
somadas com algumas regras (negação, ênfase, "but"). Ele não lê a frase:
soma as palavras que conhece. O transformer lê a frase inteira. Os dois
medem as mesmas 34 mil frases das issues, e a comparação mostra onde o
léxico erra.

Vocabulário:
- Transformer: rede neural que lê a frase toda de uma vez. Em cada camada,
  cada pedaço de palavra "olha" para todos os outros (o mecanismo de
  atenção) e ajusta o próprio vetor conforme o contexto. Por isso "kill" em
  "kill the process" e em "this bug is killing me" sai com sentido diferente.
- Token (do transformer): pedaço de palavra. O RoBERTa tem um vocabulário
  de 50 mil pedaços e quebra a palavra rara em pedaços conhecidos.
- Pré-treino e ajuste fino: o RoBERTa aprendeu inglês adivinhando palavras
  escondidas em 160 GB de texto. Esta versão continuou o pré-treino em 124
  milhões de tweets e, no fim, foi treinada com tweets rotulados à mão como
  negativo, neutro ou positivo (o ajuste fino).
- Logit e softmax: o modelo devolve uma nota crua (logit) para cada classe.
  A softmax transforma as três notas em probabilidades que somam 1, e a
  classe é a de maior probabilidade.
- Matriz de confusão: a tabela que cruza as duas classificações. Na
  diagonal, as frases em que os dois concordam.
- Kappa de Cohen: concordância descontando a que o acaso daria. 0 é o
  acaso; 1 é concordar sempre.

Onde aprender: https://huggingface.co/docs/transformers/quicktour
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

import numpy as np

import config
import pipeline

CLASSES = ["negativo", "neutro", "positivo"]  # a ordem das colunas das probabilidades
RE_PALAVRA = re.compile(r"[a-z][a-z']*")

_TOKENIZADOR = None
_MODELO = None
_DISPOSITIVO = "cpu"

# Frases da demonstração (opção 7). As duas primeiras são a frase da opção 2;
# a terceira é uma instrução técnica, e a quarta, um pedido de recurso.
FRASES_DEMO = [
    "Claude Code keeps rewriting my tests without asking.",
    "I switched back to GitHub Copilot and the build passes again.",
    "Kill the process and restart the server.",
    "We would like Copilot to support custom prompts.",
]


# ---------------------------------------------------------------------------
# VADER e transformer
# ---------------------------------------------------------------------------


def classe_vader(compound: float) -> str:
    if compound >= config.LIMITE_POSITIVO:
        return "positivo"
    if compound <= config.LIMITE_NEGATIVO:
        return "negativo"
    return "neutro"


def carregar_transformer():
    """
    O tokenizador e o modelo do Hugging Face, carregados uma vez só.

    Tenta primeiro o modelo guardado no computador, sem internet, para a demo
    não depender da rede da sala. Na primeira vez, baixa (~500 MB).
    """
    global _TOKENIZADOR, _MODELO, _DISPOSITIVO
    if _MODELO is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging

        logging.disable_progress_bar()
        # O transformers 5 imprime um "relatório de carga" a cada modelo: aqui
        # ele avisa que os pesos do pooler sobram, o que é normal para este modelo.
        logging.set_verbosity_error()
        try:
            _TOKENIZADOR = AutoTokenizer.from_pretrained(config.MODELO_SENTIMENTO, local_files_only=True)
            _MODELO = AutoModelForSequenceClassification.from_pretrained(config.MODELO_SENTIMENTO, local_files_only=True)
        except (OSError, ValueError):
            print("  baixando o transformer de sentimento (~500 MB, só na primeira vez)...")
            _TOKENIZADOR = AutoTokenizer.from_pretrained(config.MODELO_SENTIMENTO)
            _MODELO = AutoModelForSequenceClassification.from_pretrained(config.MODELO_SENTIMENTO)
        _DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
        _MODELO.to(_DISPOSITIVO).eval()  # eval: modo de uso, sem o sorteio que só serve no treino
    return _TOKENIZADOR, _MODELO


def _ordem_das_colunas() -> list[int]:
    """Posição de negativo, neutro e positivo na saída do modelo (vem do id2label dele)."""
    _, modelo = carregar_transformer()
    rotulos = {config.ROTULOS_TRANSFORMER[r.lower()]: int(i) for i, r in modelo.config.id2label.items()}
    return [rotulos[c] for c in CLASSES]


def classificar(textos: list[str], progresso: bool = False) -> np.ndarray:
    """
    Probabilidade de negativo, neutro e positivo para cada texto (n x 3).

    As frases vão em lotes. Num lote, a frase curta é completada com um token
    vazio até o tamanho da maior (padding), então o programa ordena por
    tamanho antes: frases parecidas no mesmo lote, quase nada de enchimento.
    """
    import torch

    tokenizador, modelo = carregar_transformer()
    colunas = _ordem_das_colunas()
    ordem = np.argsort([len(t) for t in textos])
    saida = np.zeros((len(textos), 3), dtype=np.float32)
    relogio = time.time()
    with torch.no_grad():  # só prever: sem guardar o que o treino precisaria
        for inicio in range(0, len(textos), config.LOTE_SENTIMENTO):
            indices = ordem[inicio:inicio + config.LOTE_SENTIMENTO]
            entrada = tokenizador([textos[i] for i in indices], padding=True, truncation=True,
                                  max_length=config.MAX_TOKENS, return_tensors="pt").to(_DISPOSITIVO)
            probabilidades = torch.softmax(modelo(**entrada).logits, dim=-1).cpu().numpy()
            saida[indices] = probabilidades[:, colunas]
            feitas = inicio + len(indices)
            if progresso and (feitas // config.LOTE_SENTIMENTO) % 100 == 0:
                print(f"    {feitas:,}/{len(textos):,} frases ({time.time() - relogio:.0f} s)".replace(",", "."))
    return saida


def nota(probabilidades: np.ndarray) -> np.ndarray:
    """Uma nota de -1 a +1, como o compound do VADER: P(positivo) - P(negativo)."""
    return probabilidades[:, 2] - probabilidades[:, 0]


# ---------------------------------------------------------------------------
# Opção 7: as duas leituras de uma frase, passo a passo
# ---------------------------------------------------------------------------


def _palavras_do_lexico(frase: str) -> list[tuple[str, float]]:
    lexico = pipeline._vader().lexicon
    return [(p, lexico[p]) for p in RE_PALAVRA.findall(frase.lower()) if p in lexico]


def demonstrar(frases: list[str] | None = None) -> None:
    """Para cada frase: o que o VADER soma e o que o transformer calcula."""
    import torch

    tokenizador, modelo = carregar_transformer()
    colunas = _ordem_das_colunas()
    vader = pipeline._vader()
    print(f"  modelo: {config.MODELO_SENTIMENTO} ({sum(p.numel() for p in modelo.parameters()) / 1e6:.0f} milhões de parâmetros, na {_DISPOSITIVO.upper()})")
    for frase in frases or FRASES_DEMO:
        print(f"\n  FRASE: {frase}")
        compound = vader.polarity_scores(frase)["compound"]
        lexico = _palavras_do_lexico(frase)
        achadas = ", ".join(f"{p} ({v:+.1f})" for p, v in lexico) or "nenhuma"
        print(f"    VADER ......... palavras do léxico: {achadas}")
        print(f"                    compound {compound:+.3f} -> {classe_vader(compound)}")

        entrada = tokenizador(frase, return_tensors="pt").to(_DISPOSITIVO)
        pedacos = tokenizador.convert_ids_to_tokens(entrada["input_ids"][0])
        with torch.no_grad():
            logits = modelo(**entrada).logits[0]
        probabilidades = torch.softmax(logits, dim=-1).cpu().numpy()[colunas]
        logits = logits.cpu().numpy()[colunas]
        # O RoBERTa marca o espaço antes da palavra com "Ġ"; aqui ele vira "_".
        print(f"    transformer ... {len(pedacos)} tokens: {' '.join(p.replace('Ġ', '_') for p in pedacos)}")
        print(f"                    logits    neg {logits[0]:+.2f}  neu {logits[1]:+.2f}  pos {logits[2]:+.2f}")
        print(f"                    softmax   neg {probabilidades[0]:.0%}   neu {probabilidades[1]:.0%}   pos {probabilidades[2]:.0%}"
              f"   -> {CLASSES[int(np.argmax(probabilidades))]}")
    print()


# ---------------------------------------------------------------------------
# Opção 8: o corpus inteiro, VADER x transformer
# ---------------------------------------------------------------------------


def classificar_corpus(avaliadas: list[dict[str, Any]], forcar: bool = False) -> np.ndarray:
    """O transformer em todas as frases avaliadas, com cache (são dezenas de minutos na CPU)."""
    if not forcar and config.ARQ_TRANSFORMER.exists():
        with config.ARQ_TRANSFORMER.open(encoding="utf-8") as arquivo:
            cache = json.load(arquivo)
        if cache.get("n") == len(avaliadas) and cache.get("modelo") == config.MODELO_SENTIMENTO:
            print(f"  classificação lida do cache ({config.ARQ_TRANSFORMER.name})")
            return np.array(cache["probabilidades"], dtype=np.float32)
    carregar_transformer()
    print(f"  classificando {len(avaliadas):,} frases com o transformer na {_DISPOSITIVO.upper()}...".replace(",", "."))
    relogio = time.time()
    probabilidades = classificar([f["texto"] for f in avaliadas], progresso=True)
    print(f"  pronto em {time.time() - relogio:.0f} s")
    with config.ARQ_TRANSFORMER.open("w", encoding="utf-8") as arquivo:
        json.dump({"modelo": config.MODELO_SENTIMENTO, "n": len(avaliadas),
                   "probabilidades": np.round(probabilidades, 4).tolist()}, arquivo)
    return probabilidades


def _distribuicao(classes: list[str]) -> dict[str, int]:
    contagem = Counter(classes)
    return {c: contagem.get(c, 0) for c in CLASSES}


def _palavras_que_enganam(textos: list[str], vader: list[str], transformer: list[str], lado: str) -> list[dict[str, Any]]:
    """
    Palavras do léxico que mais aparecem quando o VADER vê `lado` e o
    transformer não vê. Para cada palavra: em quantas frases o VADER deu
    `lado` com ela, e em quantas dessas o transformer discordou.
    """
    lexico = pipeline._vader().lexicon
    sinal = 1 if lado == "positivo" else -1
    com_palavra: Counter[str] = Counter()
    discordou: Counter[str] = Counter()
    for texto, v, t in zip(textos, vader, transformer):
        if v != lado:
            continue
        for palavra in {p for p in RE_PALAVRA.findall(texto.lower()) if lexico.get(p, 0) * sinal > 0}:
            com_palavra[palavra] += 1
            if t != lado:
                discordou[palavra] += 1
    return [
        {"palavra": p, "valencia": lexico[p], "frases": com_palavra[p], "discordou": qtd,
         "taxa": round(qtd / com_palavra[p], 3)}
        for p, qtd in discordou.most_common(12)
    ]


def comparar(
    avaliadas: list[dict[str, Any]],
    probabilidades: np.ndarray,
    corpus: list[dict[str, Any]],
    temas: dict[str, Any],
    salvar: bool = True,
) -> dict[str, Any]:
    """Cruza as duas leituras: distribuição, matriz de confusão, kappa, palavras, ferramentas e temas."""
    from sklearn.metrics import cohen_kappa_score

    textos = [f["texto"] for f in avaliadas]
    vader = [classe_vader(f["vader"]) for f in avaliadas]
    transformer = [CLASSES[i] for i in probabilidades.argmax(axis=1)]
    notas = nota(probabilidades)
    matriz = [[sum(1 for v, t in zip(vader, transformer) if v == a and t == b) for b in CLASSES] for a in CLASSES]
    concordam = sum(matriz[i][i] for i in range(3))

    def resumo(indices: list[int]) -> dict[str, Any]:
        return {
            "frases": len(indices),
            "vader": _distribuicao([vader[i] for i in indices]),
            "transformer": _distribuicao([transformer[i] for i in indices]),
            "nota_vader": round(float(np.mean([avaliadas[i]["vader"] for i in indices])), 3) if indices else None,
            "nota_transformer": round(float(np.mean(notas[indices])), 3) if indices else None,
        }

    por_ferramenta = {}
    for nome in config.FERRAMENTAS_IA:
        indices = [i for i, f in enumerate(avaliadas) if nome in f["ferramentas"]]
        if len(indices) >= 30:
            por_ferramenta[nome] = resumo(indices)

    por_tema = {}
    if temas:
        tema_da_issue = temas["tema_da_issue"]
        for tema in temas["temas"]:
            indices = [i for i, f in enumerate(avaliadas) if tema_da_issue[f["issue"]] == tema["tema"]]
            por_tema[str(tema["tema"])] = resumo(indices)

    def exemplos(condicao, chave, n=6) -> list[dict[str, Any]]:
        escolhidos = sorted((i for i in range(len(avaliadas)) if condicao(i) and 8 <= avaliadas[i]["palavras"] <= 30), key=chave)[:n]
        return [{"texto": textos[i], "vader": avaliadas[i]["vader"], "neg": round(float(probabilidades[i, 0]), 3),
                 "neu": round(float(probabilidades[i, 1]), 3), "pos": round(float(probabilidades[i, 2]), 3),
                 "issue": corpus[avaliadas[i]["issue"]]["titulo"][:80]} for i in escolhidos]

    analise = {
        "estatisticas": {
            "issues": len(corpus),
            "frases_avaliadas": len(avaliadas),
            "concordancia_pct": round(concordam / len(avaliadas) * 100, 1),
            "kappa": round(float(cohen_kappa_score(vader, transformer)), 3),
            "nota_media_vader": round(float(np.mean([f["vader"] for f in avaliadas])), 3),
            "nota_media_transformer": round(float(np.mean(notas)), 3),
        },
        "distribuicao": {
            "vader_issue_inteira": _distribuicao([issue["sentimento"] for issue in corpus if issue.get("sentimento")]),
            "vader": _distribuicao(vader),
            "transformer": _distribuicao(transformer),
        },
        "matriz": {"linhas_vader": CLASSES, "colunas_transformer": CLASSES, "contagens": matriz},
        "palavras_que_enganam": {
            "positivo": _palavras_que_enganam(textos, vader, transformer, "positivo"),
            "negativo": _palavras_que_enganam(textos, vader, transformer, "negativo"),
        },
        "por_ferramenta": por_ferramenta,
        "por_tema": por_tema,
        "exemplos": {
            # O VADER vê elogio e o transformer, queixa (as mais confiantes primeiro).
            "vader_positivo_transformer_negativo": exemplos(
                lambda i: vader[i] == "positivo" and transformer[i] == "negativo", lambda i: -probabilidades[i, 0]),
            # O VADER vê queixa e o transformer, frase neutra.
            "vader_negativo_transformer_neutro": exemplos(
                lambda i: vader[i] == "negativo" and transformer[i] == "neutro", lambda i: -probabilidades[i, 1]),
            # O VADER não vê nada (compound 0) e o transformer vê queixa.
            "queixa_invisivel_ao_vader": exemplos(
                lambda i: avaliadas[i]["vader"] == 0 and transformer[i] == "negativo", lambda i: -probabilidades[i, 0]),
        },
    }
    if salvar:
        with config.ARQ_SENTIMENTO.open("w", encoding="utf-8") as arquivo:
            json.dump(analise, arquivo, ensure_ascii=False, indent=1)
        print(f"  salvo em {config.ARQ_SENTIMENTO.name}")
    return analise


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_SENTIMENTO.exists():
        return {}
    with config.ARQ_SENTIMENTO.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _pct(contagem: dict[str, int], classe: str) -> float:
    return contagem.get(classe, 0) / (sum(contagem.values()) or 1) * 100


def imprimir(analise: dict[str, Any], temas: dict[str, Any] | None = None) -> None:
    """Mostra a comparação no terminal, formatada para a apresentação."""
    if not analise:
        print("  Nenhuma análise salva. Rode a opção 8 primeiro.")
        return
    est = analise["estatisticas"]
    dist = analise["distribuicao"]
    print(f"\n  SENTIMENTO DE {est['frases_avaliadas']:,} FRASES: VADER x TRANSFORMER".replace(",", "."))
    print("  " + "-" * 64)
    print(f"    {'medidor':<34}{'positivo':>9}{'neutro':>8}{'negativo':>10}")
    for chave, rotulo in (("vader_issue_inteira", "VADER, issue inteira (etapa 1)"),
                          ("vader", "VADER, frase a frase"), ("transformer", "transformer, frase a frase")):
        d = dist[chave]
        print(f"    {rotulo:<34}{_pct(d, 'positivo'):>8.0f}%{_pct(d, 'neutro'):>7.0f}%{_pct(d, 'negativo'):>9.0f}%")
    print(f"\n    concordam em {est['concordancia_pct']:.0f}% das frases; kappa de Cohen {est['kappa']:.2f}")

    print("\n  MATRIZ DE CONFUSÃO (linhas: VADER; colunas: transformer)")
    print(f"    {'':<12}" + "".join(f"{c:>11}" for c in CLASSES))
    for classe, linha in zip(CLASSES, analise["matriz"]["contagens"]):
        print(f"    {classe:<12}" + "".join(f"{q:>11,}".replace(",", ".") for q in linha))

    for lado, titulo in (("positivo", "VÊ ELOGIO ONDE O TRANSFORMER NÃO VÊ"), ("negativo", "VÊ QUEIXA ONDE O TRANSFORMER NÃO VÊ")):
        print(f"\n  PALAVRAS COM QUE O VADER {titulo}")
        for item in analise["palavras_que_enganam"][lado][:6]:
            print(f"    {item['palavra']:<12} ({item['valencia']:+.1f})  o transformer discordou em {item['discordou']:>4} "
                  f"de {item['frases']:>4} frases ({item['taxa']:.0%})")

    print("\n  SENTIMENTO POR FERRAMENTA (frases que citam a ferramenta)")
    print(f"    {'ferramenta':<16}{'frases':>7}{'nota VADER':>12}{'nota transformer':>18}{'neg (transf.)':>15}")
    for nome, r in analise["por_ferramenta"].items():
        print(f"    {nome:<16}{r['frases']:>7}{r['nota_vader']:>+12.2f}{r['nota_transformer']:>+18.2f}{_pct(r['transformer'], 'negativo'):>14.0f}%")

    if analise["por_tema"] and temas:
        import embeddings

        print("\n  SENTIMENTO POR TEMA (transformer)")
        for tema in temas["temas"]:
            r = analise["por_tema"][str(tema["tema"])]
            print(f"    tema {tema['tema'] + 1:<2} {embeddings.nome_curto(tema):<34} {r['frases']:>6} frases  "
                  f"neg {_pct(r['transformer'], 'negativo'):>3.0f}%  pos {_pct(r['transformer'], 'positivo'):>3.0f}%")

    print("\n  EXEMPLOS: O VADER VÊ ELOGIO, O TRANSFORMER VÊ QUEIXA")
    for item in analise["exemplos"]["vader_positivo_transformer_negativo"][:3]:
        print(f"    [VADER {item['vader']:+.2f} | transformer neg {item['neg']:.0%}] {item['texto'][:130]}")
    print("\n  EXEMPLOS: QUEIXA QUE O VADER NÃO VÊ (COMPOUND 0,00)")
    for item in analise["exemplos"]["queixa_invisivel_ao_vader"][:3]:
        print(f"    [transformer neg {item['neg']:.0%}] {item['texto'][:130]}")
    print()


if __name__ == "__main__":
    demonstrar()
