"""
Etapa 2 - VISUALIZAÇÃO E DASHBOARD.

Mesmo padrão das outras etapas: todo gráfico sai como PNG na pasta `saida/`,
pronto para o slide, com o backend "Agg" do matplotlib (desenha direto no
arquivo, sem abrir janela que travaria a apresentação).

Gráficos gerados:
  01  o vetor estático não conhece as ferramentas (palavras que dividem o vetor)
  02  similaridade entre frases: vetor estático x contextual
  03  mapa semântico das issues (t-SNE), colorido pelo tema
  04  os temas: tamanho, termos e exemplos
  05  matriz de confusão: VADER x transformer
  06  três medidas de sentimento das mesmas issues
  07  as palavras que enganam o VADER
  08  sentimento por ferramenta e por tema
  09  painel da etapa 2 (dashboard)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # precisa vir antes de importar o pyplot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

import config
import embeddings

plt.rcParams.update(
    {
        "figure.dpi": config.DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,  # a grade fica atrás das barras
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)

CLASSES = ["positivo", "neutro", "negativo"]  # a ordem das fatias nos gráficos
COR_ROTULO = {"neutro": "white"}
# Uma cor por tema, iguais no mapa, nas barras e no painel.
CORES_TEMA = ["#3D5A80", "#EE9B00", "#2A9D8F", "#B07CC6", "#D1495B", "#5B8DB8", "#8D6E63", "#2E9E5B",
              "#E76F51", "#264653", "#9C6644", "#6D597A"]


def _num(valor: float, casas: int = 0, sinal: bool = False) -> str:
    """Número escrito em português: vírgula decimal e ponto no milhar (igual às outras etapas)."""
    texto = f"{valor:{'+' if sinal else ''},.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


VIRGULA = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))


def _salvar(figura, nome: str) -> Path:
    for eixo in figura.axes:
        for eixo_xy in (eixo.xaxis, eixo.yaxis):
            if type(eixo_xy.get_major_formatter()) is ScalarFormatter:
                eixo_xy.set_major_formatter(VIRGULA)
    caminho = config.DIR_SAIDA / nome
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)
    print(f"    {nome}")
    return caminho


def _pct(contagem: dict[str, int]) -> dict[str, float]:
    total = sum(contagem.values()) or 1
    return {c: contagem.get(c, 0) / total * 100 for c in CLASSES}


def _empilhado(eixo, nomes: list[str], distribuicoes: list[dict[str, int]], notas: list[str] | None = None) -> None:
    """Barras horizontais empilhadas em percentual, com o rótulo dentro da fatia."""
    base = np.zeros(len(nomes))
    for classe in CLASSES:
        fatias = np.array([_pct(d)[classe] for d in distribuicoes])
        barras = eixo.barh(nomes, fatias, left=base, color=config.PALETA[classe], label=classe, edgecolor="white")
        eixo.bar_label(barras, labels=[f"{f:.0f}%" if f >= 6 else "" for f in fatias], label_type="center",
                       color=COR_ROTULO.get(classe, "white"), fontsize=8, fontweight="bold")
        base += fatias
    if notas:
        for indice, texto in enumerate(notas):
            eixo.text(101.5, indice, texto, va="center", fontsize=8, color="#444")
        eixo.set_xlim(0, 124)
        eixo.spines["bottom"].set_bounds(0, 100)  # a linha do eixo para no 100; depois vem a nota
    else:
        eixo.set_xlim(0, 100)
    eixo.set_xticks([0, 25, 50, 75, 100])
    eixo.grid(axis="y", visible=False)


def _cartao(eixo, titulo: str, linhas: list[tuple[str, str]], cor: str) -> None:
    eixo.axis("off")
    eixo.set_title(titulo, fontweight="bold", loc="left")
    passo = 0.8 / max(len(linhas) - 1, 1)
    for indice, (rotulo, valor) in enumerate(linhas):
        altura = 0.9 - indice * passo
        eixo.text(0.02, altura, rotulo, fontsize=10, color="#555", transform=eixo.transAxes)
        eixo.text(0.98, altura, valor, fontsize=14, fontweight="bold", ha="right", color=cor, transform=eixo.transAxes)


def _nome_tema(temas: dict[str, Any], numero: int, termos: int = 3) -> str:
    return f"{numero + 1}. {embeddings.nome_curto(temas['temas'][numero], termos)}"


# ---------------------------------------------------------------------------
# 01 e 02 Embeddings
# ---------------------------------------------------------------------------


def vetores_estaticos(palavras: dict[str, Any]) -> Path:
    """Cada ferramenta e as palavras com que ela divide o vetor no en_core_web_md."""
    itens = list(palavras["compartilhados"].items())
    figura, eixo = plt.subplots(figsize=(12, 0.62 * len(itens) + 1.3))
    eixo.axis("off")
    for linha, (palavra, outras) in enumerate(itens):
        y = 1 - (linha + 0.5) / len(itens)
        eixo.text(0.0, y, palavra, fontsize=13, fontweight="bold", color=config.PALETA["destaque"], va="center", transform=eixo.transAxes)
        if outras is None:
            texto, cor = "sem vetor: a palavra é mais nova que o modelo", config.PALETA["negativo"]
        else:
            quantas = palavras["quantos"][palavra]
            texto, cor = f"mesmo vetor que {quantas} palavras: {', '.join(outras[:9])}{'…' if quantas > 9 else ''}", "#333"
        eixo.text(0.13, y, texto, fontsize=10.5, color=cor, va="center", transform=eixo.transAxes)
    figura.suptitle("O vetor estático do spaCy não conhece as ferramentas de IA", fontsize=14, fontweight="bold", x=0.02, ha="left")
    figura.text(0.02, 0.9, "en_core_web_md: 20 mil vetores de 300 números para 684 mil palavras. "
                "Palavra rara divide o vetor de uma palavra frequente.", fontsize=9.5, color="#555", ha="left")
    figura.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(figura, "01_vetores_estaticos.png")


def similaridade_frases(palavras: dict[str, Any]) -> Path:
    """As cinco frases da demo nas duas matrizes, e os pares sorteados do corpus."""
    frases = palavras["frases_demo"]
    figura = plt.figure(figsize=(16, 6.2))
    grade = figura.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.35)
    rotulos = [f"[{i}]" for i in range(1, len(frases) + 1)]
    for coluna, (chave, titulo) in enumerate((("estatico", "Estático (média dos vetores de palavra)"),
                                              ("contextual", "Contextual (MiniLM)"))):
        eixo = figura.add_subplot(grade[0, coluna])
        matriz = np.array(palavras[chave])
        eixo.imshow(matriz, cmap="YlGnBu", vmin=0, vmax=1)
        for i in range(len(frases)):
            for j in range(len(frases)):
                eixo.text(j, i, _num(matriz[i, j], 2), ha="center", va="center", fontsize=9,
                          color="white" if matriz[i, j] > 0.6 else "#222")
        eixo.set_xticks(range(len(frases)), rotulos)
        eixo.set_yticks(range(len(frases)), rotulos)
        eixo.grid(False)
        eixo.set_title(titulo, fontweight="bold")

    eixo = figura.add_subplot(grade[0, 2])
    sorteio = palavras.get("sorteio")
    if sorteio:
        for chave, rotulo in (("estatico", "estático"), ("contextual", "contextual")):
            eixo.hist(sorteio[chave]["valores"], bins=np.linspace(-0.2, 1, 37), alpha=0.75,
                      label=f"{rotulo}: média {_num(sorteio[chave]['media'], 2)}",
                      color=config.PALETA["vader" if chave == "estatico" else "transformer"])
            # A linha da média para onde começa a folga da legenda.
            eixo.axvline(sorteio[chave]["media"], ymax=1 / 1.3, color="#333", linestyle="--", linewidth=1)
        # Folga no alto do gráfico para a legenda não cobrir nenhuma barra.
        eixo.set_ylim(top=eixo.get_ylim()[1] * 1.3)
        eixo.legend(frameon=False, loc="upper center", title="linha tracejada = média", title_fontsize=8)
        eixo.set_xlabel("similaridade do cosseno")
        eixo.set_ylabel("pares de frases")
        eixo.set_title(f"{sorteio['n']} pares sorteados do corpus, sem relação", fontweight="bold")
    legenda = "\n".join(textwrap.shorten(f"[{i}] {f}", 60) for i, f in enumerate(frases, start=1))
    figura.text(0.02, 0.08, legenda, fontsize=9, family="monospace", va="top")
    figura.suptitle("Embeddings: o vetor estático acha tudo parecido; o contextual separa as frases",
                    fontsize=14, fontweight="bold")
    return _salvar(figura, "02_similaridade_frases.png")


# ---------------------------------------------------------------------------
# 03 e 04 Temas
# ---------------------------------------------------------------------------


def mapa_temas(temas: dict[str, Any]) -> Path:
    plano = np.array(temas["plano"])
    rotulos = np.array(temas["tema_da_issue"])
    figura, eixo = plt.subplots(figsize=(13, 9))
    for tema in temas["temas"]:
        t = tema["tema"]
        pontos = plano[rotulos == t]
        eixo.scatter(pontos[:, 0], pontos[:, 1], s=16, alpha=0.75, color=CORES_TEMA[t % len(CORES_TEMA)],
                     label=f"{_nome_tema(temas, t)} ({tema['issues']})", edgecolor="none")
        centro = np.median(pontos, axis=0)
        eixo.text(centro[0], centro[1], str(t + 1), fontsize=14, fontweight="bold", ha="center", va="center",
                  color="white", bbox={"boxstyle": "circle,pad=0.3", "facecolor": CORES_TEMA[t % len(CORES_TEMA)], "edgecolor": "white"})
    eixo.axis("off")  # os eixos do t-SNE não têm unidade
    eixo.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9, title="tema (issues)")
    eixo.set_title(f"Mapa semântico das {len(rotulos)} issues: cada ponto é uma issue; perto = assunto parecido",
                   fontsize=13, fontweight="bold", loc="left")
    figura.text(0.125, 0.07, "Vetores de 384 números do MiniLM, desenhados no plano pelo t-SNE. "
                "Os eixos não têm unidade; só a vizinhança tem sentido.", fontsize=9, color="#555")
    return _salvar(figura, "03_mapa_temas.png")


def lista_temas(temas: dict[str, Any]) -> Path:
    itens = temas["temas"]
    figura, eixo = plt.subplots(figsize=(15, 0.95 * len(itens) + 1.2))
    nomes = [f"{t['tema'] + 1}" for t in itens][::-1]
    valores = [t["issues"] for t in itens][::-1]
    cores = [CORES_TEMA[t["tema"] % len(CORES_TEMA)] for t in itens][::-1]
    barras = eixo.barh(nomes, valores, color=cores)
    eixo.bar_label(barras, padding=4, fontsize=9)
    maximo = max(valores)
    for posicao, tema in enumerate(itens[::-1]):
        termos = ", ".join(l for l, _ in tema["termos"])
        exemplo = textwrap.shorten(tema["exemplos"][0], 70)
        eixo.text(maximo * 1.12, posicao + 0.12, termos, fontsize=10, fontweight="bold", va="center")
        eixo.text(maximo * 1.12, posicao - 0.25, f"ex.: {exemplo}", fontsize=8.5, color="#555", va="center")
    eixo.set_xlim(0, maximo * 3.4)
    eixo.set_xticks([])
    eixo.spines["bottom"].set_visible(False)
    eixo.grid(False)
    eixo.set_ylabel("tema")
    eixo.set_title(f"Os {temas['k']} temas achados pelo KMeans: issues por tema, termos característicos (c-TF-IDF) e um exemplo",
                   fontsize=12, fontweight="bold", loc="left")
    return _salvar(figura, "04_temas.png")


# ---------------------------------------------------------------------------
# 05 a 08 Sentimento
# ---------------------------------------------------------------------------


def matriz_confusao(analise: dict[str, Any]) -> Path:
    contagens = np.array(analise["matriz"]["contagens"])  # linhas e colunas na ordem negativo, neutro, positivo
    ordem = analise["matriz"]["linhas_vader"]
    por_linha = contagens / contagens.sum(axis=1, keepdims=True) * 100
    figura, eixo = plt.subplots(figsize=(8.2, 6.6))
    eixo.imshow(por_linha, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            eixo.text(j, i, f"{_num(contagens[i, j])}\n{por_linha[i, j]:.0f}% da linha", ha="center", va="center", fontsize=10,
                      color="white" if por_linha[i, j] > 55 else "#222", fontweight="bold" if i == j else "normal")
    eixo.set_xticks(range(3), ordem)
    eixo.set_yticks(range(3), ordem)
    eixo.set_xlabel("transformer (RoBERTa)", fontweight="bold")
    eixo.set_ylabel("VADER, frase a frase", fontweight="bold")
    eixo.grid(False)
    est = analise["estatisticas"]
    eixo.set_title(f"As duas leituras das mesmas {_num(est['frases_avaliadas'])} frases\n"
                   f"concordam em {est['concordancia_pct']:.0f}%  ·  kappa de Cohen {_num(est['kappa'], 2)}",
                   fontsize=12, fontweight="bold")
    return _salvar(figura, "05_vader_x_transformer.png")


def tres_medidas(analise: dict[str, Any]) -> Path:
    dist = analise["distribuicao"]
    nomes = ["VADER, issue inteira\n(como a etapa 1 mediu)", "VADER, frase a frase", "transformer, frase a frase"][::-1]
    chaves = ["vader_issue_inteira", "vader", "transformer"][::-1]
    figura, eixo = plt.subplots(figsize=(12, 4.2))
    _empilhado(eixo, nomes, [dist[c] for c in chaves])
    eixo.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False)
    eixo.set_xlabel("percentual")
    figura.suptitle("O mesmo corpus, três medidas de sentimento", fontsize=13, fontweight="bold", y=1.04)
    return _salvar(figura, "06_sentimento_tres_medidas.png")


def palavras_que_enganam(analise: dict[str, Any]) -> Path:
    figura, eixos = plt.subplots(1, 2, figsize=(15, 5.6))
    for eixo, (lado, titulo, cor) in zip(eixos, (
        ("positivo", "O VADER vê elogio e o transformer não", config.PALETA["positivo"]),
        ("negativo", "O VADER vê queixa e o transformer não", config.PALETA["negativo"]),
    )):
        itens = analise["palavras_que_enganam"][lado][:10][::-1]
        barras = eixo.barh([f"{i['palavra']} ({_num(i['valencia'], 1, sinal=True)})" for i in itens],
                           [i["discordou"] for i in itens], color=cor)
        eixo.bar_label(barras, labels=[f"{i['discordou']} de {i['frases']} ({i['taxa']:.0%})" for i in itens], padding=3, fontsize=8)
        eixo.margins(x=0.32)
        eixo.set_xlabel("frases em que o transformer discordou")
        eixo.set_title(titulo, fontweight="bold")
        eixo.grid(axis="y", visible=False)
    figura.suptitle("As palavras do léxico que enganam o VADER nas issues (entre parênteses, a nota da palavra no léxico)",
                    fontsize=13, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "07_palavras_que_enganam.png")


def sentimento_tema_ferramenta(analise: dict[str, Any], temas: dict[str, Any]) -> Path:
    figura, (esquerda, direita) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ferramentas = sorted(analise["por_ferramenta"].items(), key=lambda par: par[1]["nota_transformer"])
    _empilhado(esquerda, [f"{n} ({_num(r['frases'])} frases)" for n, r in ferramentas], [r["transformer"] for _, r in ferramentas],
               notas=[f"nota {_num(r['nota_transformer'], 2, sinal=True)}" for _, r in ferramentas])
    esquerda.set_title("Por ferramenta citada na frase", fontweight="bold", pad=24)
    esquerda.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=3, frameon=False, fontsize=8)
    esquerda.set_xlabel("percentual (transformer)")

    if analise["por_tema"]:
        itens = sorted(temas["temas"], key=lambda t: analise["por_tema"][str(t["tema"])]["nota_transformer"])
        _empilhado(direita, [f"{_nome_tema(temas, t['tema'])} ({_num(analise['por_tema'][str(t['tema'])]['frases'])} frases)" for t in itens],
                   [analise["por_tema"][str(t["tema"])]["transformer"] for t in itens],
                   notas=[f"nota {_num(analise['por_tema'][str(t['tema'])]['nota_transformer'], 2, sinal=True)}" for t in itens])
        direita.set_title("Por tema da issue", fontweight="bold", pad=24)
        direita.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=3, frameon=False, fontsize=8)
        direita.set_xlabel("percentual (transformer)")
    figura.suptitle("Sentimento das frases pelo transformer (nota = probabilidade de positivo menos a de negativo)",
                    fontsize=13, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "08_sentimento_ferramenta_tema.png")


# ---------------------------------------------------------------------------
# 09 Painel
# ---------------------------------------------------------------------------


def painel(pipe: dict[str, Any], temas: dict[str, Any], analise: dict[str, Any]) -> Path:
    """Painel único com a visão geral da etapa: o dashboard pedido no enunciado."""
    figura = plt.figure(figsize=(16, 9))
    grade = figura.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
    est, sent = pipe["estatisticas"], analise["estatisticas"]
    _cartao(figura.add_subplot(grade[0, 0]), "O corpus pelo pipeline", [
        ("issues da etapa 1", _num(est["issues"])),
        ("tokens", _num(est["tokens"])),
        ("frases (parser do spaCy)", _num(est["frases"])),
        ("frases com ferramenta", _num(est["frases_com_ferramenta"])),
        ("temas (KMeans)", str(temas["k"])),
        ("VADER e transformer concordam", f"{sent['concordancia_pct']:.0f}%"),
    ], config.PALETA["destaque"])

    eixo = figura.add_subplot(grade[0, 1])
    plano, rotulos = np.array(temas["plano"]), np.array(temas["tema_da_issue"])
    for tema in temas["temas"]:
        pontos = plano[rotulos == tema["tema"]]
        eixo.scatter(pontos[:, 0], pontos[:, 1], s=5, color=CORES_TEMA[tema["tema"] % len(CORES_TEMA)], edgecolor="none")
    eixo.axis("off")
    eixo.set_title(f"Mapa semântico: {temas['k']} temas", fontweight="bold")

    eixo = figura.add_subplot(grade[0, 2])
    itens = temas["temas"][::-1]
    barras = eixo.barh([_nome_tema(temas, t["tema"], 2) for t in itens], [t["issues"] for t in itens],
                       color=[CORES_TEMA[t["tema"] % len(CORES_TEMA)] for t in itens])
    eixo.bar_label(barras, padding=3, fontsize=8)
    eixo.margins(x=0.18)
    eixo.tick_params(axis="y", labelsize=8)
    eixo.set_title("Issues por tema", fontweight="bold")
    eixo.grid(axis="y", visible=False)

    eixo = figura.add_subplot(grade[1, 0])
    dist = analise["distribuicao"]
    _empilhado(eixo, ["transformer", "VADER"], [dist["transformer"], dist["vader"]])
    eixo.set_title("Sentimento frase a frase", fontweight="bold", pad=22)
    eixo.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False, fontsize=8)

    eixo = figura.add_subplot(grade[1, 1])
    # O lado do elogio: palavras que puxam o VADER para o positivo sem o transformer concordar.
    itens = analise["palavras_que_enganam"]["positivo"][:6][::-1]
    barras = eixo.barh([i["palavra"] for i in itens], [i["discordou"] for i in itens], color=config.PALETA["positivo"])
    eixo.bar_label(barras, labels=[f"{i['taxa']:.0%} de {_num(i['frases'])}" for i in itens], padding=3, fontsize=8)
    eixo.margins(x=0.25)
    eixo.set_title("Palavras que o VADER lê como elogio", fontweight="bold")
    eixo.set_xlabel("frases em que o transformer discordou")
    eixo.grid(axis="y", visible=False)

    # A parcela de frases negativas separa as ferramentas. Com 85% de frases
    # neutras, a nota média de todas fica perto de zero.
    eixo = figura.add_subplot(grade[1, 2])
    negativas = {n: _pct(r["transformer"])["negativo"] for n, r in analise["por_ferramenta"].items()}
    ferramentas = sorted(analise["por_ferramenta"].items(), key=lambda par: negativas[par[0]])
    barras = eixo.barh([f"{n} ({_num(r['frases'])})" for n, r in ferramentas], [negativas[n] for n, _ in ferramentas],
                       color=config.PALETA["negativo"])
    eixo.bar_label(barras, labels=[f"{negativas[n]:.0f}%" for n, _ in ferramentas], padding=3, fontsize=8)
    eixo.margins(x=0.15)
    eixo.tick_params(axis="y", labelsize=8)
    eixo.set_title("Frases negativas por ferramenta (transformer)", fontweight="bold")
    eixo.set_xlabel("% das frases que citam a ferramenta")
    eixo.grid(axis="y", visible=False)

    figura.suptitle("DevIA Radar · Etapa 2  |  O que desenvolvedores QUEREM DIZER sobre IA no desenvolvimento de software",
                    fontsize=15, fontweight="bold")
    return _salvar(figura, "09_painel_dashboard.png")


def gerar_todos(pipe: dict[str, Any], palavras: dict[str, Any], temas: dict[str, Any], analise: dict[str, Any]) -> None:
    print("  gerando gráficos em saida/ ...")
    vetores_estaticos(palavras)
    similaridade_frases(palavras)
    if temas:
        mapa_temas(temas)
        lista_temas(temas)
    else:
        print("    [pulado] 03 e 04 precisam dos temas (opção 5)")
    matriz_confusao(analise)
    tres_medidas(analise)
    palavras_que_enganam(analise)
    sentimento_tema_ferramenta(analise, temas)
    if pipe and temas:
        painel(pipe, temas, analise)
    print(f"  pronto: {config.DIR_SAIDA}")
