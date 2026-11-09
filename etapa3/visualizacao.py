"""
Etapa 3 - VISUALIZAÇÃO, DASHBOARD E PAINEL DAS TRÊS ETAPAS.

Mesmo padrão das etapas 1 e 2: todo gráfico sai como PNG na pasta `saida/`,
pronto para o slide, com o backend "Agg" do matplotlib (desenha direto no
arquivo, sem abrir janela que travaria a apresentação).

Gráficos gerados:
  01  mosaico das telas de cada palestra, coloridas pelo tipo
  02  do vídeo aos números: quadro, regiões, diferença, bordas e rostos
  03  linha do tempo de uma palestra (tela, fala e câmera)
  04  perfil visual de cada palestra
  05  ferramentas: tela x fala x legenda oficial
  06  a fala acompanha a tela? (velocidade e sentimento por tipo de tela)
  07  o que está escrito x o que é dito (nuvens de palavras)
  08  painel da etapa 3 (dashboard)
  09  painel final das três etapas: texto, fala e vídeo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # precisa vir antes de importar o pyplot

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, ScalarFormatter

import config

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

ORDEM_SENTIMENTO = ["positivo", "neutro", "negativo"]
TIPOS = list(config.PALETA_TELA)  # texto, visual, código, interface, movimento
COR_ROTULO = {"movimento": "#333"}  # rótulo dentro da fatia: branco, menos no cinza claro


def _classe(compound: float) -> str:
    if compound >= config.LIMITE_POSITIVO:
        return "positivo"
    if compound <= config.LIMITE_NEGATIVO:
        return "negativo"
    return "neutro"


def _num(valor: float, casas: int = 0, sinal: bool = False) -> str:
    """Número escrito em português: vírgula decimal e ponto no milhar (igual à etapa 2)."""
    texto = f"{valor:{'+' if sinal else ''},.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# Marcas dos eixos numéricos com vírgula decimal (0,5 em vez de 0.5).
VIRGULA = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))


def _salvar(figura, nome: str) -> Path:
    # Só troca o formatador padrão: eixo de categoria tem formatador próprio.
    for eixo in figura.axes:
        for eixo_xy in (eixo.xaxis, eixo.yaxis):
            if type(eixo_xy.get_major_formatter()) is ScalarFormatter:
                eixo_xy.set_major_formatter(VIRGULA)
    caminho = config.DIR_SAIDA / nome
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)
    print(f"    {nome}")
    return caminho


def _curto(titulo: str, limite: int = 34) -> str:
    """Título de palestra no eixo: a parte antes dos dois-pontos, se ela disser algo."""
    antes = titulo.split(":")[0].strip()
    nome = antes if len(antes) >= 14 else titulo
    return nome if len(nome) <= limite else nome[: limite - 1] + "…"


def _rgb(imagem: np.ndarray) -> np.ndarray:
    """O OpenCV guarda BGR; o matplotlib desenha RGB."""
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def _empilhado(eixo, nomes, distribuicoes, ordem, cores, medias=None):
    """Barras horizontais empilhadas em percentual, com o rótulo dentro da fatia."""
    base = np.zeros(len(nomes))
    for rotulo in ordem:
        fatias = []
        for dist in distribuicoes:
            total = sum(dist.values()) or 1
            fatias.append(dist.get(rotulo, 0) / total * 100)
        barras = eixo.barh(nomes, fatias, left=base, color=cores[rotulo], label=rotulo, edgecolor="white")
        eixo.bar_label(barras, labels=[f"{f:.0f}%" if f >= 7 else "" for f in fatias], label_type="center",
                       color=COR_ROTULO.get(rotulo, "white"), fontsize=8, fontweight="bold")
        base += np.array(fatias)
    if medias:
        for indice, media in enumerate(medias):
            if media is not None:
                eixo.text(101.5, indice, f"média {_num(media, 2, sinal=True)}", va="center", fontsize=8, color="#444")
        eixo.set_xlim(0, 120)
    else:
        eixo.set_xlim(0, 100)
    eixo.set_xticks([0, 25, 50, 75, 100])
    eixo.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# 01 Mosaico das telas
# ---------------------------------------------------------------------------


def mosaico_telas(palestras: list[dict[str, Any]], visual: dict[str, Any], lidas: dict[str, Any], por_linha: int = 8) -> Path | None:
    """Oito telas de cada palestra, espalhadas do começo ao fim, com a moldura na cor do tipo."""
    linhas = [p for p in palestras if p["id"] in lidas and p["id"] in visual.get("palestras", {})]
    if not linhas:
        print("    [pulado] 01 precisa das telas lidas (opções 4 e 6)")
        return None
    figura, eixos = plt.subplots(len(linhas), por_linha, figsize=(16, 1.32 * len(linhas) + 1.0))
    eixos = np.atleast_2d(eixos)
    for r, palestra in enumerate(linhas):
        telas = visual["palestras"][palestra["id"]]["telas"]
        tipos = {l["numero"]: l["tipo"] for l in lidas[palestra["id"]]}
        posicoes = np.linspace(0, len(telas) - 1, min(por_linha, len(telas))).round().astype(int)
        escolhidas = [telas[i] for i in posicoes]
        for c in range(por_linha):
            eixo = eixos[r, c]
            eixo.set_xticks([])
            eixo.set_yticks([])
            eixo.grid(False)
            if c >= len(escolhidas):
                eixo.axis("off")
                continue
            tela = escolhidas[c]
            imagem = cv2.imread(str(config.DIR_TELAS / palestra["id"] / tela["arquivo"]))
            eixo.imshow(_rgb(cv2.resize(imagem, (320, 180), interpolation=cv2.INTER_AREA)))
            tipo = tipos.get(tela["numero"], "texto")
            for lado in eixo.spines.values():
                lado.set_visible(True)
                lado.set_color(config.PALETA_TELA[tipo])
                lado.set_linewidth(3.5)
            minuto, segundo = divmod(int(tela["inicio"]), 60)
            eixo.set_title(f"{minuto}:{segundo:02d}", fontsize=7, pad=2)
        eixos[r, 0].set_ylabel(_curto(palestra["titulo"], 26), rotation=0, ha="right", va="center", fontsize=8)
    parados = [t for t in TIPOS if t != "movimento"]  # tela de movimento não vira imagem
    # Título numa linha e legenda na linha de baixo, os dois alinhados à esquerda.
    figura.suptitle("O que as palestras mostram: 8 telas de cada uma, com a moldura na cor do tipo lido pelo OCR",
                    fontsize=13, fontweight="bold", x=0.02, y=0.995, ha="left")
    figura.legend(
        handles=[Patch(color=config.PALETA_TELA[t], label=t) for t in parados],
        loc="upper left", ncols=len(parados), frameon=False, bbox_to_anchor=(0.015, 0.97),
    )
    figura.tight_layout(rect=(0, 0, 1, 0.955))
    return _salvar(figura, "01_mosaico_telas.png")


# ---------------------------------------------------------------------------
# 02 Do vídeo aos números (um quadro)
# ---------------------------------------------------------------------------


def figura_quadro(demo: dict[str, Any]) -> Path:
    """
    O mesmo segundo de vídeo desenhado em quatro camadas de análise.

    A série de diferenças (painel 3) vem da análise da palestra inteira; o
    resto é calculado na hora, a partir de um quadro só.
    """
    import quadros

    palestra, quadro = demo["palestra"], demo["quadro"]
    altura, largura = quadro.shape[:2]
    figura = plt.figure(figsize=(15, 8.8))
    grade = figura.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.3, wspace=0.1)

    eixo = figura.add_subplot(grade[0, :2])
    eixo.imshow(_rgb(quadro))
    for (x0, y0, x1, y1), cor, rotulo in (
        (config.REGIAO_SLIDE, config.PALETA["slide"], "slide"),
        (config.REGIAO_CAMERA, config.PALETA["video"], "câmera"),
    ):
        eixo.add_patch(Rectangle((x0 * largura, y0 * altura), (x1 - x0) * largura, (y1 - y0) * altura,
                                 fill=False, edgecolor=cor, linewidth=2.5))
        eixo.text(x0 * largura + 14, y0 * altura + 50, rotulo, color="white", fontsize=10, fontweight="bold",
                  bbox={"facecolor": cor, "edgecolor": "none", "pad": 3})
    eixo.set_title(f"1 e 2. Um quadro é uma matriz de {altura} x {largura} pixels x 3 cores; recortamos slide e câmera",
                   loc="left", fontweight="bold", fontsize=10)
    eixo.axis("off")

    eixo = figura.add_subplot(grade[0, 2])
    eixo.imshow(_rgb(demo["camera"]))
    for x, y, w, h in demo["rostos_haar"]:
        eixo.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=config.PALETA["negativo"], linewidth=2, linestyle="--"))
    for linha in demo["rostos_yunet"]:
        x, y, w, h = linha[:4]
        eixo.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#4CFF7A", linewidth=2))
        eixo.text(x, y - 6, _num(linha[-1], 2), color="#4CFF7A", fontsize=8, fontweight="bold")
    eixo.set_title(f"5. Rostos na câmera\nYuNet (rede neural, verde): {len(demo['rostos_yunet'])}  ·  "
                   f"Haar (2001, vermelho): {len(demo['rostos_haar'])}", loc="left", fontweight="bold", fontsize=10)
    eixo.axis("off")

    eixo = figura.add_subplot(grade[1, :2])
    serie = quadros.carregar_analise().get("palestras", {}).get(palestra["id"], {}).get("serie")
    if serie:
        minutos = np.array(serie["t"]) / 60
        dif = np.array(serie["dif_slide"])
        eixo.plot(minutos, dif, color=config.PALETA["slide"], linewidth=0.8)
        eixo.axhline(config.LIMIAR_TROCA, color=config.PALETA["negativo"], linestyle="--", linewidth=1,
                     label=f"limiar de troca ({_num(config.LIMIAR_TROCA)})")
        eixo.legend(loc="upper right", fontsize=8, frameon=False)  # em cima, onde não há picos
        eixo.axvline(demo["segundo"] / 60, color=config.PALETA["apoio"], linewidth=1.5)
        eixo.text(demo["segundo"] / 60, dif.max() * 0.95, "  este quadro", color=config.PALETA["apoio"], fontsize=8)
        eixo.set_xlabel("minuto da palestra")
        eixo.set_ylabel("diferença média (0 a 255)")
        eixo.set_title(f"3. Diferença do slide para o segundo anterior: os {int((dif > config.LIMIAR_TROCA).sum())} "
                       f"picos acima do limiar são trocas de slide", loc="left", fontweight="bold", fontsize=10)
    else:
        eixo.axis("off")
        eixo.text(0.5, 0.5, f"diferença neste segundo: {_num(demo['dif_slide'], 2)}\n(rode a opção 4 para ver a palestra inteira)",
                  ha="center", va="center", transform=eixo.transAxes)

    eixo = figura.add_subplot(grade[1, 2])
    eixo.imshow(demo["bordas"], cmap="gray")
    eixo.set_title(f"4. Bordas (Canny): {_num((demo['bordas'] > 0).mean() * 100, 1)}% dos pixels do slide",
                   loc="left", fontweight="bold", fontsize=10)
    eixo.axis("off")

    minuto, segundo = divmod(int(demo["segundo"]), 60)
    figura.suptitle(f"Do vídeo aos números · {_curto(palestra['titulo'], 50)}, quadro em {minuto}:{segundo:02d}",
                    fontsize=13, fontweight="bold")
    return _salvar(figura, "02_do_video_aos_numeros.png")


# ---------------------------------------------------------------------------
# 03 Linha do tempo de uma palestra
# ---------------------------------------------------------------------------


def linha_do_tempo(analise: dict[str, Any], visual: dict[str, Any], lidas: dict[str, Any], palestra_id: str) -> Path | None:
    import conteudo

    v = visual.get("palestras", {}).get(palestra_id)
    info = analise["por_palestra"].get(palestra_id)
    if not (v and info and palestra_id in lidas):
        print("    [pulado] 03 precisa da palestra da demo analisada")
        return None
    por_segundo = conteudo.tipo_por_segundo(v, lidas[palestra_id])
    frases = [f for f in analise["frases"] if f["palestra"] == palestra_id]
    minutos = np.array(v["serie"]["t"]) / 60

    figura, (topo, meio, baixo) = plt.subplots(3, 1, figsize=(13, 8.4), sharex=True,
                                              gridspec_kw={"height_ratios": [1.1, 1.2, 0.8]})
    inicio = 0
    for fim in range(1, len(por_segundo) + 1):
        if fim == len(por_segundo) or por_segundo[fim] != por_segundo[inicio]:
            if por_segundo[inicio]:
                topo.axvspan(inicio / 60, fim / 60, color=config.PALETA_TELA[por_segundo[inicio]], alpha=0.45, linewidth=0)
            inicio = fim
    topo.plot(minutos, v["serie"]["dif_slide"], color="#222", linewidth=0.7)
    topo.axhline(config.LIMIAR_TROCA, color=config.PALETA["negativo"], linestyle="--", linewidth=0.9)
    topo.set_ylabel("diferença do slide")
    topo.set_title("Tela: fundo na cor do tipo de tela · linha = diferença para o segundo anterior", loc="left", fontweight="bold")
    topo.legend(handles=[Patch(color=config.PALETA_TELA[t], alpha=0.45, label=t) for t in TIPOS],
                ncols=len(TIPOS), loc="upper right", fontsize=8, framealpha=0.9)

    for f in frases:
        if f["avaliado"]:
            meio.bar(f["inicio"] / 60, f["compound"], width=max((f["fim"] - f["inicio"]) / 60, 0.05),
                     color=config.PALETA[_classe(f["compound"])], align="edge")
    # Toda menção vira uma linha. O nome só é escrito quando está a mais de
    # 0,35 minuto do último nome escrito, senão os rótulos se encavalam.
    citadas = [f for f in frases if f["ferramentas"]]
    ultimo = -np.inf
    for f in citadas:
        minuto = f["inicio"] / 60
        meio.axvline(minuto, color=config.PALETA["apoio"], alpha=0.4, linewidth=0.8)
        if minuto - ultimo >= 0.35:
            # Só o primeiro nome, mais "+N" quando a frase cita outras.
            extras = len(f["ferramentas"]) - 1
            nome = f["ferramentas"][0] + (f" +{extras}" if extras else "")
            meio.annotate(nome, (minuto, 1.02), xycoords=("data", "axes fraction"),
                          rotation=60, fontsize=7, color=config.PALETA["apoio"], ha="left")
            ultimo = minuto
    meio.axhline(0, color="#999", linewidth=0.8)
    meio.set_ylim(-1, 1)
    meio.set_ylabel("compound (VADER)")
    meio.set_title(f"Fala: sentimento de cada frase · {len(citadas)} linhas laranja = frase que cita ferramenta de IA",
                   loc="left", fontweight="bold", pad=40)

    rosto = (np.array(v["serie"]["rostos"]) > 0).astype(float)
    movimento = np.convolve(np.array(v["serie"]["dif_camera"], dtype=float), np.ones(15) / 15, mode="same")
    teto = max(float(movimento.max()), 1.0) * 1.35  # folga em cima para a legenda não cobrir a curva
    baixo.fill_between(minutos, 0, rosto * teto, color=config.PALETA["video"], alpha=0.15, step="mid", linewidth=0,
                       label="rosto visível (YuNet)")
    baixo.plot(minutos, movimento, color=config.PALETA["video"], linewidth=1.0, label="movimento na câmera (média de 15 s)")
    baixo.set_ylim(0, teto)
    baixo.set_ylabel("movimento")
    baixo.set_xlabel("minuto da palestra")
    baixo.legend(loc="upper right", fontsize=8, ncols=2, framealpha=0.9)
    baixo.set_title("Câmera: palestrante visível e movimento", loc="left", fontweight="bold")

    figura.suptitle(f"Linha do tempo · {info['titulo'][:80]}", fontsize=12, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "03_linha_tempo_palestra.png")


# ---------------------------------------------------------------------------
# 04 Perfil visual de cada palestra
# ---------------------------------------------------------------------------


def perfil_palestras(analise: dict[str, Any]) -> Path:
    # Ordem: da palestra mais "slide" para a mais "ferramenta" (código + interface).
    ps = sorted(analise["por_palestra"].values(),
                key=lambda p: p["tempo_por_tipo"]["código"] + p["tempo_por_tipo"]["interface"])
    nomes = [_curto(p["titulo"]) for p in ps]
    figura, eixos = plt.subplots(1, 3, figsize=(16, 6), sharey=True, gridspec_kw={"width_ratios": [1.7, 1, 1]})

    _empilhado(eixos[0], nomes, [p["tempo_por_tipo"] for p in ps], TIPOS, config.PALETA_TELA)
    eixos[0].set_title("Tempo de vídeo por tipo de tela (%)", fontweight="bold", pad=24)
    eixos[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(TIPOS), frameon=False, fontsize=8)

    for eixo, (titulo, valores, cor, rotulos) in zip(eixos[1:], (
        ("Ritmo: telas novas por minuto", [p["visual"]["telas_por_min"] for p in ps], config.PALETA["slide"],
         [_num(p["visual"]["telas_por_min"], 1) for p in ps]),
        ("Palestrante visível (% do tempo)", [p["visual"]["rosto_pct"] for p in ps], config.PALETA["video"],
         [f"{p['visual']['rosto_pct']:.0f}%" for p in ps]),
    )):
        barras = eixo.barh(nomes, valores, color=cor)
        eixo.bar_label(barras, labels=rotulos, padding=3, fontsize=8)
        eixo.set_title(titulo, fontweight="bold", pad=24)
        eixo.margins(x=0.2)
        eixo.grid(axis="y", visible=False)
    eixos[2].set_xlim(0, 118)  # espaço para o rótulo "100%", sem marcar 120% no eixo
    eixos[2].set_xticks([0, 25, 50, 75, 100])
    figura.suptitle("Perfil visual de cada palestra: o que a tela mostra, em que ritmo, e quanto o palestrante aparece",
                    fontsize=13, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "04_perfil_palestras.png")


# ---------------------------------------------------------------------------
# 05 Ferramentas: tela x fala x legenda
# ---------------------------------------------------------------------------


def _corte(texto: str, limite: int = 100) -> str:
    return texto if len(texto) <= limite else texto[: limite - 1] + "…"


def ferramentas_fontes(analise: dict[str, Any], eixo=None) -> Path | None:
    fontes = analise["ferramentas"]["por_fonte"]
    proprio = eixo is None
    # No painel o espaço é menor: as 5 mais citadas e a legenda curta, que cabe
    # no canto onde as barras das menos citadas não chegam.
    nomes = list(fontes)[: 8 if proprio else 5][::-1]
    if proprio:
        figura, eixo = plt.subplots(figsize=(11, 6))
    posicoes = np.arange(len(nomes))
    altura = 0.27
    series = (
        ("tela", "telas em que aparece escrita (OCR)", "tela (OCR)", config.PALETA_TELA["texto"]),
        ("fala", "vezes dita (nosso Whisper, com lista de vocabulário)", "fala (Whisper)", config.PALETA["video"]),
        ("legenda", "vezes na legenda oficial da FOSDEM", "legenda oficial", config.PALETA["neutro"]),
    )
    for deslocamento, (chave, rotulo, curto, cor) in zip((altura, 0, -altura), series):
        barras = eixo.barh(posicoes + deslocamento, [fontes[n][chave] for n in nomes], altura, color=cor,
                           label=rotulo if proprio else curto)
        eixo.bar_label(barras, padding=2, fontsize=7)
    eixo.set_yticks(posicoes, nomes)
    eixo.set_xlabel("contagem")
    eixo.margins(x=0.12)
    eixo.grid(axis="y", visible=False)
    eixo.legend(loc="lower right", frameon=False, fontsize=8)
    if not proprio:
        eixo.set_title("Ferramentas: tela x fala x legenda", fontweight="bold")
        return None
    eixo.set_title("Ferramentas de IA: tela x fala x legenda oficial", fontweight="bold")
    leg = analise["ferramentas"]["legenda"]
    nota = (f"Nomes que só a nossa transcrição escreveu: {sum(leg['ganhos'].values())}  ·  "
            f"que só a legenda oficial escreveu: {sum(leg['perdas'].values())}")
    if leg["exemplos"]:
        exemplo = leg["exemplos"][0]
        nota += f"\nex.: nosso   \"{_corte(exemplo['nosso'])}\"\n     oficial \"{_corte(exemplo['oficial'])}\""
    figura.text(0.01, -0.02, nota, fontsize=9, color="#333", va="top", family="monospace")
    figura.tight_layout()
    return _salvar(figura, "05_ferramentas_tela_fala_legenda.png")


# ---------------------------------------------------------------------------
# 06 A fala acompanha a tela?
# ---------------------------------------------------------------------------


def fala_por_tela(analise: dict[str, Any]) -> Path:
    tipos = [t for t in TIPOS if analise["tipos"]["wpm"].get(t)]
    figura, (esquerda, direita) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})

    valores = [analise["tipos"]["wpm"][t] for t in tipos]
    barras = esquerda.bar(tipos, valores, color=[config.PALETA_TELA[t] for t in tipos])
    esquerda.bar_label(barras, labels=[f"{v:.0f}" for v in valores], padding=3, fontsize=9)
    esquerda.set_ylabel("palavras por minuto")
    esquerda.set_title("Velocidade da fala com cada tipo de tela", fontweight="bold", pad=24)  # alinhado ao da direita
    esquerda.margins(y=0.15)
    esquerda.grid(axis="x", visible=False)

    sentimento = analise["tipos"]["sentimento"]
    nomes = [f"{t}\n(n={_num(sentimento[t]['n'])})" for t in tipos][::-1]
    _empilhado(direita, nomes, [sentimento[t]["distribuicao"] for t in tipos][::-1], ORDEM_SENTIMENTO,
               config.PALETA, medias=[sentimento[t]["media"] for t in tipos][::-1])
    direita.set_title("Sentimento das frases ditas com cada tipo de tela", fontweight="bold", pad=24)
    direita.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=3, frameon=False, fontsize=8)
    direita.set_xlabel("percentual")

    figura.suptitle("A fala acompanha a tela?", fontsize=13, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "06_fala_por_tipo_tela.png")


# ---------------------------------------------------------------------------
# 07 Escrito x dito
# ---------------------------------------------------------------------------


def nuvens(analise: dict[str, Any]) -> Path | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("    [pulado] wordcloud não instalado")
        return None
    figura, eixos = plt.subplots(1, 2, figsize=(16, 5.8))
    for eixo, chave, titulo, mapa in zip(
        eixos, ("tela", "fala"),
        ("O que está ESCRITO na tela (OCR das telas)", "O que é DITO na palestra (Whisper)"),
        ("Dark2", "viridis"),
    ):
        eixo.axis("off")
        frequencias = dict(analise["vocabulario"][chave])
        if not frequencias:
            continue
        nuvem = WordCloud(width=1200, height=820, background_color="white", colormap=mapa, max_words=90,
                          relative_scaling=0.45, random_state=42).generate_from_frequencies(frequencias)
        eixo.imshow(nuvem, interpolation="bilinear")
        eixo.set_title(titulo, fontweight="bold", fontsize=12)
    figura.tight_layout()
    return _salvar(figura, "07_escrito_x_dito.png")


# ---------------------------------------------------------------------------
# 08 Painel da etapa 3
# ---------------------------------------------------------------------------


def _cartao(eixo, titulo: str, linhas: list[tuple[str, str]], cor: str) -> None:
    eixo.axis("off")
    eixo.set_title(titulo, fontweight="bold", loc="left")
    passo = 0.8 / max(len(linhas) - 1, 1)
    for indice, (rotulo, valor) in enumerate(linhas):
        altura = 0.9 - indice * passo
        eixo.text(0.02, altura, rotulo, fontsize=10, color="#555", transform=eixo.transAxes)
        eixo.text(0.98, altura, valor, fontsize=14, fontweight="bold", ha="right", color=cor, transform=eixo.transAxes)


def painel(analise: dict[str, Any]) -> Path:
    """Painel único com a visão geral da etapa: o dashboard pedido no enunciado."""
    figura = plt.figure(figsize=(16, 9))
    grade = figura.add_gridspec(2, 3, hspace=0.55, wspace=0.45)
    est = analise["estatisticas"]
    _cartao(figura.add_subplot(grade[0, 0]), "O vídeo em números", [
        ("palestras analisadas", f"{est['palestras']}"),
        ("horas de vídeo", _num(est["horas_video"], 1)),
        ("telas distintas", _num(est["telas"])),
        ("palavras lidas na tela", _num(est["palavras_tela"])),
        ("palavras faladas", _num(est["palavras_fala"])),
        ("palestrante visível", f"{est['rosto_pct_medio']:.0f}% do tempo"),
    ], config.PALETA["video"])

    eixo = figura.add_subplot(grade[0, 1])
    tempo = analise["tipos"]["tempo_pct"]
    barras = eixo.barh(TIPOS[::-1], [tempo[t] for t in TIPOS[::-1]], color=[config.PALETA_TELA[t] for t in TIPOS[::-1]])
    eixo.bar_label(barras, fmt="%.0f%%", padding=3, fontsize=8)
    eixo.set_title("Tempo de vídeo por tipo de tela", fontweight="bold")
    eixo.margins(x=0.18)
    eixo.grid(axis="y", visible=False)

    ferramentas_fontes(analise, figura.add_subplot(grade[0, 2]))

    eixo = figura.add_subplot(grade[1, 0])
    termos = analise["vocabulario"]["tela"][:10][::-1]
    barras = eixo.barh([t for t, _ in termos], [n for _, n in termos], color=config.PALETA_TELA["texto"])
    eixo.bar_label(barras, padding=3, fontsize=8)
    eixo.set_title("Top 10 termos escritos na tela", fontweight="bold")
    eixo.margins(x=0.15)
    eixo.grid(axis="y", visible=False)

    eixo = figura.add_subplot(grade[1, 1])
    wpm = analise["tipos"]["wpm"]
    tipos = [t for t in TIPOS if wpm.get(t)][::-1]  # deitado, na mesma ordem do tempo por tela
    barras = eixo.barh(tipos, [wpm[t] for t in tipos], color=[config.PALETA_TELA[t] for t in tipos])
    eixo.bar_label(barras, fmt="%.0f", padding=3, fontsize=8)
    eixo.set_title("Velocidade da fala (WPM) por tela", fontweight="bold")
    eixo.margins(x=0.15)
    eixo.grid(axis="y", visible=False)

    eixo = figura.add_subplot(grade[1, 2])
    ps = sorted(analise["por_palestra"].values(), key=lambda p: p["visual"]["telas_por_min"])
    eixo.barh([_curto(p["titulo"], 22) for p in ps], [p["visual"]["telas_por_min"] for p in ps], color=config.PALETA["slide"])
    eixo.set_title("Telas novas por minuto", fontweight="bold")
    eixo.tick_params(axis="y", labelsize=7)
    eixo.grid(axis="y", visible=False)

    figura.suptitle("DevIA Radar · Etapa 3  |  Como desenvolvedores APRESENTAM IA no desenvolvimento de software",
                    fontsize=15, fontweight="bold")
    return _salvar(figura, "08_painel_dashboard.png")


# ---------------------------------------------------------------------------
# 09 Painel final das três etapas
# ---------------------------------------------------------------------------

ETAPAS = (
    ("texto", "Texto · etapa 1", "issues do GitHub", "texto"),
    ("fala", "Fala · etapa 2", "podcasts", "fala"),
    ("video", "Vídeo · etapa 3", "palestras da FOSDEM", "video"),
)


def _participacao(contagem: dict[str, int]) -> dict[str, float]:
    total = sum(contagem.values()) or 1
    return {nome: qtd / total * 100 for nome, qtd in contagem.items()}


def painel_tres_etapas(analise: dict[str, Any]) -> Path | None:
    """
    O dashboard final prometido no slide 9 da etapa 1: as três modalidades
    lado a lado, com o mesmo dicionário de ferramentas e o mesmo VADER.
    """
    etapas = [(chave, titulo, fonte, cor) for chave, titulo, fonte, cor in ETAPAS if analise["tres_etapas"].get(chave)]
    if len(etapas) < 2:
        print("    [pulado] 09 precisa de pelo menos duas etapas com resultado")
        return None
    figura = plt.figure(figsize=(16, 9.6))
    grade = figura.add_gridspec(2, len(etapas), height_ratios=[0.42, 1.25], hspace=0.38, wspace=0.32)

    for coluna, (chave, titulo, fonte, cor) in enumerate(etapas):
        dados = analise["tres_etapas"][chave]
        participacao = _participacao(dados["ferramentas"])
        topo = max(participacao, key=participacao.get) if participacao else "-"
        _cartao(figura.add_subplot(grade[0, coluna]), f"{titulo}: {fonte}", [
            ("amostra", dados["volume"]),
            ("frases com sentimento", _num(dados["frases_avaliadas"])),
            ("compound médio", _num(dados["media"], 3, sinal=True) if dados["media"] is not None else "-"),
            ("mais citada", f"{topo} ({participacao.get(topo, 0):.0f}%)"),
        ], config.PALETA[cor])

    meio = len(etapas) - 1
    eixo = figura.add_subplot(grade[1, :meio])
    participacoes = {chave: _participacao(analise["tres_etapas"][chave]["ferramentas"]) for chave, *_ in etapas}
    nomes = sorted({n for p in participacoes.values() for n in p}, key=lambda n: -sum(p.get(n, 0) for p in participacoes.values()))[:7][::-1]
    posicoes = np.arange(len(nomes))
    altura = 0.8 / len(etapas)
    for indice, (chave, titulo, _, cor) in enumerate(etapas):
        deslocamento = (len(etapas) - 1 - indice - (len(etapas) - 1) / 2) * altura
        barras = eixo.barh(posicoes + deslocamento, [participacoes[chave].get(n, 0) for n in nomes], altura,
                           color=config.PALETA[cor], label=titulo)
        eixo.bar_label(barras, fmt="%.0f%%", padding=2, fontsize=7)
    eixo.set_yticks(posicoes, nomes)
    eixo.set_xlabel("participação no total de menções a ferramentas (%)")
    eixo.set_title("Quais ferramentas de IA cada modalidade cita", fontweight="bold")
    eixo.legend(loc="lower right", frameon=False)
    eixo.margins(x=0.12)
    eixo.grid(axis="y", visible=False)

    eixo = figura.add_subplot(grade[1, meio])
    rotulos = [titulo for _, titulo, _, _ in etapas][::-1]
    _empilhado(eixo, rotulos, [analise["tres_etapas"][c]["distribuicao"] for c, *_ in etapas][::-1],
               ORDEM_SENTIMENTO, config.PALETA)
    eixo.set_title("Sentimento (VADER)", fontweight="bold", pad=24)
    eixo.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False, fontsize=8)
    eixo.set_xlabel("percentual")

    figura.suptitle("DevIA Radar · as três etapas juntas: como desenvolvedores ESCREVEM, FALAM e APRESENTAM IA",
                    fontsize=15, fontweight="bold")
    return _salvar(figura, "09_painel_tres_etapas.png")


def gerar_todos(analise: dict[str, Any], visual: dict[str, Any], lidas: dict[str, Any], palestras: list[dict[str, Any]]) -> None:
    import quadros

    print("  gerando gráficos em saida/ ...")
    mosaico_telas(palestras, visual, lidas)
    demo = quadros.preparar_demo(palestras)
    if demo:
        figura_quadro(demo)
    else:
        print("    [pulado] 02 precisa do vídeo no disco (opção 1)")
    linha_do_tempo(analise, visual, lidas, demo["palestra"]["id"] if demo else config.DEMO_PALESTRA)
    perfil_palestras(analise)
    ferramentas_fontes(analise)
    fala_por_tela(analise)
    nuvens(analise)
    painel(analise)
    painel_tres_etapas(analise)
    print(f"  pronto: {config.DIR_SAIDA}")
