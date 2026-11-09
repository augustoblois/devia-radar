"""
Etapa 2 - VISUALIZAÇÃO E DASHBOARD.

Mesmo padrão da etapa 1: todo gráfico sai como PNG na pasta `saida/`,
pronto para o slide, com o backend "Agg" do matplotlib (desenha direto no
arquivo, sem abrir janela que travaria a apresentação).

Gráficos gerados:
  01  nuvem de palavras da fala
  02  do som aos números: onda, energia, espectrograma Mel e MFCC de um trecho
  03  perfil acústico de cada episódio (velocidade, pausas, altura da voz)
  04  ferramentas citadas: fala (podcasts) x texto (issues da etapa 1)
  05  sentimento: fala x texto
  06  linha do tempo de um episódio (energia, sentimento e ferramentas)
  07  a voz muda com o sentimento?
  08  painel consolidado (dashboard)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # precisa vir antes de importar o pyplot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

import config

plt.rcParams.update(
    {
        "figure.dpi": config.DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,  # grade atrás das barras, não riscando por cima
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)

ORDEM_SENTIMENTO = ["positivo", "neutro", "negativo"]


def _classe(compound: float) -> str:
    """Mesmas faixas do VADER usadas em todo o projeto."""
    if compound >= config.LIMITE_POSITIVO:
        return "positivo"
    if compound <= config.LIMITE_NEGATIVO:
        return "negativo"
    return "neutro"


def _num(valor: float, casas: int = 0, sinal: bool = False) -> str:
    """Número escrito em português: vírgula decimal e ponto no milhar."""
    texto = f"{valor:{'+' if sinal else ''},.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# Marcas dos eixos numéricos com vírgula decimal (0,5 em vez de 0.5).
VIRGULA = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))


def _salvar(figura, nome: str) -> Path:
    # Só troca o formatador padrão: eixo de categoria e eixo de tempo do
    # librosa têm formatador próprio e ficam como estão.
    for eixo in figura.axes:
        for eixo_xy in (eixo.xaxis, eixo.yaxis):
            if type(eixo_xy.get_major_formatter()) is ScalarFormatter:
                eixo_xy.set_major_formatter(VIRGULA)
    caminho = config.DIR_SAIDA / nome
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)
    print(f"    {nome}")
    return caminho


def _curto(nome: str, limite: int = 26) -> str:
    """Nome de podcast cabe no eixo do gráfico só se for curto."""
    nome = nome.split(":")[0].split(" - ")[0].split(" – ")[0].strip()
    return nome if len(nome) <= limite else nome[: limite - 1] + "…"


def _barra_horizontal(eixo, itens, cor, titulo, rotulo_x="ocorrências", formato="{:.0f}"):
    rotulos = [str(item[0]) for item in itens][::-1]
    valores = [item[1] for item in itens][::-1]
    barras = eixo.barh(rotulos, valores, color=cor)
    eixo.set_title(titulo, fontweight="bold")
    eixo.set_xlabel(rotulo_x)
    eixo.bar_label(barras, labels=[formato.format(v) for v in valores], padding=3, fontsize=8)
    eixo.margins(x=0.15)
    eixo.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# 01 Nuvem de palavras
# ---------------------------------------------------------------------------


def nuvem_de_palavras(analise: dict[str, Any]) -> Path | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("    [pulado] wordcloud não instalado")
        return None
    nuvem = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=120,
        relative_scaling=0.45,
        random_state=42,
    ).generate_from_frequencies(dict(analise["vocabulario"]["top_lemas"]))
    figura, eixo = plt.subplots(figsize=(11, 6))
    eixo.imshow(nuvem, interpolation="bilinear")
    eixo.axis("off")
    eixo.set_title("Vocabulário dominante na fala dos podcasts sobre IA no desenvolvimento", fontweight="bold", pad=12)
    return _salvar(figura, "01_nuvem_fala.png")


# ---------------------------------------------------------------------------
# 02 Do som aos números (um trecho)
# ---------------------------------------------------------------------------


def figura_trecho(sinal, q: dict[str, np.ndarray], pausas, episodio: dict[str, Any], inicio: float) -> Path:
    """
    O mesmo trecho desenhado quatro vezes, uma camada de análise por linha.

    Todas as linhas dividem o mesmo eixo de tempo, então dá para apontar
    uma pausa na onda e ver o buraco correspondente no espectrograma.
    """
    import librosa.display

    sr = config.TAXA_AMOSTRAGEM
    tempo = np.arange(len(sinal)) / sr
    tempo_quadros = np.arange(q["energia_db"].size) * config.SALTO / sr

    figura, eixos = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True, gridspec_kw={"height_ratios": [1, 0.8, 1.5, 1.1]})

    eixos[0].plot(tempo, sinal, color=config.PALETA["destaque"], linewidth=0.4)
    for inicio_p, fim_p in pausas:
        eixos[0].axvspan(inicio_p, fim_p, color=config.PALETA["apoio"], alpha=0.25)
    eixos[0].set_ylabel("amplitude")
    eixos[0].set_title(
        f"1. Forma de onda: {len(sinal):,} amostras ({sr:,} por segundo). Faixas laranja = pausas longas".replace(",", "."),
        loc="left", fontweight="bold", fontsize=10,
    )

    eixos[1].plot(tempo_quadros, q["energia_db"], color=config.PALETA["destaque"], linewidth=0.9)
    limiar = q["energia_db"].max() - config.TOP_DB_SILENCIO
    eixos[1].axhline(limiar, color=config.PALETA["negativo"], linestyle="--", linewidth=1)
    eixos[1].text(tempo_quadros[0], limiar, f" limiar de silêncio ({config.TOP_DB_SILENCIO} dB abaixo do pico) ",
                  ha="left", va="center", fontsize=8, color=config.PALETA["negativo"],
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    eixos[1].set_ylabel("energia (dB)")
    eixos[1].set_title("2. Energia RMS por quadro de 25 ms", loc="left", fontweight="bold", fontsize=10)

    librosa.display.specshow(
        q["mel_db"], sr=sr, hop_length=config.SALTO, x_axis="time", y_axis="mel",
        fmax=sr // 2, ax=eixos[2], cmap="magma",
    )
    f0 = np.where(q["sonoro"], q["f0"], np.nan)
    eixos[2].plot(tempo_quadros, f0, color="#7FDBFF", linewidth=1.6, label="F0 (altura da voz, YIN)")
    eixos[2].legend(loc="upper right", fontsize=8, framealpha=0.8)
    eixos[2].set_ylabel("frequência (Hz, escala Mel)")
    eixos[2].set_xlabel("")  # o specshow escreve "Time" sozinho; o eixo é o de baixo
    eixos[2].set_title(f"3. Espectrograma Mel ({config.N_MELS} bandas): cor clara = mais energia naquela frequência",
                       loc="left", fontweight="bold", fontsize=10)

    librosa.display.specshow(q["mfcc"], sr=sr, hop_length=config.SALTO, x_axis="time", ax=eixos[3], cmap="coolwarm")
    eixos[3].set_ylabel("coeficiente")
    eixos[3].set_yticks(range(0, config.N_MFCC, 2))
    eixos[3].set_xlabel("segundos do trecho")
    eixos[3].set_title(f"4. MFCC: {config.N_MFCC} números por quadro, a impressão digital do timbre",
                       loc="left", fontweight="bold", fontsize=10)

    for eixo in eixos:
        eixo.grid(False)
    minuto, segundo = divmod(int(inicio), 60)
    figura.suptitle(
        f"Do som aos números · {_curto(episodio['podcast'], 40)}, {config.TRECHO_DEMO} s a partir de {minuto}:{segundo:02d}",
        fontsize=13, fontweight="bold",
    )
    figura.tight_layout()
    return _salvar(figura, "02_do_som_aos_numeros.png")


# ---------------------------------------------------------------------------
# 03 Perfil acústico por episódio
# ---------------------------------------------------------------------------


def perfil_episodios(analise: dict[str, Any]) -> Path:
    eps = [e for e in analise["por_episodio"].values() if e.get("acustica")]
    eps.sort(key=lambda e: e["wpm"])
    nomes = [_curto(e["podcast"]) for e in eps]

    figura, eixos = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    paineis = [
        ("Velocidade da fala", "palavras por minuto", [e["wpm"] for e in eps], config.PALETA["destaque"], 0),
        ("Pausas longas (≥ 0,7 s)", "pausas por minuto", [e["acustica"]["pausas_por_min"] for e in eps], config.PALETA["apoio"], 1),
        ("Variação da altura da voz", "desvio da F0 (Hz)", [e["acustica"]["f0_desvio_hz"] for e in eps], "#5B8C5A", 0),
    ]
    for eixo, (titulo, rotulo, valores, cor, casas) in zip(eixos, paineis):
        barras = eixo.barh(nomes, valores, color=cor)
        eixo.bar_label(barras, labels=[_num(v, casas) for v in valores], padding=3, fontsize=8)
        eixo.set_title(titulo, fontweight="bold")
        eixo.set_xlabel(rotulo)
        eixo.margins(x=0.18)
        eixo.grid(axis="y", visible=False)
    figura.suptitle("Como cada podcast fala: velocidade, pausas e entonação", fontsize=13, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "03_perfil_episodios.png")


# ---------------------------------------------------------------------------
# 04 e 05 Fala x texto
# ---------------------------------------------------------------------------


def _participacao(contagem: dict[str, int]) -> dict[str, float]:
    total = sum(contagem.values()) or 1
    return {nome: qtd / total * 100 for nome, qtd in contagem.items()}


def ferramentas_fala_texto(analise: dict[str, Any], eixo=None) -> Path | None:
    fala = _participacao(analise["ferramentas"]["mencoes"])
    texto = _participacao(analise["etapa1"].get("ferramentas", {}))
    # Ferramenta com menos de 0,5% nas duas mídias só ocupa espaço.
    nomes = [n for n in set(fala) | set(texto) if max(fala.get(n, 0), texto.get(n, 0)) >= 0.5]
    nomes = sorted(nomes, key=lambda n: -(fala.get(n, 0) + texto.get(n, 0)))[:8][::-1]

    proprio = eixo is None
    if proprio:
        figura, eixo = plt.subplots(figsize=(10, 5.8))
    posicoes = np.arange(len(nomes))
    altura = 0.38
    # No painel o eixo é estreito e a legenda longa cobria o rótulo das barras.
    rotulo_texto = "texto (frases das issues, etapa 1)" if proprio else "texto (issues)"
    b1 = eixo.barh(posicoes + altura / 2, [fala.get(n, 0) for n in nomes], altura, color=config.PALETA["fala"], label="fala (podcasts)")
    b2 = eixo.barh(posicoes - altura / 2, [texto.get(n, 0) for n in nomes], altura, color=config.PALETA["texto"], label=rotulo_texto)
    eixo.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=8)
    eixo.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=8)
    eixo.set_yticks(posicoes, nomes)
    eixo.set_xlabel("% das menções a ferramentas")
    eixo.set_title("Ferramentas citadas: fala x texto", fontweight="bold")
    eixo.legend(loc="lower right", frameon=False)
    eixo.margins(x=0.15)
    eixo.grid(axis="y", visible=False)
    if proprio:
        return _salvar(figura, "04_ferramentas_fala_texto.png")
    return None


def sentimento_fala_texto(analise: dict[str, Any], eixo=None) -> Path | None:
    """
    Fala e texto frase a frase, e embaixo o texto como a etapa 1 mediu (a
    issue inteira), para mostrar o efeito da unidade de medida.
    """
    linhas = {
        "fala (frases)": (analise["sentimento"]["distribuicao"], analise["sentimento"]["media"]),
        "texto (frases)": (analise["etapa1"].get("distribuicao", {}), analise["etapa1"].get("media")),
    }
    por_issue = analise["etapa1"].get("por_issue")
    if por_issue:
        linhas["texto (issue inteira)"] = (por_issue["distribuicao"], por_issue["media"])

    proprio = eixo is None
    if proprio:
        figura, eixo = plt.subplots(figsize=(10, 4.2))
    nomes = list(linhas)[::-1]
    # A média vai no próprio rótulo do eixo: texto solto à direita da barra
    # invadia o gráfico vizinho no painel.
    rotulos = [nome if linhas[nome][1] is None else f"{nome}\nmédia {_num(linhas[nome][1], 2, sinal=True)}" for nome in nomes]
    base = np.zeros(len(nomes))
    for rotulo in ORDEM_SENTIMENTO:
        fatias = []
        for nome in nomes:
            total = sum(linhas[nome][0].values()) or 1
            fatias.append(linhas[nome][0].get(rotulo, 0) / total * 100)
        barras = eixo.barh(rotulos, fatias, left=base, color=config.PALETA[rotulo], label=rotulo, edgecolor="white")
        eixo.bar_label(barras, labels=[f"{f:.0f}%" if f >= 6 else "" for f in fatias], label_type="center", color="white", fontsize=9, fontweight="bold")
        base += np.array(fatias)
    eixo.set_xlim(0, 100)
    eixo.set_xticks([0, 25, 50, 75, 100])
    eixo.set_xlabel("percentual")
    eixo.set_title("Sentimento (VADER): fala x texto", fontweight="bold", pad=26)
    eixo.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=3, frameon=False)
    eixo.grid(axis="y", visible=False)
    if proprio:
        return _salvar(figura, "05_sentimento_fala_texto.png")
    return None


# ---------------------------------------------------------------------------
# 06 Linha do tempo de um episódio
# ---------------------------------------------------------------------------


def linha_do_tempo(analise: dict[str, Any], acustica: dict[str, Any], episodio_id: str) -> Path | None:
    curvas = acustica.get("curvas", {}).get(episodio_id)
    info = analise["por_episodio"].get(episodio_id)
    if not curvas or not info:
        print("    [pulado] episódio sem curvas acústicas")
        return None
    frases = [f for f in analise["frases"] if f["episodio"] == episodio_id]
    energia = np.array(curvas["energia_db"], dtype=float)
    janela = 15  # média móvel de 15 s: a curva crua oscila demais para ler
    suave = np.convolve(energia, np.ones(janela) / janela, mode="same")
    minutos = np.arange(energia.size) / 60

    figura, (topo, baixo) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    topo.plot(minutos, suave, color=config.PALETA["destaque"], linewidth=1.2)
    topo.set_ylabel("energia (dB)")
    topo.set_title("Energia da voz (média móvel de 15 s)", loc="left", fontweight="bold")

    for f in frases:
        if not f["avaliado"]:
            continue
        baixo.bar(
            f["inicio"] / 60, f["compound"], width=max((f["fim"] - f["inicio"]) / 60, 0.05),
            color=config.PALETA[_classe(f["compound"])], align="edge",
        )
    # Toda menção vira uma linha. O nome só é escrito quando está a mais de
    # 1,5 minuto do último nome escrito, senão os rótulos se encavalam.
    citadas = [f for f in frases if f["ferramentas"]]
    ultimo = -np.inf
    for f in citadas:
        minuto = f["inicio"] / 60
        baixo.axvline(minuto, color=config.PALETA["apoio"], alpha=0.35, linewidth=0.8)
        if minuto - ultimo >= 1.5:
            # Só o primeiro nome, mais "+N" quando a frase cita outras.
            extras = len(f["ferramentas"]) - 1
            nome = f["ferramentas"][0] + (f" +{extras}" if extras else "")
            baixo.annotate(nome, (minuto, 1.02), xycoords=("data", "axes fraction"),
                           rotation=60, fontsize=7, color=config.PALETA["apoio"], ha="left")
            ultimo = minuto
    baixo.axhline(0, color="#999", linewidth=0.8)
    baixo.set_ylim(-1, 1)
    baixo.set_ylabel("compound (VADER)")
    baixo.set_xlabel("minuto do episódio")
    baixo.set_title(f"Sentimento de cada frase transcrita · {len(citadas)} linhas laranja = frase que cita ferramenta de IA",
                    loc="left", fontweight="bold", pad=60)
    figura.suptitle(f"Linha do tempo · {info['podcast']}: {info['titulo'][:60]}", fontsize=12, fontweight="bold")
    figura.tight_layout()
    return _salvar(figura, "06_linha_tempo_episodio.png")


# ---------------------------------------------------------------------------
# 07 Voz x sentimento
# ---------------------------------------------------------------------------


def voz_sentimento(analise: dict[str, Any], eixos=None) -> Path | None:
    frases = [f for f in analise["frases"] if f["avaliado"]]
    proprio = eixos is None
    if proprio:
        figura, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for eixo, (chave, titulo) in zip(eixos, (("energia_db_z", "Energia da voz"), ("f0_hz_z", "Altura da voz (F0)"))):
        grupos = [
            [f[chave] for f in frases if f.get(chave) is not None and _classe(f["compound"]) == rotulo]
            for rotulo in ORDEM_SENTIMENTO
        ]
        caixas = eixo.boxplot(grupos, tick_labels=ORDEM_SENTIMENTO, patch_artist=True, showfliers=False, widths=0.55)
        for caixa, rotulo in zip(caixas["boxes"], ORDEM_SENTIMENTO):
            caixa.set_facecolor(config.PALETA[rotulo])
            caixa.set_alpha(0.75)
        for mediana in caixas["medians"]:
            mediana.set_color("black")
        rho = analise["voz_x_sentimento"][chave]["spearman"]
        eixo.axhline(0, color="#999", linestyle="--", linewidth=0.8)
        eixo.set_title(f"{titulo}\nSpearman rho = {_num(rho['rho'], 2, sinal=True)} (n = {_num(rho['n'])})", fontweight="bold")
        eixo.set_ylabel("escore z dentro do episódio")
        eixo.grid(axis="x", visible=False)
    if proprio:
        figura.suptitle("A voz muda quando a frase é positiva ou negativa?", fontsize=13, fontweight="bold")
        figura.tight_layout()
        return _salvar(figura, "07_voz_sentimento.png")
    return None


# ---------------------------------------------------------------------------
# 08 Painel
# ---------------------------------------------------------------------------


def painel(analise: dict[str, Any]) -> Path:
    """Painel único com a visão geral da etapa: o dashboard pedido no enunciado."""
    figura = plt.figure(figsize=(16, 9))
    grade = figura.add_gridspec(2, 3, hspace=0.5, wspace=0.52)

    est = analise["estatisticas"]
    cartao = figura.add_subplot(grade[0, 0])
    cartao.axis("off")
    linhas = [
        ("episódios analisados", f"{est['episodios']}"),
        ("podcasts diferentes", f"{est['podcasts']}"),
        ("horas de áudio", _num(est["horas_audio"], 1)),
        ("palavras transcritas", _num(est["palavras"])),
        ("velocidade média", f"{_num(est['wpm_medio'])} WPM"),
        ("compound médio da fala", _num(analise["sentimento"]["media"], 3, sinal=True)),
    ]
    cartao.set_title("A fala em números", fontweight="bold", loc="left")
    for indice, (rotulo, valor) in enumerate(linhas):
        altura = 0.88 - indice * 0.16
        cartao.text(0.02, altura, rotulo, fontsize=10, color="#555")
        cartao.text(0.98, altura, valor, fontsize=14, fontweight="bold", ha="right", color=config.PALETA["destaque"])

    _barra_horizontal(figura.add_subplot(grade[0, 1]), analise["vocabulario"]["top_lemas"][:10], config.PALETA["destaque"], "Top 10 termos falados")
    ferramentas_fala_texto(analise, figura.add_subplot(grade[0, 2]))
    sentimento_fala_texto(analise, figura.add_subplot(grade[1, 0]))
    _barra_horizontal(figura.add_subplot(grade[1, 1]), analise["vocabulario"]["top_bigramas"][:10], "#5B8C5A", "Top 10 bigramas falados")

    eixo_wpm = figura.add_subplot(grade[1, 2])
    eps = sorted(analise["por_episodio"].values(), key=lambda e: e["wpm"])
    eixo_wpm.barh([_curto(e["podcast"], 20) for e in eps], [e["wpm"] for e in eps], color=config.PALETA["apoio"])
    eixo_wpm.set_title("Velocidade da fala (WPM)", fontweight="bold")
    eixo_wpm.tick_params(axis="y", labelsize=7)
    eixo_wpm.grid(axis="y", visible=False)

    figura.suptitle("DevIA Radar · Etapa 2  |  Como desenvolvedores FALAM de IA no desenvolvimento de software", fontsize=15, fontweight="bold")
    return _salvar(figura, "08_painel_dashboard.png")


def gerar_todos(analise: dict[str, Any], acustica: dict[str, Any], episodios: list[dict[str, Any]]) -> None:
    import sinal

    print("  gerando gráficos em saida/ ...")
    nuvem_de_palavras(analise)
    trecho = sinal.preparar_trecho(episodios)
    if trecho:
        figura_trecho(*trecho)
    else:
        print("    [pulado] 02 precisa do áudio no disco (opção 1)")
    perfil_episodios(analise)
    ferramentas_fala_texto(analise)
    sentimento_fala_texto(analise)
    linha_do_tempo(analise, acustica, trecho[3]["id"] if trecho else config.TRECHO_DEMO_EPISODIO)
    voz_sentimento(analise)
    painel(analise)
    print(f"  pronto: {config.DIR_SAIDA}")
