"""
Etapa 1 - VISUALIZAÇÃO E DASHBOARD.

Todos os gráficos saem como PNG na pasta `saída/`, prontos para colar
nos slides. Usamos o backend "Agg" do matplotlib, que desenha direto em
arquivo sem abrir janela. Isso evita o programa travar esperando a janela
fechar durante a apresentação.

Gráficos gerados:
  01  nuvem de palavras dos lemas
  02  top lemas do corpus
  03  ferramentas de IA mais citadas
  04  distribuição de sentimento
  05  sentimento por ferramenta (barras 100%)
  06  linha do tempo (volume e sentimento médio)
  07  entidades nomeadas por tipo
  08  painel consolidado (o dashboard pedido no enunciado)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")  # precisa vir antes de importar o pyplot

import matplotlib.pyplot as plt

import config

plt.rcParams.update(
    {
        "figure.dpi": config.DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def _salvar(figura, nome: str) -> str:
    caminho = config.DIR_SAIDA / nome
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)
    print(f"    {nome}")
    return str(caminho)


def _barra_horizontal(eixo, itens, cor, titulo, rotulo_x="ocorrências"):
    """Barra horizontal com o maior valor no topo, que é como se lê ranking."""
    rotulos = [str(item[0]) for item in itens][::-1]
    valores = [item[1] for item in itens][::-1]
    barras = eixo.barh(rotulos, valores, color=cor)
    eixo.set_title(titulo, fontweight="bold")
    eixo.set_xlabel(rotulo_x)
    eixo.bar_label(barras, padding=3, fontsize=8)
    eixo.margins(x=0.12)
    eixo.grid(axis="y", visible=False)


def nuvem_de_palavras(analise_pln: dict[str, Any]) -> str | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("    [pulado] wordcloud não instalado")
        return None

    frequencias = dict(analise_pln["top_lemas"])
    nuvem = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=120,
        relative_scaling=0.45,
    ).generate_from_frequencies(frequencias)

    figura, eixo = plt.subplots(figsize=(11, 6))
    eixo.imshow(nuvem, interpolation="bilinear")
    eixo.axis("off")
    eixo.set_title(
        "Vocabulário dominante nas issues sobre IA no desenvolvimento",
        fontweight="bold",
        pad=12,
    )
    return _salvar(figura, "01_nuvem_palavras.png")


def top_lemas(analise_pln: dict[str, Any]) -> str:
    figura, eixo = plt.subplots(figsize=(9, 7))
    _barra_horizontal(
        eixo,
        analise_pln["top_lemas"][:20],
        config.PALETA["destaque"],
        "Top 20 lemas (palavras na forma de dicionário)",
    )
    return _salvar(figura, "02_top_lemas.png")


def _contar_ferramentas(corpus: list[dict[str, Any]], minimo: int = 5):
    """
    Conta menções por ferramenta, cortando as raras.

    Ferramenta com duas menções vira uma barra de um pixel no gráfico e não
    sustenta nenhuma conclusão, entao fica de fora.
    """
    contagem = Counter(
        ferramenta for doc in corpus for ferramenta in doc.get("ferramentas_ia", [])
    )
    return [(nome, qtd) for nome, qtd in contagem.most_common() if qtd >= minimo]


def ferramentas_citadas(corpus: list[dict[str, Any]]) -> str:
    figura, eixo = plt.subplots(figsize=(9, 5.5))
    _barra_horizontal(
        eixo,
        _contar_ferramentas(corpus)[:12],
        config.PALETA["apoio"],
        "Ferramentas de IA mais citadas (extraídas por regex)",
        "issues que citam",
    )
    return _salvar(figura, "03_ferramentas.png")


def distribuicao_sentimento(analise_sent: dict[str, Any]) -> str:
    dados = analise_sent["distribuicao"]
    ordem = ["positivo", "neutro", "negativo"]
    valores = [dados.get(rotulo, 0) for rotulo in ordem]
    total = sum(valores) or 1
    cores = [config.PALETA[rotulo] for rotulo in ordem]

    figura, (esq, dir_) = plt.subplots(1, 2, figsize=(11, 4.6))

    barras = esq.bar(ordem, valores, color=cores)
    esq.bar_label(barras, padding=3)
    esq.set_title("Issues por sentimento", fontweight="bold")
    esq.set_ylabel("quantidade de issues")
    esq.grid(axis="x", visible=False)

    dir_.pie(
        valores,
        labels=[f"{r}\n{v / total * 100:.1f}%" for r, v in zip(ordem, valores)],
        colors=cores,
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    dir_.set_title(
        f"Proporção (compound médio {analise_sent['media_geral']:+.3f})",
        fontweight="bold",
    )
    return _salvar(figura, "04_sentimento.png")


def sentimento_por_ferramenta(analise_sent: dict[str, Any]) -> str | None:
    dados = analise_sent["por_ferramenta"]
    if not dados:
        print("    [pulado] nenhuma ferramenta com menções suficientes")
        return None

    nomes = list(dados)[:8][::-1]
    figura, eixo = plt.subplots(figsize=(10, 5.5))

    base = [0.0] * len(nomes)
    for rotulo in ("positivo", "neutro", "negativo"):
        chave = rotulo + "s"
        fatia = [
            dados[n][chave] / max(dados[n]["mencoes"], 1) * 100 for n in nomes
        ]
        eixo.barh(
            nomes,
            fatia,
            left=base,
            color=config.PALETA[rotulo],
            label=rotulo,
            edgecolor="white",
        )
        base = [b + f for b, f in zip(base, fatia)]

    for indice, nome in enumerate(nomes):
        eixo.text(
            101,
            indice,
            f"n={dados[nome]['mencoes']}  média {dados[nome]['media']:+.2f}",
            va="center",
            fontsize=8,
            color="#444",
        )

    eixo.set_xlim(0, 132)
    # A escala do eixo para em 100, porque de 100 a 132 e a faixa reservada
    # para o texto com n e media. Tick ali sugeriria dado que não existe.
    eixo.set_xticks([0, 20, 40, 60, 80, 100])
    eixo.set_xlabel("percentual das issues que citam a ferramenta")
    eixo.set_title("Como cada ferramenta é discutida", fontweight="bold", pad=28)
    # A legenda vai acima da área de plotagem. Dentro do eixo ela cobria a
    # última barra, que fica justamente no canto inferior direito.
    eixo.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        frameon=False,
    )
    eixo.grid(axis="y", visible=False)
    return _salvar(figura, "05_sentimento_ferramenta.png")


def linha_do_tempo(analise_sent: dict[str, Any]) -> str | None:
    linha = analise_sent["linha_tempo"]
    if len(linha) < 3:
        print("    [pulado] histórico curto demais")
        return None

    meses = list(linha)[-36:]  # últimos 3 anos
    volume = [linha[m]["volume"] for m in meses]
    media = [linha[m]["media"] for m in meses]

    figura, eixo = plt.subplots(figsize=(12, 5))
    eixo.bar(meses, volume, color=config.PALETA["destaque"], alpha=0.45, label="issues")
    eixo.set_ylabel("issues no mês", color=config.PALETA["destaque"])
    eixo.set_title(
        "Evolução do volume e do sentimento ao longo do tempo", fontweight="bold"
    )

    gemeo = eixo.twinx()
    gemeo.plot(
        meses, media, color=config.PALETA["negativo"], marker="o",
        linewidth=2, markersize=4, label="compound médio",
    )
    gemeo.axhline(0, color="#999", linestyle="--", linewidth=0.8)
    gemeo.set_ylabel("compound médio", color=config.PALETA["negativo"])
    gemeo.grid(False)

    passo = max(len(meses) // 14, 1)
    eixo.set_xticks(range(0, len(meses), passo))
    eixo.set_xticklabels(meses[::passo], rotation=45, ha="right", fontsize=8)
    return _salvar(figura, "06_linha_tempo.png")


def entidades(analise_pln: dict[str, Any]) -> str:
    tipos = list(analise_pln["top_entidades"])[:4]
    figura, eixos = plt.subplots(2, 2, figsize=(12, 8))

    for eixo, tipo in zip(eixos.flat, tipos):
        _barra_horizontal(
            eixo,
            analise_pln["top_entidades"][tipo][:8],
            config.PALETA["destaque"],
            f"Entidades do tipo {tipo}",
        )
    for eixo in eixos.flat[len(tipos):]:
        eixo.axis("off")

    figura.suptitle(
        "Entidades nomeadas reconhecidas pelo spaCy (NER)",
        fontweight="bold",
        fontsize=13,
    )
    figura.tight_layout()
    return _salvar(figura, "07_entidades.png")


def painel(
    corpus: list[dict[str, Any]],
    analise_pln: dict[str, Any],
    analise_sent: dict[str, Any],
) -> str:
    """Painel único com a visao geral, o dashboard pedido no enunciado."""
    figura = plt.figure(figsize=(16, 9))
    grade = figura.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    est = analise_pln["estatisticas"]
    cartao = figura.add_subplot(grade[0, 0])
    cartao.axis("off")
    linhas = [
        ("issues analisadas", f"{est['documentos']:,}".replace(",", ".")),
        ("repositórios", f"{len({d['repositorio'] for d in corpus}):,}".replace(",", ".")),
        ("tokens processados", f"{est['tokens_totais']:,}".replace(",", ".")),
        ("vocabulário único", f"{est['vocabulario_unico']:,}".replace(",", ".")),
        ("riqueza lexical (TTR)", f"{est['riqueza_lexical_ttr']:.3f}"),
        ("compound médio", f"{analise_sent['media_geral']:+.3f}"),
    ]
    cartao.set_title("Corpus em números", fontweight="bold", loc="left")
    for indice, (rotulo, valor) in enumerate(linhas):
        altura = 0.88 - indice * 0.16
        cartao.text(0.02, altura, rotulo, fontsize=10, color="#555")
        cartao.text(
            0.98, altura, valor, fontsize=14, fontweight="bold",
            ha="right", color=config.PALETA["destaque"],
        )

    _barra_horizontal(
        figura.add_subplot(grade[0, 1]),
        analise_pln["top_lemas"][:10],
        config.PALETA["destaque"],
        "Top 10 termos",
    )

    _barra_horizontal(
        figura.add_subplot(grade[0, 2]),
        _contar_ferramentas(corpus)[:8],
        config.PALETA["apoio"],
        "Ferramentas citadas",
        "issues",
    )

    eixo_sent = figura.add_subplot(grade[1, 0])
    ordem = ["positivo", "neutro", "negativo"]
    valores = [analise_sent["distribuicao"].get(r, 0) for r in ordem]
    barras = eixo_sent.bar(ordem, valores, color=[config.PALETA[r] for r in ordem])
    eixo_sent.bar_label(barras, padding=3, fontsize=8)
    eixo_sent.set_title("Sentimento", fontweight="bold")
    eixo_sent.grid(axis="x", visible=False)

    _barra_horizontal(
        figura.add_subplot(grade[1, 1]),
        analise_pln["top_bigramas"][:10],
        "#5B8C5A",
        "Top 10 bigramas",
    )

    eixo_tempo = figura.add_subplot(grade[1, 2])
    linha = analise_sent["linha_tempo"]
    meses = list(linha)[-24:]
    if meses:
        eixo_tempo.plot(
            meses,
            [linha[m]["volume"] for m in meses],
            color=config.PALETA["destaque"],
            linewidth=2,
        )
        eixo_tempo.fill_between(
            meses,
            [linha[m]["volume"] for m in meses],
            color=config.PALETA["destaque"],
            alpha=0.2,
        )
        passo = max(len(meses) // 6, 1)
        eixo_tempo.set_xticks(range(0, len(meses), passo))
        eixo_tempo.set_xticklabels(meses[::passo], rotation=45, ha="right", fontsize=8)
    eixo_tempo.set_title("Volume de issues por mês", fontweight="bold")

    figura.suptitle(
        "DevIA Radar  |  Como desenvolvedores discutem IA no desenvolvimento de software",
        fontsize=15,
        fontweight="bold",
    )
    return _salvar(figura, "08_painel_dashboard.png")


def gerar_todos(
    corpus: list[dict[str, Any]],
    analise_pln: dict[str, Any],
    analise_sent: dict[str, Any],
) -> None:
    print("  gerando gráficos em saida/ ...")
    nuvem_de_palavras(analise_pln)
    top_lemas(analise_pln)
    ferramentas_citadas(corpus)
    distribuicao_sentimento(analise_sent)
    sentimento_por_ferramenta(analise_sent)
    linha_do_tempo(analise_sent)
    entidades(analise_pln)
    painel(corpus, analise_pln, analise_sent)
    print(f"  pronto: {config.DIR_SAIDA}")
