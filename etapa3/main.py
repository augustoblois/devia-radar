"""
DevIA Radar - Etapa 3 (vídeo)
Como desenvolvedores APRESENTAM Inteligência Artificial no desenvolvimento
de software, a partir de palestras gravadas da FOSDEM.

Disciplina: Dados Não Estruturados (UFPB)
Etapa 3: análise de vídeo com módulos Python.

Uso:
    python main.py            abre o menu interativo
    python main.py --tudo     roda o pipeline inteiro sem perguntar nada

Pipeline:
    agenda XML da FOSDEM -> PyAV + OpenCV (quadros: troca de slide, bordas,
    rosto) -> RapidOCR (texto da tela) + Whisper (fala) + legenda oficial
    -> regex + spaCy + VADER (conteúdo) -> matplotlib (dashboard e painel
    final das três etapas)
"""

from __future__ import annotations

import re
import sys
from typing import Any

import cv2

import coleta
import config
import conteudo
import leitura
import quadros
import visualizacao

LARGURA = 72


class Estado:
    """Guarda o que já foi carregado na sessão, para não reler o disco toda hora."""

    def __init__(self) -> None:
        self.palestras: list[dict[str, Any]] = []
        self.visual: dict[str, Any] = {}
        self.lidas: dict[str, Any] = {}
        self.transcricoes: dict[str, Any] = {}
        self.analise: dict[str, Any] = {}

    def carregar_do_disco(self) -> None:
        if config.ARQ_PALESTRAS.exists():
            self.palestras = coleta.selecionar()
        self.visual = quadros.carregar_analise()
        self.lidas = leitura.carregar_leitura()
        self.transcricoes = leitura.carregar_transcricoes(self.palestras)
        self.analise = conteudo.carregar_analise()


def titulo(texto: str) -> None:
    print()
    print("=" * LARGURA)
    print(f" {texto}")
    print("=" * LARGURA)


def exigir(condicao: bool, mensagem: str) -> bool:
    if not condicao:
        print(f"\n  [!] {mensagem}")
        return False
    return True


# ---------------------------------------------------------------------------
# Ações do menu
# ---------------------------------------------------------------------------


def acao_coletar(estado: Estado, forcar: bool = False) -> None:
    titulo("1. COLETA: AGENDA XML DA FOSDEM, VÍDEO E LEGENDA")
    estado.palestras = coleta.coletar(forcar=forcar)
    coleta.resumo(estado.palestras)


def acao_resumo(estado: Estado) -> None:
    titulo("2. RESUMO DAS PALESTRAS COLETADAS")
    coleta.resumo(estado.palestras)


def acao_demo_quadro(estado: Estado) -> None:
    titulo("3. DEMONSTRAÇÃO: DO VÍDEO AOS NÚMEROS (OpenCV)")
    quadros.demonstrar(estado.palestras)


def acao_visual(estado: Estado) -> None:
    titulo("4. CARACTERÍSTICAS VISUAIS DE TODAS AS PALESTRAS")
    if not exigir(bool(estado.palestras), "Colete as palestras primeiro (opção 1)."):
        return
    estado.visual = quadros.analisar(estado.palestras)


def acao_demo_leitura(estado: Estado) -> None:
    titulo("5. DEMONSTRAÇÃO: TELA (OCR), FALA (WHISPER) E LEGENDA OFICIAL")
    leitura.demonstrar(estado.palestras, estado.visual)


def acao_ler(estado: Estado) -> None:
    titulo("6. LEITURA DAS TELAS (OCR) E TRANSCRIÇÃO DAS FALAS (WHISPER)")
    if not exigir(bool(estado.visual), "Extraia as características visuais primeiro (opção 4)."):
        return
    estado.lidas = leitura.ler_telas(estado.palestras, estado.visual)
    print()
    leitura.transcrever_todas(estado.palestras)
    estado.transcricoes = leitura.carregar_transcricoes(estado.palestras)


def acao_conteudo(estado: Estado) -> None:
    titulo("7. ANÁLISE: TIPO DE TELA, FERRAMENTAS, FALA E AS TRÊS ETAPAS")
    if not exigir(bool(estado.lidas and estado.transcricoes), "Leia as telas e transcreva as falas (opção 6)."):
        return
    estado.analise = conteudo.analisar(estado.palestras, estado.visual, estado.lidas, estado.transcricoes)
    conteudo.imprimir(estado.analise)


def acao_graficos(estado: Estado) -> None:
    titulo("8. GRÁFICOS, DASHBOARD E PAINEL DAS TRÊS ETAPAS")
    if not exigir(bool(estado.analise), "Rode a análise primeiro (opção 7)."):
        return
    visualizacao.gerar_todos(estado.analise, estado.visual, estado.lidas, estado.palestras)


def contexto(texto: str, padrao: re.Pattern[str], largura: int = 70) -> str:
    """O achado com 70 caracteres de cada lado, como a KWIC da etapa 1."""
    casamento = padrao.search(texto)
    return texto[max(casamento.start() - largura, 0):casamento.end() + largura]


def acao_buscar(estado: Estado) -> None:
    """
    Busca por regex no que foi dito e no que estava escrito na tela.

    É a KWIC (keyword in context) das etapas anteriores. Cada achado tem o
    segundo do vídeo, então o programa abre o QUADRO daquele momento e toca
    o áudio: a turma vê o slide e ouve a frase.
    """
    titulo("9. BUSCA NA FALA E NA TELA, COM O QUADRO E O ÁUDIO DO MOMENTO")
    if not exigir(bool(estado.transcricoes or estado.lidas), "Leia as telas e transcreva as falas (opção 6)."):
        return

    entrada = input("  padrão regex (ex.: claude|copilot): ").strip()
    if not entrada:
        return
    try:
        padrao = re.compile(entrada, re.IGNORECASE)
    except re.error as erro:
        print(f"  [!] regex inválida: {erro}")
        return

    achados = conteudo.buscar(padrao, estado.palestras, estado.transcricoes, estado.lidas, estado.visual)
    for numero, achado in enumerate(achados[:12], start=1):
        minuto, segundo = divmod(int(achado["inicio"]), 60)
        print(f"\n  ({numero:>2}) [{achado['fonte']}] {achado['palestra']['titulo'][:40]}, {minuto}:{segundo:02d}")
        print(f"       ...{contexto(achado['texto'], padrao)}...")
        print(f"       casou com: {achado['casou']!r}")
    print(f"\n  {len(achados)} momentos com o padrão (mostrando {min(len(achados), 12)}).")
    if not achados:
        return

    escolha = input("\n  número do momento para ver e ouvir (enter pula): ").strip()
    if not escolha.isdigit() or not 1 <= int(escolha) <= min(len(achados), 12):
        return
    achado = achados[int(escolha) - 1]
    if not coleta.video_disponivel(achado["palestra"]):
        print("  [!] o vídeo dessa palestra não está no disco (opção 1)")
        return
    caminho = coleta.caminho_video(achado["palestra"])
    destino = config.DIR_TRECHOS / "quadro_busca.jpg"
    cv2.imwrite(str(destino), quadros.quadro_em(caminho, achado["inicio"] + 0.5))
    quadros.abrir(destino)
    # Um segundo de folga antes, para não cortar a primeira sílaba.
    inicio = max(achado["inicio"] - 1, 0)
    fim = inicio + min(achado["fim"] - achado["inicio"] + 2, 15)
    sinal = leitura.audio(achado["palestra"])
    leitura.tocar(sinal[int(inicio * config.TAXA_AMOSTRAGEM):int(fim * config.TAXA_AMOSTRAGEM)], "trecho_busca.wav")


def acao_pipeline(estado: Estado) -> None:
    titulo("PIPELINE COMPLETO")
    acao_coletar(estado)
    acao_visual(estado)
    acao_ler(estado)
    acao_conteudo(estado)
    acao_graficos(estado)
    titulo("CONCLUÍDO")
    print(f"  dados     -> {config.DIR_PROCESSADOS}")
    print(f"  gráficos  -> {config.DIR_SAIDA}\n")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

OPCOES = {
    "1": ("Coletar palestras (agenda XML da FOSDEM) e baixar vídeo e legenda", acao_coletar),
    "2": ("Ver resumo das palestras coletadas", acao_resumo),
    "3": ("Demonstrar um quadro: regiões, diferença, bordas e rostos", acao_demo_quadro),
    "4": ("Extrair características visuais de todas (OpenCV)", acao_visual),
    "5": ("Demonstrar a leitura: OCR do slide, Whisper e legenda oficial", acao_demo_leitura),
    "6": ("Ler as telas (OCR) e transcrever as falas (Whisper)", acao_ler),
    "7": ("Analisar: tipo de tela, ferramentas, fala e as três etapas", acao_conteudo),
    "8": ("Gerar gráficos, dashboard e painel das três etapas", acao_graficos),
    "9": ("Buscar palavra na fala e na tela e ver o momento (KWIC)", acao_buscar),
    "0": ("Rodar o pipeline completo", acao_pipeline),
}


def estado_atual(estado: Estado) -> str:
    com_video = sum(coleta.video_disponivel(p) for p in estado.palestras)
    marcas = [
        f"palestras:{len(estado.palestras)}",
        f"vídeo:{com_video}",
        f"visual:{len(estado.visual.get('palestras', {}))}",
        f"telas lidas:{len(estado.lidas)}",
        f"falas:{len(estado.transcricoes)}",
        f"análise:{'ok' if estado.analise else '-'}",
    ]
    return " | ".join(marcas)


def menu() -> None:
    estado = Estado()
    estado.carregar_do_disco()

    while True:
        titulo("DevIA Radar - Etapa 3 (vídeo) - Dados Não Estruturados (UFPB)")
        print(f" estado: {estado_atual(estado)}")
        print("-" * LARGURA)
        for chave in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"):
            print(f"  [{chave}] {OPCOES[chave][0]}")
        print("  [r] Recoletar do zero (ignora o cache)")
        print("  [s] Sair")
        print("-" * LARGURA)

        escolha = input("  opção: ").strip().lower()

        if escolha in ("s", "sair", "q"):
            print("\n  até mais.\n")
            return
        if escolha == "r":
            acao_coletar(estado, forcar=True)
            continue
        if escolha in OPCOES:
            try:
                OPCOES[escolha][1](estado)
            except KeyboardInterrupt:
                print("\n  [interrompido pelo usuário]")
            except Exception as erro:  # noqa: BLE001 (a apresentação não pode morrer no meio)
                print(f"\n  [erro] {type(erro).__name__}: {erro}")
        else:
            print("\n  [!] opção inválida")

        input("\n  enter para voltar ao menu...")


if __name__ == "__main__":
    if "--tudo" in sys.argv:
        estado = Estado()
        estado.carregar_do_disco()
        acao_pipeline(estado)
    else:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\n  encerrado.\n")
