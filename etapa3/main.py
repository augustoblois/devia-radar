"""
DevIA Radar - Etapa 3 (áudio e vídeo)
Como desenvolvedores APRESENTAM Inteligência Artificial no desenvolvimento
de software, a partir de palestras gravadas da FOSDEM.

Disciplina: Dados Não Estruturados (UFPB)
Etapa 3: análise de áudio e vídeo com módulos Python.

Uso:
    python main.py            abre o menu interativo
    python main.py --tudo     roda o pipeline inteiro sem perguntar nada

Pipeline:
    agenda XML da FOSDEM -> PyAV + OpenCV (imagem: troca de slide, bordas,
    rosto) + librosa (som: pausas, energia, altura da voz) -> RapidOCR
    (texto da tela) + Whisper (fala) + legenda oficial -> regex + spaCy +
    VADER (conteúdo) -> matplotlib (dashboard e painel final: o escrito
    nas issues x o falado nas palestras)
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
import sinal
import visualizacao

LARGURA = 72


class Estado:
    """Guarda o que já foi carregado na sessão, para não reler o disco toda hora."""

    def __init__(self) -> None:
        self.palestras: list[dict[str, Any]] = []
        self.visual: dict[str, Any] = {}
        self.acustica: dict[str, Any] = {}
        self.lidas: dict[str, Any] = {}
        self.transcricoes: dict[str, Any] = {}
        self.analise: dict[str, Any] = {}

    def carregar_do_disco(self) -> None:
        if config.ARQ_PALESTRAS.exists():
            self.palestras = coleta.selecionar()
        self.visual = quadros.carregar_analise()
        self.acustica = sinal.carregar_analise()
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


def acao_demo_som(estado: Estado) -> None:
    titulo("5. DEMONSTRAÇÃO: DO SOM AOS NÚMEROS (librosa)")
    sinal.demonstrar(estado.palestras)


def acao_acustica(estado: Estado) -> None:
    titulo("6. CARACTERÍSTICAS DO SOM DE TODAS AS PALESTRAS (librosa)")
    if not exigir(bool(estado.palestras), "Colete as palestras primeiro (opção 1)."):
        return
    estado.acustica = sinal.analisar(estado.palestras)


def acao_demo_leitura(estado: Estado) -> None:
    titulo("7. DEMONSTRAÇÃO: TELA (OCR), FALA (WHISPER) E LEGENDA OFICIAL")
    leitura.demonstrar(estado.palestras, estado.visual)


def acao_ler(estado: Estado) -> None:
    titulo("8. LEITURA DAS TELAS (OCR) E TRANSCRIÇÃO DAS FALAS (WHISPER)")
    if not exigir(bool(estado.visual), "Extraia as características visuais primeiro (opção 4)."):
        return
    estado.lidas = leitura.ler_telas(estado.palestras, estado.visual)
    print()
    leitura.transcrever_todas(estado.palestras)
    estado.transcricoes = leitura.carregar_transcricoes(estado.palestras)


def acao_conteudo(estado: Estado) -> None:
    titulo("9. ANÁLISE: TELA, VOZ, FERRAMENTAS, FALA E O ESCRITO x O FALADO")
    if not exigir(bool(estado.acustica), "Meça o som das palestras primeiro (opção 6)."):
        return
    if not exigir(bool(estado.lidas and estado.transcricoes), "Leia as telas e transcreva as falas (opção 8)."):
        return
    estado.analise = conteudo.analisar(estado.palestras, estado.visual, estado.lidas, estado.transcricoes,
                                       estado.acustica)
    conteudo.imprimir(estado.analise)


def acao_graficos(estado: Estado) -> None:
    titulo("10. GRÁFICOS, DASHBOARD E PAINEL FINAL")
    if not exigir(bool(estado.analise), "Rode a análise primeiro (opção 9)."):
        return
    visualizacao.gerar_todos(estado.analise, estado.visual, estado.lidas, estado.acustica, estado.palestras)


def contexto(texto: str, padrao: re.Pattern[str], largura: int = 70) -> str:
    """O achado com 70 caracteres de cada lado, como a KWIC da etapa 1."""
    casamento = padrao.search(texto)
    return texto[max(casamento.start() - largura, 0):casamento.end() + largura]


def acao_buscar(estado: Estado) -> None:
    """
    Busca por regex no que foi dito e no que estava escrito na tela.

    É a KWIC (keyword in context) da etapa 1. Cada achado tem o segundo do
    vídeo, então o programa abre o QUADRO daquele momento e toca o áudio: a
    turma vê o slide e ouve a frase.
    """
    titulo("11. BUSCA NA FALA E NA TELA, COM O QUADRO E O ÁUDIO DO MOMENTO")
    if not exigir(bool(estado.transcricoes or estado.lidas), "Leia as telas e transcreva as falas (opção 8)."):
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
    som = leitura.audio(achado["palestra"])
    leitura.tocar(som[int(inicio * config.TAXA_AMOSTRAGEM):int(fim * config.TAXA_AMOSTRAGEM)], "trecho_busca.wav")


def acao_pipeline(estado: Estado) -> None:
    titulo("PIPELINE COMPLETO")
    acao_coletar(estado)
    acao_visual(estado)
    acao_acustica(estado)
    acao_ler(estado)
    acao_conteudo(estado)
    acao_graficos(estado)
    titulo("CONCLUÍDO")
    print(f"  dados     -> {config.DIR_PROCESSADOS}")
    print(f"  gráficos  -> {config.DIR_SAIDA}\n")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

# Na ordem do menu: imagem (3 e 4), som (5 e 6), tela e fala (7 e 8), resultado.
OPCOES = {
    "1": ("Coletar palestras (agenda XML da FOSDEM), vídeo e legenda", acao_coletar),
    "2": ("Ver resumo das palestras coletadas", acao_resumo),
    "3": ("Demonstrar um quadro: regiões, diferença, bordas e rostos", acao_demo_quadro),
    "4": ("Extrair características visuais de todas (OpenCV)", acao_visual),
    "5": ("Demonstrar o som: onda, energia, espectrograma, MFCC e ouvir", acao_demo_som),
    "6": ("Medir o som de todas: pausas, energia e altura da voz (librosa)", acao_acustica),
    "7": ("Demonstrar a leitura: OCR do slide, Whisper e legenda oficial", acao_demo_leitura),
    "8": ("Ler as telas (OCR) e transcrever as falas (Whisper)", acao_ler),
    "9": ("Analisar: tela, voz, ferramentas, fala e o escrito x o falado", acao_conteudo),
    "10": ("Gerar gráficos, dashboard e painel final", acao_graficos),
    "11": ("Buscar palavra na fala e na tela e ver o momento (KWIC)", acao_buscar),
    "0": ("Rodar o pipeline completo", acao_pipeline),
}


def estado_atual(estado: Estado) -> str:
    com_video = sum(coleta.video_disponivel(p) for p in estado.palestras)
    marcas = [
        f"palestras:{len(estado.palestras)}",
        f"vídeo:{com_video}",
        f"visual:{len(estado.visual.get('palestras', {}))}",
        f"som:{len(estado.acustica.get('palestras', {}))}",
        f"telas lidas:{len(estado.lidas)}",
        f"falas:{len(estado.transcricoes)}",
        f"análise:{'ok' if estado.analise else '-'}",
    ]
    return " | ".join(marcas)


def menu() -> None:
    estado = Estado()
    estado.carregar_do_disco()

    while True:
        titulo("DevIA Radar - Etapa 3 (áudio e vídeo) - Dados Não Estruturados (UFPB)")
        print(f" estado: {estado_atual(estado)}")
        print("-" * LARGURA)
        for chave, (texto, _) in OPCOES.items():
            print(f"  {'[' + chave + ']':<5}{texto}")
        print(f"  {'[r]':<5}Recoletar do zero (ignora o cache)")
        print(f"  {'[s]':<5}Sair")
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
