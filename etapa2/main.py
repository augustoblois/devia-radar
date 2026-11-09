"""
DevIA Radar - Etapa 2 (áudio)
Como desenvolvedores FALAM de Inteligência Artificial no desenvolvimento de
software, a partir de episódios reais de podcast.

Disciplina: Dados Não Estruturados (UFPB)
Etapa 2: análise de áudio com módulos Python.

Uso:
    python main.py            abre o menu interativo
    python main.py --tudo     roda o pipeline inteiro sem perguntar nada

Pipeline:
    iTunes Search API -> librosa (sinal: energia, pausas, espectrograma,
    MFCC, altura da voz) -> Whisper (transcrição) -> regex + spaCy + VADER
    (conteúdo da fala) -> matplotlib (dashboard)
"""

from __future__ import annotations

import re
import sys
from typing import Any

import coleta
import config
import conteudo
import sinal
import transcricao
import visualizacao

LARGURA = 72


class Estado:
    """Guarda o que já foi carregado na sessão, para não reler o disco toda hora."""

    def __init__(self) -> None:
        self.episodios: list[dict[str, Any]] = []
        self.acustica: dict[str, Any] = {}
        self.transcricoes: dict[str, Any] = {}
        self.fala: dict[str, Any] = {}

    def carregar_do_disco(self) -> None:
        if config.ARQ_EPISODIOS.exists():
            self.episodios = coleta.selecionar()
        self.acustica = sinal.carregar_analise()
        self.transcricoes = transcricao.carregar_transcricoes(self.episodios)
        self.fala = conteudo.carregar_analise()


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
    titulo("1. COLETA VIA iTUNES SEARCH API E DOWNLOAD DO ÁUDIO")
    estado.episodios = coleta.coletar(forcar=forcar)
    coleta.resumo(estado.episodios)


def acao_resumo(estado: Estado) -> None:
    titulo("2. RESUMO DOS EPISÓDIOS COLETADOS")
    coleta.resumo(estado.episodios)


def acao_demo_sinal(estado: Estado) -> None:
    titulo("3. DEMONSTRAÇÃO: DO SOM AOS NÚMEROS (librosa)")
    sinal.demonstrar(estado.episodios)


def acao_acustica(estado: Estado) -> None:
    titulo("4. CARACTERÍSTICAS ACÚSTICAS DE TODOS OS EPISÓDIOS")
    if not exigir(bool(estado.episodios), "Colete os episódios primeiro (opção 1)."):
        return
    estado.acustica = sinal.analisar(estado.episodios)


def acao_demo_whisper(estado: Estado) -> None:
    titulo("5. DEMONSTRAÇÃO: TRANSCRIÇÃO AO VIVO (Whisper)")
    transcricao.demonstrar(estado.episodios)


def acao_transcrever(estado: Estado) -> None:
    titulo("6. TRANSCRIÇÃO DE TODOS OS EPISÓDIOS (Whisper)")
    if not exigir(bool(estado.episodios), "Colete os episódios primeiro (opção 1)."):
        return
    transcricao.transcrever_todos(estado.episodios)
    estado.transcricoes = transcricao.carregar_transcricoes(estado.episodios)


def acao_fala(estado: Estado) -> None:
    titulo("7. ANÁLISE DA FALA: FERRAMENTAS, SENTIMENTO E VOZ")
    if not exigir(bool(estado.transcricoes), "Transcreva os episódios primeiro (opção 6)."):
        return
    if not exigir(bool(estado.acustica), "Extraia as características acústicas (opção 4)."):
        return
    estado.fala = conteudo.analisar(estado.episodios, estado.transcricoes, estado.acustica)
    conteudo.imprimir(estado.fala)


def acao_graficos(estado: Estado) -> None:
    titulo("8. GERAÇÃO DOS GRÁFICOS E DO DASHBOARD")
    if not exigir(bool(estado.fala), "Rode a análise da fala primeiro (opção 7)."):
        return
    visualizacao.gerar_todos(estado.fala, estado.acustica, estado.episodios)


def acao_buscar(estado: Estado) -> None:
    """
    Busca por regex na fala transcrita e toca o momento encontrado.

    É a KWIC (keyword in context) da etapa 1. A diferença é que cada achado
    tem marca de tempo, então o programa recorta o áudio daquele segundo e
    toca, e a turma OUVE a frase em vez de só ler.
    """
    titulo("9. BUSCA NA FALA E REPRODUÇÃO DO MOMENTO (KWIC com áudio)")
    if not exigir(bool(estado.transcricoes), "Transcreva os episódios primeiro (opção 6)."):
        return

    entrada = input("  padrão regex (ex.: hallucinat\\w+): ").strip()
    if not entrada:
        return
    try:
        padrao = re.compile(entrada, re.IGNORECASE)
    except re.error as erro:
        print(f"  [!] regex inválida: {erro}")
        return

    achados = conteudo.buscar(padrao, estado.episodios, estado.transcricoes)
    for numero, achado in enumerate(achados[:12], start=1):
        minuto, segundo = divmod(int(achado["inicio"]), 60)
        print(f"\n  ({numero:>2}) [{achado['episodio']['podcast'][:30]}, {minuto}:{segundo:02d}]")
        print(f"       ...{achado['texto'][:150]}...")
        print(f"       casou com: {achado['casou']!r}")
    print(f"\n  {len(achados)} momentos com o padrão (mostrando {min(len(achados), 12)}).")
    if not achados:
        return

    escolha = input("\n  número do momento para ouvir (enter pula): ").strip()
    if not escolha.isdigit() or not 1 <= int(escolha) <= min(len(achados), 12):
        return
    achado = achados[int(escolha) - 1]
    if not coleta.audio_disponivel(achado["episodio"]):
        print("  [!] o áudio desse episódio não está no disco (opção 1)")
        return
    # Um segundo de folga antes e depois, para não cortar a primeira sílaba.
    inicio = max(achado["inicio"] - 1, 0)
    duracao = min(achado["fim"] - achado["inicio"] + 2, 15)
    trecho = sinal.carregar(config.DIR_AUDIO / achado["episodio"]["arquivo"], inicio, duracao)
    sinal.tocar(sinal.salvar_wav(trecho, "trecho_busca.wav"))


def acao_pipeline(estado: Estado) -> None:
    titulo("PIPELINE COMPLETO")
    acao_coletar(estado)
    acao_acustica(estado)
    acao_transcrever(estado)
    acao_fala(estado)
    acao_graficos(estado)
    titulo("CONCLUÍDO")
    print(f"  dados     -> {config.DIR_PROCESSADOS}")
    print(f"  gráficos  -> {config.DIR_SAIDA}\n")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

OPCOES = {
    "1": ("Coletar episódios (iTunes Search API) e baixar o áudio", acao_coletar),
    "2": ("Ver resumo dos episódios coletados", acao_resumo),
    "3": ("Demonstrar o sinal de um trecho (onda, Mel, MFCC) e ouvir", acao_demo_sinal),
    "4": ("Extrair características acústicas de todos (librosa)", acao_acustica),
    "5": ("Demonstrar a transcrição ao vivo (Whisper)", acao_demo_whisper),
    "6": ("Transcrever todos os episódios (Whisper)", acao_transcrever),
    "7": ("Analisar a fala: ferramentas, sentimento e voz", acao_fala),
    "8": ("Gerar gráficos e dashboard", acao_graficos),
    "9": ("Buscar palavra na fala e ouvir o momento (KWIC)", acao_buscar),
    "0": ("Rodar o pipeline completo", acao_pipeline),
}


def estado_atual(estado: Estado) -> str:
    com_audio = sum(coleta.audio_disponivel(e) for e in estado.episodios)
    marcas = [
        f"episódios:{len(estado.episodios)}",
        f"áudio:{com_audio}",
        f"acústica:{'ok' if estado.acustica else '-'}",
        f"transcrições:{len(estado.transcricoes)}",
        f"fala:{'ok' if estado.fala else '-'}",
    ]
    return " | ".join(marcas)


def menu() -> None:
    estado = Estado()
    estado.carregar_do_disco()

    while True:
        titulo("DevIA Radar - Etapa 2 (áudio) - Dados Não Estruturados (UFPB)")
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
