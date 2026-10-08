"""
DevIA Radar - Etapa 1
Analisador de discussoes sobre Inteligência Artificial no desenvolvimento
de software, a partir de issues reais do GitHub.

Disciplina: Dados Não Estruturados (UFPB)
Etapa 1: coleta via Web API + PLN com expressões regulares e spaCy.

Uso:
    python main.py            abre o menu interativo
    python main.py --tudo     roda o pipeline inteiro sem perguntar nada

Pipeline:
    GitHub API -> regex (limpeza e extração) -> spaCy (PLN) -> VADER
    (sentimento) -> matplotlib (dashboard)
"""

from __future__ import annotations

import re
import sys
from typing import Any

import coleta
import config
import limpeza
import pln
import sentimento
import visualizacao

LARGURA = 72


class Estado:
    """Guarda o que já foi carregado na sessão, para não reler o disco toda hora."""

    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self.corpus: list[dict[str, Any]] = []
        self.analise_pln: dict[str, Any] = {}
        self.analise_sent: dict[str, Any] = {}

    def carregar_do_disco(self) -> None:
        if config.ARQ_ISSUES.exists():
            import json

            with config.ARQ_ISSUES.open(encoding="utf-8") as arquivo:
                self.issues = json.load(arquivo)
        self.corpus = limpeza.carregar_corpus()
        self.analise_pln = pln.carregar_analise()
        self.analise_sent = sentimento.carregar_analise()


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
# Acoes do menu
# ---------------------------------------------------------------------------


def acao_coletar(estado: Estado, forcar: bool = False) -> None:
    titulo("1. COLETA VIA GITHUB SEARCH API")
    estado.issues = coleta.coletar(forcar=forcar)
    coleta.resumo(estado.issues)


def acao_resumo(estado: Estado) -> None:
    titulo("2. RESUMO DO CORPUS COLETADO")
    coleta.resumo(estado.issues)


def acao_demo_regex(estado: Estado) -> None:
    titulo("3. DEMONSTRAÇÃO DAS EXPRESSÕES REGULARES")
    limpeza.demonstrar(estado.issues)


def acao_limpar(estado: Estado) -> None:
    titulo("4. LIMPEZA E EXTRAÇÃO DE PADRÕES COM REGEX")
    if not exigir(bool(estado.issues), "Colete as issues primeiro (opção 1)."):
        return
    estado.corpus = limpeza.processar(estado.issues)


def acao_demo_spacy(_: Estado) -> None:
    titulo("5. DEMONSTRAÇÃO DO spaCy TOKEN A TOKEN")
    pln.demonstrar()


def acao_pln(estado: Estado) -> None:
    titulo("6. PROCESSAMENTO DE LINGUAGEM NATURAL COM spaCy")
    if not exigir(bool(estado.corpus), "Rode a limpeza primeiro (opção 4)."):
        return
    estado.analise_pln = pln.analisar(estado.corpus)
    pln.imprimir(estado.analise_pln)


def acao_sentimento(estado: Estado) -> None:
    titulo("7. ANÁLISE DE SENTIMENTO COM VADER")
    if not exigir(bool(estado.corpus), "Rode a limpeza primeiro (opção 4)."):
        return
    estado.analise_sent = sentimento.analisar(estado.corpus)
    sentimento.imprimir(estado.analise_sent)


def acao_graficos(estado: Estado) -> None:
    titulo("8. GERAÇÃO DOS GRÁFICOS E DO DASHBOARD")
    if not exigir(bool(estado.corpus), "Rode a limpeza primeiro (opção 4)."):
        return
    if not exigir(bool(estado.analise_pln), "Rode a análise de PLN (opção 6)."):
        return
    if not exigir(bool(estado.analise_sent), "Rode o sentimento (opção 7)."):
        return
    visualizacao.gerar_todos(estado.corpus, estado.analise_pln, estado.analise_sent)


def acao_buscar(estado: Estado) -> None:
    """
    Busca por regex dentro do corpus e mostra a palavra no contexto.

    Essa visao chama-se KWIC (keyword in context) e e clássica em linguística
    de corpus: em vez de só contar quantas vezes o termo aparece, a gente le
    a vizinhança dele e entende COMO o termo e usado.
    """
    titulo("9. BUSCA POR PADRÃO NO CORPUS (KWIC)")
    if not exigir(bool(estado.corpus), "Rode a limpeza primeiro (opção 4)."):
        return

    entrada = input("  padrão regex (ex.: hallucinat\\w+): ").strip()
    if not entrada:
        return

    try:
        padrao = re.compile(entrada, re.IGNORECASE)
    except re.error as erro:
        print(f"  [!] regex inválida: {erro}")
        return

    total = 0
    mostrados = 0
    vistos: set[str] = set()
    for doc in estado.corpus:
        # uma ocorrência por documento já basta para ilustrar
        casamento = padrao.search(doc["texto_limpo"])
        if not casamento:
            continue
        total += 1
        if mostrados >= 12:
            continue
        inicio = max(casamento.start() - 70, 0)
        fim = min(casamento.end() + 70, len(doc["texto_limpo"]))
        trecho = doc["texto_limpo"][inicio:fim].replace("\n", " ")
        # Issues abertas a partir do mesmo template repetem o trecho idêntico.
        # A segunda cópia não ilustra nada, então só a primeira aparece.
        if trecho in vistos:
            continue
        vistos.add(trecho)
        print(f"\n  [{doc['repositorio']}] ...{trecho}...")
        print(f"       casou com: {casamento.group(0)!r}")
        mostrados += 1

    print(f"\n  {total} documentos com o padrão (mostrando {mostrados}).\n")


def acao_pipeline(estado: Estado) -> None:
    titulo("PIPELINE COMPLETO")
    acao_coletar(estado)
    acao_limpar(estado)
    acao_pln(estado)
    acao_sentimento(estado)
    acao_graficos(estado)
    titulo("CONCLUÍDO")
    print(f"  dados     -> {config.DIR_PROCESSADOS}")
    print(f"  gráficos  -> {config.DIR_SAIDA}\n")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

OPCOES = {
    "1": ("Coletar issues do GitHub (Web API)", acao_coletar),
    "2": ("Ver resumo do corpus coletado", acao_resumo),
    "3": ("Demonstrar as expressões regulares (antes e depois)", acao_demo_regex),
    "4": ("Limpar e extrair padrões com regex", acao_limpar),
    "5": ("Demonstrar o spaCy token a token", acao_demo_spacy),
    "6": ("Analisar o corpus com spaCy (lemas, POS, NER)", acao_pln),
    "7": ("Analisar sentimento com VADER", acao_sentimento),
    "8": ("Gerar gráficos e dashboard", acao_graficos),
    "9": ("Buscar padrão no corpus (KWIC)", acao_buscar),
    "0": ("Rodar o pipeline completo", acao_pipeline),
}


def estado_atual(estado: Estado) -> str:
    marcas = [
        f"issues:{len(estado.issues)}",
        f"corpus:{len(estado.corpus)}",
        f"pln:{'ok' if estado.analise_pln else '-'}",
        f"sentimento:{'ok' if estado.analise_sent else '-'}",
    ]
    return " | ".join(marcas)


def menu() -> None:
    estado = Estado()
    estado.carregar_do_disco()

    while True:
        titulo("DevIA Radar - Etapa 1 - Dados Não Estruturados (UFPB)")
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
            except Exception as erro:  # a apresentação não pode morrer no meio
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
