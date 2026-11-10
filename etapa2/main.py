"""
DevIA Radar - Etapa 2 (semântica e sentimento)
O que desenvolvedores QUEREM DIZER quando escrevem sobre Inteligência
Artificial no desenvolvimento de software, nas mesmas issues do GitHub da
etapa 1.

Disciplina: Dados Não Estruturados (UFPB)
Etapa 2: análise semântica e de sentimentos, pipeline do spaCy, embeddings
e transformers com módulos Python.

Uso:
    python main.py            abre o menu interativo
    python main.py --tudo     roda o pipeline inteiro sem perguntar nada

Pipeline:
    corpus da etapa 1 -> spaCy (pipeline com entity_ruler e VADER: frases,
    ferramentas, entidades) -> sentence-transformers (vetores) -> KMeans e
    t-SNE (temas) -> transformer RoBERTa (sentimento) x VADER ->
    matplotlib (dashboard)
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

import config
import embeddings
import pipeline
import sentimento
import visualizacao

LARGURA = 72


class Estado:
    """Guarda o que já foi carregado na sessão, para não reler o disco toda hora."""

    def __init__(self) -> None:
        self.corpus: list[dict[str, Any]] = []
        self.frases: list[dict[str, Any]] = []
        self.lemas_por_issue: list[dict[str, int]] = []
        self.pipeline: dict[str, Any] = {}
        self.vetores_frases: np.ndarray | None = None
        self.vetores_issues: np.ndarray | None = None
        self.temas: dict[str, Any] = {}
        self.palavras: dict[str, Any] = {}
        self.sentimento: dict[str, Any] = {}

    def carregar_do_disco(self) -> None:
        self.corpus = pipeline.carregar_corpus()
        dados = pipeline.carregar_frases()
        self.frases = dados.get("frases", [])
        self.lemas_por_issue = dados.get("lemas_por_issue", [])
        self.pipeline = pipeline.carregar_analise()
        avaliadas = embeddings.frases_avaliadas(self.frases)
        if config.ARQ_VETORES_FRASES.exists() and config.ARQ_VETORES_ISSUES.exists():
            vf = np.load(config.ARQ_VETORES_FRASES)
            if len(vf) == len(avaliadas):
                self.vetores_frases, self.vetores_issues = vf, np.load(config.ARQ_VETORES_ISSUES)
        self.temas = embeddings.carregar_temas()
        self.sentimento = sentimento.carregar_analise()

    @property
    def avaliadas(self) -> list[dict[str, Any]]:
        return embeddings.frases_avaliadas(self.frases)


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


def perguntar(texto: str) -> str:
    """input() que devolve vazio quando não há teclado (python main.py --tudo < nada)."""
    try:
        return input(texto).strip()
    except EOFError:
        return ""


# ---------------------------------------------------------------------------
# Ações do menu
# ---------------------------------------------------------------------------


def acao_corpus(estado: Estado) -> None:
    titulo("1. O CORPUS: AS ISSUES DO GITHUB DA ETAPA 1")
    estado.corpus = pipeline.carregar_corpus()
    pipeline.resumo(estado.corpus)


def acao_demo_pipeline(estado: Estado) -> None:
    titulo("2. DEMONSTRAÇÃO: O PIPELINE DO spaCy, COMPONENTE A COMPONENTE")
    frase = perguntar("  frase em inglês (enter usa a de exemplo): ")
    pipeline.demonstrar(frase or None)


def acao_pipeline(estado: Estado, forcar: bool = False) -> None:
    titulo("3. O CORPUS PELO PIPELINE: FRASES, FERRAMENTAS E ENTIDADES")
    estado.pipeline = pipeline.processar(estado.corpus, forcar=forcar)
    dados = pipeline.carregar_frases()
    estado.frases, estado.lemas_por_issue = dados["frases"], dados["lemas_por_issue"]
    pipeline.imprimir(estado.pipeline)


def acao_demo_embeddings(estado: Estado) -> None:
    titulo("4. DEMONSTRAÇÃO: EMBEDDINGS ESTÁTICOS x CONTEXTUAIS")
    estado.palavras = embeddings.demonstrar_palavras([f["texto"] for f in estado.frases] or None)


def acao_temas(estado: Estado, forcar: bool = False) -> None:
    titulo("5. VETORES DO CORPUS E TEMAS (sentence-transformers e KMeans)")
    if not exigir(bool(estado.frases), "Passe o corpus pelo pipeline primeiro (opção 3)."):
        return
    estado.vetores_frases, estado.vetores_issues = embeddings.vetorizar(estado.corpus, estado.frases, forcar=forcar)
    estado.temas = embeddings.agrupar(estado.corpus, estado.vetores_issues, estado.lemas_por_issue)
    embeddings.imprimir_temas(estado.temas)


def acao_busca(estado: Estado) -> None:
    """
    Busca pelo significado: a consulta vira vetor e o programa devolve as
    frases de vetor mais parecido, mesmo sem nenhuma palavra em comum.
    """
    titulo("6. BUSCA SEMÂNTICA: FRASES PELO SIGNIFICADO")
    if not exigir(estado.vetores_frases is not None, "Calcule os vetores do corpus primeiro (opção 5)."):
        return
    print("  Escreva em inglês o que procura. Enter vazio volta ao menu.")
    print("  Exemplo: who owns the copyright of code written by a machine")
    while consulta := perguntar("\n  consulta: "):
        embeddings.demonstrar_busca(consulta, estado.avaliadas, estado.vetores_frases, estado.corpus)


def acao_demo_sentimento(estado: Estado) -> None:
    titulo("7. DEMONSTRAÇÃO: O TRANSFORMER DE SENTIMENTO x VADER")
    frase = perguntar("  frase em inglês (enter usa as de exemplo): ")
    sentimento.demonstrar([frase] if frase else None)


def acao_sentimento(estado: Estado, forcar: bool = False) -> None:
    titulo("8. SENTIMENTO DE TODAS AS FRASES: TRANSFORMER x VADER")
    if not exigir(bool(estado.frases), "Passe o corpus pelo pipeline primeiro (opção 3)."):
        return
    if not estado.temas:
        print("  (sem os temas da opção 5, a análise sai sem o sentimento por tema)")
    probabilidades = sentimento.classificar_corpus(estado.avaliadas, forcar=forcar)
    estado.sentimento = sentimento.comparar(estado.avaliadas, probabilidades, estado.corpus, estado.temas)
    sentimento.imprimir(estado.sentimento, estado.temas)


def acao_graficos(estado: Estado) -> None:
    titulo("9. GERAÇÃO DOS GRÁFICOS E DO DASHBOARD")
    if not exigir(bool(estado.sentimento), "Rode a análise de sentimento primeiro (opção 8)."):
        return
    if not estado.palavras:
        print("  calculando os exemplos de vetores da opção 4...")
        estado.palavras = embeddings.demonstrar_palavras([f["texto"] for f in estado.frases] or None)
    visualizacao.gerar_todos(estado.pipeline, estado.palavras, estado.temas, estado.sentimento)


def acao_tudo(estado: Estado, forcar: bool = False) -> None:
    titulo("PIPELINE COMPLETO")
    acao_corpus(estado)
    acao_pipeline(estado, forcar=forcar)
    acao_temas(estado, forcar=forcar)
    acao_sentimento(estado, forcar=forcar)
    acao_graficos(estado)
    titulo("CONCLUÍDO")
    print(f"  dados     -> {config.DIR_PROCESSADOS}")
    print(f"  gráficos  -> {config.DIR_SAIDA}\n")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

OPCOES = {
    "1": ("Ler o corpus da etapa 1 (979 issues) e ver o resumo", acao_corpus),
    "2": ("Demonstrar o pipeline do spaCy numa frase, componente a componente", acao_demo_pipeline),
    "3": ("Passar o corpus pelo pipeline: frases, ferramentas, entidades", acao_pipeline),
    "4": ("Demonstrar embeddings: vetor estático x contextual", acao_demo_embeddings),
    "5": ("Calcular os vetores do corpus e achar os temas (KMeans)", acao_temas),
    "6": ("Busca semântica: achar frases pelo significado", acao_busca),
    "7": ("Demonstrar o transformer de sentimento numa frase (x VADER)", acao_demo_sentimento),
    "8": ("Sentimento de todas as frases: transformer x VADER", acao_sentimento),
    "9": ("Gerar gráficos e dashboard", acao_graficos),
    "0": ("Rodar o pipeline completo", acao_tudo),
}


def estado_atual(estado: Estado) -> str:
    marcas = [
        f"issues:{len(estado.corpus)}",
        f"frases:{len(estado.frases)}",
        f"vetores:{'ok' if estado.vetores_frases is not None else '-'}",
        f"temas:{estado.temas.get('k', '-')}",
        f"sentimento:{'ok' if estado.sentimento else '-'}",
    ]
    return " | ".join(marcas)


def menu() -> None:
    estado = Estado()
    estado.carregar_do_disco()

    while True:
        titulo("DevIA Radar - Etapa 2 (semântica e sentimento) - Dados Não Estruturados (UFPB)")
        print(f" estado: {estado_atual(estado)}")
        print("-" * LARGURA)
        for chave in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"):
            print(f"  [{chave}] {OPCOES[chave][0]}")
        print("  [r] Refazer tudo do zero (ignora o cache)")
        print("  [s] Sair")
        print("-" * LARGURA)

        escolha = input("  opção: ").strip().lower()

        if escolha in ("s", "sair", "q"):
            print("\n  até mais.\n")
            return
        if escolha == "r":
            acao_tudo(estado, forcar=True)
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
        acao_tudo(estado)
    else:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\n  encerrado.\n")
