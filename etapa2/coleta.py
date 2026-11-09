"""
Etapa 2 - COLETA.

Busca episódios de podcast pela iTunes Search API, a busca pública de
podcasts da Apple, e baixa o áudio de cada episódio escolhido.

Conceitos usados aqui:
- Web API: a mesma ideia da etapa 1. Um endereço na web que devolve dados em
  JSON. Aqui, cada resultado é um episódio com título, podcast, data,
  duração e o link direto do arquivo de áudio (MP3).
- Filtro de amostra: a API devolve episódio em espanhol, episódio de 3
  minutos, episódio de 2021. Os filtros abaixo deixam só o que serve para a
  pergunta de pesquisa, e cada filtro tem um motivo escrito do lado.
- Feed RSS: arquivo XML que todo podcast publica com a lista dos episódios.
  É dele que o aplicativo de podcast tira o que mostra. Aqui ele serve para
  ler a declaração de voz gerada por IA, quando o podcast faz essa
  declaração.
- Download em streaming: o MP3 chega em pedaços de 1 MB gravados direto no
  disco, sem carregar o arquivo inteiro na memória.
- Cache: arquivo já baixado não é baixado de novo. A apresentação funciona
  sem internet na sala.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from typing import Any

import requests

import config

CABECALHOS = {"User-Agent": "DevIA-Radar/2.0 (projeto academico UFPB)"}

# Palavras funcionais do inglês. Se elas somam uma boa fração do texto, o
# texto está em inglês. É um detector de idioma de 1 linha, suficiente para
# descartar os episódios em espanhol e italiano que a busca devolve.
PALAVRAS_INGLES = {
    "the", "and", "you", "with", "for", "this", "that", "what", "how", "are",
    "your", "from", "about", "we", "our", "is", "in", "of", "to", "it", "on",
    "can", "will", "be",
}

# O episódio precisa falar de IA E de desenvolvimento de software. Só IA
# traz episódio de investimento; só software traz episódio sem IA.
RE_TEMA_IA = re.compile(
    r"\b(ai|llms?|copilot|claude|chatgpt|cursor|gpt|agents?|agentic|vibe)\b",
    re.IGNORECASE,
)
RE_TEMA_SOFTWARE = re.compile(
    r"\b(code|coding|coder|developers?|engineers?|engineering|programming|"
    r"programmers?|software|devs?)\b",
    re.IGNORECASE,
)

# O podcast declara no próprio feed que a voz é gerada por IA. Mesmo motivo
# do filtro de robôs da etapa 1: voz sintética lendo roteiro gerado não é
# desenvolvedor falando. A regra exige que a frase fale da NARRAÇÃO ou da
# VOZ do próprio programa, porque "AI-generated code" e "AI voice agents"
# aparecem em muito podcast humano como assunto do episódio.
RE_VOZ_SINTETICA = re.compile(
    r"\b(?:podcast|episode|show|series)\s+(?:uses|features|is|was)\s+(?:[\w-]+\s+){0,3}?"
    r"(?:ai|artificial intelligence)[- ](?:generated|produced|created)\s+"
    r"(?:narration|voices?|hosts?|speech)\b"
    r"|\b(?:narration|voices?|hosts?)\s+(?:is|are|was|were)\s+"
    r"(?:ai[- ]generated|generated\s+(?:by|with|using)\s+ai|synthetic)\b"
    r"|\b(?:narrated|voiced)\s+by\s+(?:an?\s+)?ai\b",
    re.IGNORECASE,
)


def _fracao_ingles(texto: str) -> float:
    palavras = re.findall(r"[a-z']+", texto.lower())
    return sum(p in PALAVRAS_INGLES for p in palavras) / max(len(palavras), 1)


def _normalizar(item: dict[str, Any], termo: str) -> dict[str, Any]:
    """Reduz a resposta da API aos campos que o projeto usa."""
    extensao = (item.get("episodeFileExtension") or "mp3").lower()
    return {
        "id": str(item["trackId"]),
        "termo_busca": termo,
        "podcast": item.get("collectionName", ""),
        "titulo": item.get("trackName", ""),
        "descricao": (item.get("description") or "")[:1500],
        "publicado_em": item.get("releaseDate", ""),
        "duracao_min": round((item.get("trackTimeMillis") or 0) / 60000, 1),
        "url_audio": item.get("episodeUrl", ""),
        "url_pagina": item.get("trackViewUrl", ""),
        "feed": item.get("feedUrl", ""),
        "arquivo": f"{item['trackId']}.{extensao}",
    }


def _motivo_descarte(item: dict[str, Any]) -> str | None:
    """Devolve por que o episódio não serve, ou None se ele passa."""
    texto = f"{item.get('trackName', '')} {item.get('description', '')}"
    duracao = (item.get("trackTimeMillis") or 0) / 60000
    if not item.get("episodeUrl"):
        return "sem áudio"
    if (item.get("releaseDate") or "")[:4] not in config.ANOS_ACEITOS:
        return "antigo"
    if not config.DURACAO_MIN <= duracao <= config.DURACAO_MAX:
        return "duração"
    if _fracao_ingles(texto) < 0.12:
        return "idioma"
    if not (RE_TEMA_IA.search(texto) and RE_TEMA_SOFTWARE.search(texto)):
        return "fora do tema"
    return None


def _voz_sintetica(feed: str, lidos: dict[str, bool]) -> bool:
    """
    Lê o feed RSS do podcast e procura a declaração de voz gerada por IA.

    Só conta a declaração do próprio podcast. Suspeita (fala quase sem
    pausa, roteiro genérico) vai para o diário de bordo e não vira regra:
    tirar episódio por palpite seria escolher a amostra pelo resultado.
    O dicionário `lidos` guarda a resposta de cada feed já lido.
    """
    if feed not in lidos:
        try:
            resposta = requests.get(feed, headers=CABECALHOS, timeout=30)
            resposta.raise_for_status()
            texto = re.sub(r"<[^>]+>", " ", html.unescape(resposta.text))
            lidos[feed] = bool(RE_VOZ_SINTETICA.search(texto))
        except requests.RequestException as erro:
            # Feed fora do ar não prova nada: o episódio segue na amostra.
            print(f"    [rede] feed ilegível ({type(erro).__name__}): {feed[:60]}")
            lidos[feed] = False
    return lidos[feed]


def buscar_termo(termo: str) -> list[dict[str, Any]]:
    """Uma consulta na iTunes Search API."""
    parametros = {
        "term": termo,
        "media": "podcast",
        "entity": "podcastEpisode",  # um resultado por episódio (o padrão é um por podcast)
        "limit": config.RESULTADOS_POR_TERMO,
        "country": "US",
    }
    resposta = requests.get(
        config.API_ITUNES, params=parametros, headers=CABECALHOS, timeout=30
    )
    resposta.raise_for_status()
    return resposta.json().get("results", [])


def selecionar(forcar: bool = False) -> list[dict[str, Any]]:
    """
    Consulta a API e escolhe os episódios da amostra.

    Regra de escolha: para cada termo, os primeiros episódios na ordem de
    relevância da própria API que passam nos filtros, no máximo um episódio
    por podcast. Sem esse teto, um único podcast prolífico ocupava metade da
    amostra e a análise viraria retrato de um apresentador só.

    Por último, o feed de cada candidato é lido atrás da declaração de voz
    gerada por IA. Quem declara sai, e o próximo episódio do mesmo termo
    entra no lugar, mantendo 2 episódios por termo.
    """
    if config.ARQ_EPISODIOS.exists() and not forcar:
        with config.ARQ_EPISODIOS.open(encoding="utf-8") as arquivo:
            episodios = json.load(arquivo)
        print(f"[cache] {len(episodios)} episódios lidos de {config.ARQ_EPISODIOS.name}")
        return episodios

    escolhidos: list[dict[str, Any]] = []
    podcasts_usados: set[str] = set()
    descartes: Counter[str] = Counter()
    feeds_lidos: dict[str, bool] = {}

    for termo in config.TERMOS_BUSCA:
        print(f"  buscando: {termo!r}")
        resultados = buscar_termo(termo)
        novos = 0
        for item in resultados:
            motivo = _motivo_descarte(item)
            if motivo is None and item.get("collectionName") in podcasts_usados:
                motivo = "podcast repetido"
            # Por último porque é a única regra que custa uma requisição.
            if motivo is None and _voz_sintetica(item.get("feedUrl", ""), feeds_lidos):
                motivo = "voz sintética declarada"
                print(f"    fora: {item.get('collectionName', '')[:50]} (voz sintética declarada no feed)")
            if motivo:
                descartes[motivo] += 1
                continue
            escolhidos.append(_normalizar(item, termo))
            podcasts_usados.add(item["collectionName"])
            novos += 1
            if novos == config.EPISODIOS_POR_TERMO:
                break
        print(f"    {len(resultados)} resultados, {novos} escolhidos")
        time.sleep(config.PAUSA_ENTRE_CONSULTAS)

    print("  descartes:", ", ".join(f"{m} {q}" for m, q in descartes.most_common()))

    with config.ARQ_EPISODIOS.open("w", encoding="utf-8") as arquivo:
        json.dump(escolhidos, arquivo, ensure_ascii=False, indent=2)
    print(f"[ok] {len(escolhidos)} episódios salvos em {config.ARQ_EPISODIOS.name}")
    return escolhidos


def baixar(episodio: dict[str, Any], tentativas: int = 3) -> bool:
    """
    Baixa o áudio de um episódio, em pedaços, se ainda não existir.

    Tenta até 3 vezes. Na primeira coleta, um episódio falhou porque o DNS
    não resolveu o servidor de anúncios por onde o link passa, e na segunda
    tentativa baixou normalmente. Foi instabilidade de rede.
    """
    destino = config.DIR_AUDIO / episodio["arquivo"]
    if destino.exists() and destino.stat().st_size > 0:
        return True

    parcial = destino.with_suffix(destino.suffix + ".parcial")
    for tentativa in range(1, tentativas + 1):
        try:
            with requests.get(
                episodio["url_audio"], headers=CABECALHOS, stream=True, timeout=60
            ) as resposta:
                resposta.raise_for_status()
                with parcial.open("wb") as arquivo:
                    for pedaco in resposta.iter_content(chunk_size=1 << 20):
                        arquivo.write(pedaco)
            # Grava com nome provisório e só renomeia no fim. Se a internet
            # cair no meio, o arquivo pela metade não passa por completo.
            parcial.replace(destino)
            return True
        except requests.RequestException as erro:
            parcial.unlink(missing_ok=True)
            print(f"    [rede] tentativa {tentativa}: {episodio['titulo'][:40]}: {type(erro).__name__}")
            if tentativa < tentativas:
                time.sleep(5 * tentativa)
    return False


def coletar(forcar: bool = False) -> list[dict[str, Any]]:
    """Escolhe os episódios e baixa o áudio de todos."""
    episodios = selecionar(forcar=forcar)
    for indice, episodio in enumerate(episodios, start=1):
        ok = baixar(episodio)
        tamanho = (config.DIR_AUDIO / episodio["arquivo"]).stat().st_size if ok else 0
        print(
            f"  [{indice:>2}/{len(episodios)}] {'ok ' if ok else 'ERRO'} "
            f"{tamanho / 1e6:>5.1f} MB  {episodio['podcast'][:38]}"
        )
    return episodios


def audio_disponivel(episodio: dict[str, Any]) -> bool:
    return (config.DIR_AUDIO / episodio["arquivo"]).exists()


def resumo(episodios: list[dict[str, Any]]) -> None:
    """Imprime um panorama rápido da amostra."""
    if not episodios:
        print("Nenhum episódio coletado ainda. Rode a opção de coleta primeiro.")
        return

    horas = sum(e["duracao_min"] for e in episodios) / 60
    datas = sorted(e["publicado_em"][:10] for e in episodios)
    baixados = sum(audio_disponivel(e) for e in episodios)

    print(f"  episódios ....... {len(episodios)} ({baixados} com áudio no disco)")
    print(f"  podcasts ........ {len({e['podcast'] for e in episodios})}")
    # Duração que o iTunes anuncia. O áudio baixado pode vir mais longo, com
    # anúncio inserido na hora do download (ver o diário de bordo).
    horas_texto = f"{horas:.1f}".replace(".", ",")
    print(f"  áudio anunciado . {horas_texto} horas")
    print(f"  período ......... {datas[0]} até {datas[-1]}")
    print("\n  amostra:")
    for e in episodios:
        print(f"    {e['duracao_min']:>4.0f} min  {e['podcast'][:34]:<34}  {e['titulo'][:44]}")


if __name__ == "__main__":
    resumo(coletar())
