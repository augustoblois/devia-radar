"""
Etapa 3 - COLETA.

Lê a agenda pública da FOSDEM (um arquivo XML com as mais de mil palestras
de cada ano), escolhe as que tratam de IA no desenvolvimento de software e
baixa o vídeo e a legenda de cada uma.

Conceitos usados aqui:
- XML: formato de dado SEMIESTRUTURADO. Tem marcação (<event>, <title>,
  <abstract>), mas o conteúdo de dentro é texto livre. O módulo
  xml.etree.ElementTree, que vem com o Python, lê a marcação; a regex lê o
  texto livre de dentro, como na etapa 1.
- Pontuação de relevância: cada palestra ganha pontos por termo de IA no
  título (peso 3, o título é a promessa da palestra) e no resumo (peso 1).
  Ficam as de maior pontuação.
- Licença: os vídeos da FOSDEM são Creative Commons Attribution. Pode
  analisar e mostrar quadros, citando a fonte.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

import config

CABECALHOS = {"User-Agent": "DevIA-Radar/3.0 (projeto academico UFPB)"}

RE_TERMOS_IA = re.compile(
    r"\b(copilot|chatgpt|claude|codex|cursor|gemini|llms?|large language models?|"
    r"ai[- ]assisted|ai[- ]generated|ai[- ]powered|generative ai|genai|"
    r"coding assistants?|code generation|vibe coding|ai agents?|agentic|"
    r"artificial intelligence|machine[- ]generated)\b",
    re.IGNORECASE,
)

# A palestra precisa falar de FAZER software com IA. Palestra sobre a IA
# como produto sai. A primeira versão da regra olhava o resumo, e a amostra
# encheu de palestra sobre filtro de spam com LLM e servidor de inferência,
# porque "code" e "open source" aparecem no resumo de quase tudo na FOSDEM.
# Agora o termo de desenvolvimento precisa estar no TÍTULO.
RE_DESENVOLVIMENTO = re.compile(
    r"\b(code|coding|developers?|development|programming|reviews?|testing|"
    r"tests|debug\w*|contribut\w*|communit(?:y|ies)|fix(?:es)?|rollouts?|"
    r"maintainers?)\b",
    re.IGNORECASE,
)

# Título sobre treinar ou servir modelo é IA como produto: sai da amostra.
RE_IA_COMO_PRODUTO = re.compile(r"\b(training|datasets?|inference|serving)\b", re.IGNORECASE)

RE_TAG_HTML = re.compile(r"<[^>]+>")


def _texto(no: ET.Element | None) -> str:
    """O resumo vem com HTML dentro do XML. A regex tira as tags."""
    if no is None or no.text is None:
        return ""
    return " ".join(RE_TAG_HTML.sub(" ", no.text).split())


def _minutos(duracao: str) -> int:
    horas, minutos = (duracao or "0:0").split(":")[:2]
    return int(horas) * 60 + int(minutos)


def _palestra(evento: ET.Element, ano: int) -> dict[str, Any] | None:
    """Extrai uma palestra do XML, ou None se ela não tem vídeo em AV1."""
    links = [l.get("href", "") for l in evento.findall("./links/link")]
    video = next((l for l in links if l.endswith(config.SUFIXO_VIDEO)), None)
    if not video:
        return None
    base = video[: -len(config.SUFIXO_VIDEO)]
    titulo = evento.findtext("title") or ""
    resumo = f"{_texto(evento.find('abstract'))} {_texto(evento.find('description'))}".strip()
    pontos_titulo = len(RE_TERMOS_IA.findall(titulo))
    return {
        "id": f"{ano}-{evento.get('id')}",
        "ano": ano,
        "titulo": titulo,
        "trilha": evento.findtext("track") or "",
        "palestrantes": [p.text for p in evento.findall("./persons/person") if p.text],
        "duracao_min": _minutos(evento.findtext("duration") or "0:0"),
        "resumo": resumo[:1500],
        "url_pagina": evento.findtext("url") or "",
        "url_video": video,
        "url_legenda": base + config.SUFIXO_LEGENDA,
        "arquivo": base.rsplit("/", 1)[-1] + config.SUFIXO_VIDEO,
        "pontos": pontos_titulo * 3 + len(RE_TERMOS_IA.findall(resumo)),
        "pontos_titulo": pontos_titulo,
        "tem_desenvolvimento": bool(RE_DESENVOLVIMENTO.search(titulo))
        and not RE_IA_COMO_PRODUTO.search(titulo),
    }


def candidatas() -> list[dict[str, Any]]:
    """Todas as palestras dos anos configurados, já pontuadas."""
    todas = []
    for ano, url in config.AGENDAS.items():
        print(f"  lendo a agenda de {ano}...")
        resposta = requests.get(url, headers=CABECALHOS, timeout=60)
        resposta.raise_for_status()
        raiz = ET.fromstring(resposta.content)
        eventos = raiz.findall(".//event")
        palestras = [p for p in (_palestra(e, ano) for e in eventos) if p]
        print(f"    {len(eventos)} eventos, {len(palestras)} com vídeo")
        todas.extend(palestras)
    return todas


def selecionar(forcar: bool = False) -> list[dict[str, Any]]:
    """
    Escolhe as palestras da amostra.

    Regras, em ordem: termo de IA no TÍTULO (só no resumo pega palestra que
    cita IA de passagem), termo de desenvolvimento de software também no
    título, nenhum termo de IA como produto no título (treino, dataset,
    inferência), duração entre 15 e 50 minutos. Das que passam, ficam as N
    de maior pontuação.
    """
    if config.ARQ_PALESTRAS.exists() and not forcar:
        with config.ARQ_PALESTRAS.open(encoding="utf-8") as arquivo:
            palestras = json.load(arquivo)
        print(f"[cache] {len(palestras)} palestras lidas de {config.ARQ_PALESTRAS.name}")
        return palestras

    todas = candidatas()
    aptas = [
        p for p in todas
        if p["pontos_titulo"] > 0
        and p["tem_desenvolvimento"]
        and config.DURACAO_MIN <= p["duracao_min"] <= config.DURACAO_MAX
    ]
    aptas.sort(key=lambda p: (-p["pontos"], -p["duracao_min"]))
    escolhidas = aptas[: config.N_PALESTRAS]
    print(f"  {len(todas)} palestras com vídeo, {len(aptas)} aptas, {len(escolhidas)} escolhidas")

    with config.ARQ_PALESTRAS.open("w", encoding="utf-8") as arquivo:
        json.dump(escolhidas, arquivo, ensure_ascii=False, indent=2)
    return escolhidas


def _baixar(url: str, destino, tentativas: int = 3) -> bool:
    """Download em pedaços de 1 MB, com nome provisório e nova tentativa."""
    if destino.exists() and destino.stat().st_size > 0:
        return True
    parcial = destino.with_name(destino.name + ".parcial")
    for tentativa in range(1, tentativas + 1):
        try:
            with requests.get(url, headers=CABECALHOS, stream=True, timeout=60) as resposta:
                resposta.raise_for_status()
                with parcial.open("wb") as arquivo:
                    for pedaco in resposta.iter_content(chunk_size=1 << 20):
                        arquivo.write(pedaco)
            parcial.replace(destino)
            return True
        except requests.RequestException as erro:
            parcial.unlink(missing_ok=True)
            print(f"    [rede] tentativa {tentativa}: {type(erro).__name__}")
            if tentativa < tentativas:
                time.sleep(5 * tentativa)
    return False


def caminho_video(palestra: dict[str, Any]):
    return config.DIR_VIDEOS / palestra["arquivo"]


def caminho_legenda(palestra: dict[str, Any]):
    return config.DIR_LEGENDAS / palestra["arquivo"].replace(config.SUFIXO_VIDEO, config.SUFIXO_LEGENDA)


def coletar(forcar: bool = False) -> list[dict[str, Any]]:
    """Escolhe as palestras e baixa vídeo e legenda de cada uma."""
    palestras = selecionar(forcar=forcar)
    for indice, palestra in enumerate(palestras, start=1):
        ok_video = _baixar(palestra["url_video"], caminho_video(palestra))
        ok_legenda = _baixar(palestra["url_legenda"], caminho_legenda(palestra))
        tamanho = caminho_video(palestra).stat().st_size / 1e6 if ok_video else 0
        print(
            f"  [{indice}/{len(palestras)}] vídeo {'ok ' if ok_video else 'ERRO'} {tamanho:>5.0f} MB  "
            f"legenda {'ok' if ok_legenda else '-'}  {palestra['titulo'][:46]}"
        )
    return palestras


def video_disponivel(palestra: dict[str, Any]) -> bool:
    return caminho_video(palestra).exists()


def resumo(palestras: list[dict[str, Any]]) -> None:
    if not palestras:
        print("Nenhuma palestra coletada ainda. Rode a opção de coleta primeiro.")
        return
    minutos = sum(p["duracao_min"] for p in palestras)
    baixadas = sum(video_disponivel(p) for p in palestras)
    print(f"  palestras ....... {len(palestras)} ({baixadas} com vídeo no disco)")
    # O horário reservado na agenda. A gravação tem outra duração, em geral
    # um pouco menor, e a duração real é medida na opção 4.
    print(f"  vídeo na agenda . {minutos // 60} h {minutos % 60:02d} min")
    print(f"  anos ............ {', '.join(str(a) for a in sorted({p['ano'] for p in palestras}))}")
    print(f"  licença ......... {config.LICENCA}")
    print("\n  amostra:")
    for p in palestras:
        print(f"    {p['ano']}  {p['duracao_min']:>2} min  {p['pontos']:>2} pts  {p['titulo'][:58]}")
        print(f"                       trilha: {p['trilha'][:50]}")


if __name__ == "__main__":
    resumo(coletar())
