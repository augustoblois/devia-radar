"""
Etapa 1 - COLETA.

Busca issues reais do GitHub usando a Search API (uma Web API pública).
Cada consulta devolve issues cujo título ou corpo menciona um dos termos
ligados a IA no desenvolvimento de software.

Conceitos usados aqui:
- API REST: endereço na web que devolve dados estruturados (JSON) em vez de
  página HTML. A gente pede com uma requisicao HTTP GET e recebe texto JSON.
- Rate limit: teto de requisicoes por minuto que o servidor aceita. Passou do
  teto, ele responde 403 e a coleta morre. Por isso existe a pausa.
- Cache: salvamos o resultado em disco. Rodar de novo le do arquivo, entao a
  apresentação funciona mesmo sem internet na sala.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

import requests

import config


def _descobrir_token() -> str | None:
    """
    Procura um token do GitHub em duas fontes, nessa ordem.

    1. Variável de ambiente GITHUB_TOKEN / GH_TOKEN.
    2. GitHub CLI já autenticado na máquina (comando `gh auth token`).

    Token não é obrigatório. Sem ele a API funciona, só que com teto de
    requisicao bem mais baixo.
    """
    import os

    for nome in ("GITHUB_TOKEN", "GH_TOKEN"):
        valor = os.environ.get(nome)
        if valor:
            return valor.strip()

    if shutil.which("gh"):
        try:
            saida = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if saida.returncode == 0 and saida.stdout.strip():
                return saida.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

    return None


def _cabecalhos(token: str | None) -> dict[str, str]:
    cabecalhos = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "DevIA-Radar/1.0 (projeto academico UFPB)",
    }
    if token:
        cabecalhos["Authorization"] = f"Bearer {token}"
    return cabecalhos


def _normalizar(item: dict[str, Any], termo: str) -> dict[str, Any]:
    """
    Reduz a resposta gigante da API aos campos que o projeto realmente usa.

    A API devolve dezenas de campos por issue. Guardar tudo deixa o JSON com
    vários megabytes sem ganho nenhum para a análise de texto.
    """
    repo_url = item.get("repository_url", "")
    repositorio = repo_url.replace("https://api.github.com/repos/", "")

    return {
        "id": item.get("id"),
        "termo_busca": termo,
        "repositorio": repositorio,
        "numero": item.get("number"),
        "titulo": item.get("title") or "",
        "corpo": item.get("body") or "",
        "autor": (item.get("user") or {}).get("login", "desconhecido"),
        "estado": item.get("state", ""),
        "criado_em": item.get("created_at", ""),
        "qtd_comentarios": item.get("comments", 0),
        "labels": [rotulo.get("name", "") for rotulo in item.get("labels", [])],
        "url": item.get("html_url", ""),
    }


def buscar_termo(
    termo: str,
    token: str | None,
    paginas: int = config.PAGINAS_POR_TERMO,
) -> list[dict[str, Any]]:
    """Executa a busca de um termo, percorrendo as páginas de resultado."""
    coletadas: list[dict[str, Any]] = []
    pausa = config.PAUSA_COM_TOKEN if token else config.PAUSA_SEM_TOKEN

    for pagina in range(1, paginas + 1):
        parametros = {
            # in:title,body limita a busca ao texto da issue.
            # type:issue descarta pull requests, que tem formato diferente.
            "q": f'"{termo}" in:title,body type:issue',
            "per_page": config.RESULTADOS_POR_PAGINA,
            "page": pagina,
            "sort": "comments",  # issues discutidas rendem mais texto
            "order": "desc",
        }

        try:
            resposta = requests.get(
                config.API_BUSCA_ISSUES,
                params=parametros,
                headers=_cabecalhos(token),
                timeout=30,
            )
        except requests.RequestException as erro:
            print(f"    [rede] falhou em '{termo}' página {pagina}: {erro}")
            break

        if resposta.status_code == 403:
            print(f"    [rate limit] GitHub barrou. Aguardando 60s...")
            time.sleep(60)
            continue

        if resposta.status_code != 200:
            print(f"    [http {resposta.status_code}] '{termo}' página {pagina}")
            break

        itens = resposta.json().get("items", [])
        if not itens:
            break

        coletadas.extend(_normalizar(item, termo) for item in itens)
        print(f"    página {pagina}: +{len(itens)} issues")
        time.sleep(pausa)

    return coletadas


def coletar(forcar: bool = False) -> list[dict[str, Any]]:
    """
    Coleta o corpus completo.

    Se o arquivo de cache já existe e `forcar` e False, le do disco.
    """
    if config.ARQ_ISSUES.exists() and not forcar:
        with config.ARQ_ISSUES.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        print(f"[cache] {len(dados)} issues lidas de {config.ARQ_ISSUES.name}")
        return dados

    token = _descobrir_token()
    print(f"[auth] token do GitHub: {'encontrado' if token else 'ausente (modo lento)'}")

    todas: list[dict[str, Any]] = []
    for termo in config.TERMOS_BUSCA:
        print(f"  buscando: {termo!r}")
        todas.extend(buscar_termo(termo, token))

    # A mesma issue pode aparecer em dois termos diferentes.
    # O dicionário por id elimina a duplicata mantendo a primeira ocorrência.
    unicas = {issue["id"]: issue for issue in todas if issue.get("id")}
    resultado = list(unicas.values())

    with config.ARQ_ISSUES.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)

    print(f"[ok] {len(todas)} coletadas, {len(resultado)} únicas")
    print(f"[ok] salvo em {config.ARQ_ISSUES}")
    return resultado


def resumo(issues: list[dict[str, Any]]) -> None:
    """Imprime um panorama rápido do que foi coletado."""
    if not issues:
        print("Nenhuma issue coletada ainda. Rode a opção de coleta primeiro.")
        return

    caracteres = sum(len(i["titulo"]) + len(i["corpo"]) for i in issues)
    repos = {i["repositorio"] for i in issues}
    datas = sorted(i["criado_em"] for i in issues if i["criado_em"])

    print(f"  issues ......... {len(issues)}")
    print(f"  repositórios ... {len(repos)}")
    print(f"  caracteres ..... {caracteres:,}".replace(",", "."))
    print(f"  período ........ {datas[0][:10]} até {datas[-1][:10]}")
    print("  top repositórios:")
    from collections import Counter

    for repo, qtd in Counter(i["repositorio"] for i in issues).most_common(5):
        print(f"    {qtd:>4}  {repo}")


if __name__ == "__main__":
    resumo(coletar())
