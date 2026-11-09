"""
Etapa 3 - LEITURA: o que a tela mostra e o que a palestra diz.

Três fontes de texto para cada palestra, todas com marca de tempo:

1. A TELA, lida por OCR (optical character recognition, reconhecimento
   óptico de caracteres) em cada slide que o quadros.py guardou. O RapidOCR
   usa duas redes neurais: a primeira acha ONDE há texto na imagem
   (detecção, devolve caixas) e a segunda lê cada caixa (reconhecimento,
   devolve o texto e uma confiança de 0 a 1).
2. A FALA, transcrita pelo Whisper com a lista de vocabulário, como na
   etapa 2. O áudio sai do próprio arquivo de vídeo.
3. A LEGENDA OFICIAL da FOSDEM (arquivo .vtt). Ela também sai de um
   reconhecimento automático de fala, mas sem lista de vocabulário.
   Comparar as duas mede, em dados novos, o efeito da lista.

Tipo de tela, pelo texto que o OCR leu (detalhes em tipo_de_tela):
- visual: slide com menos de MIN_PALAVRAS_TELA palavras (foto, diagrama);
- texto: slide de frases, tópicos e tabelas;
- código: captura de editor ou terminal;
- interface: captura de site ou aplicativo (GitHub, n8n, navegador).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import coleta
import config

# ---------------------------------------------------------------------------
# Tela (OCR)
# ---------------------------------------------------------------------------

# Linha com cara de código. Sem IGNORECASE: "Import" no título de um slide
# não conta. A primeira versão só tinha as três primeiras linhas e achou 3
# telas de código em 320; o editor da palestra de Swift mostrava YAML,
# número de linha e nome de arquivo, que ela não conhecia (diário de bordo).
RE_LINHA_CODIGO = re.compile(
    r"[{};]|\(\)|->|=>|::|==|!=|</|/>|"                     # símbolo de linguagem
    r"\b\w+\.\w+\(|\b[a-z]+_[a-z_]+\b|"                     # metodo(, nome_com_sublinhado
    r"\b(?:def|import|return|func|const|let|var|fn|struct|async|await|"
    r"sudo|npm|pip|kubectl|git|docker|curl)\s|"             # palavra reservada, comando
    r"^\s*[$>#%❯]\s|"                                       # prompt de terminal
    r"^\s*-?\s*\"?[a-z_][\w.-]*\"?:(?:\s|$)|"               # chave de YAML ou JSON
    r"^\s*\d{1,4}\s*$|"                                     # número de linha do editor
    r"\b[\w-]+\.(?:py|js|ts|go|rs|swift|java|rb|c|h|cpp|yaml|yml|json|toml|sh|md)\b"  # arquivo
)
RE_CURSOR_ESCRITA = re.compile(config.CURSOR_ESCRITA)

_OCR = None  # as duas redes do OCR carregam uma vez só


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


def ler_imagem(imagem: Path | np.ndarray) -> list[tuple[str, float]]:
    """
    Lê o texto de uma imagem: lista de (linha, confiança), de cima para baixo.

    Linha com confiança abaixo de CONFIANCA_OCR é descartada pelo próprio
    RapidOCR (parâmetro text_score). Nesta versão a confiança vem como
    texto ("0.93"), então ela é convertida para número aqui.
    """
    origem = str(imagem) if isinstance(imagem, Path) else imagem
    resultado, _ = _ocr()(origem, text_score=config.CONFIANCA_OCR)
    return [(texto.strip(), round(float(confianca), 3)) for _, texto, confianca in resultado or [] if texto.strip()]


def tipo_de_tela(linhas: list[tuple[str, float]], brilho: float) -> tuple[str, float]:
    """
    Classifica a tela. Devolve o tipo e a fração de linhas com cara de código.

    1. Poucas palavras: slide "visual" (foto, diagrama, meme).
    2. Muitas linhas de texto: captura de ferramenta. Slide é feito para ser
       lido do fundo da sala e tem poucas linhas de letra grande; editor,
       terminal e site têm dezenas de linhas miúdas. Editor e terminal em
       tema escuro costumam usar letra maior, por isso o limite é menor
       quando a tela é escura. Dentro da captura:
       - fundo escuro ou linhas com cara de código: "código";
       - o resto: "interface" (GitHub, n8n, navegador).
    3. Sobrou: slide de "texto". Slide com um bloco de código ou com um
       print dentro continua slide, porque foi preparado para a palestra.
    """
    palavras = sum(len(texto.split()) for texto, _ in linhas)
    codigo = sum(bool(RE_LINHA_CODIGO.search(texto)) for texto, _ in linhas)
    fracao = round(codigo / len(linhas), 3) if linhas else 0.0
    escura = brilho < config.LIMIAR_ESCURO
    if palavras < config.MIN_PALAVRAS_TELA:
        return "visual", fracao
    if len(linhas) >= config.LINHAS_CAPTURA or (escura and len(linhas) >= config.LINHAS_CAPTURA_ESCURA):
        return ("código" if escura or fracao >= config.FRACAO_CODIGO else "interface"), fracao
    return "texto", fracao


def ferramentas_na_tela(texto: str) -> list[str]:
    """
    Ferramentas de IA escritas na tela.

    O OCR às vezes cola palavras ("usingClaudeCode"), então aqui o texto e
    cada termo do dicionário perdem os espaços e viram minúsculas antes da
    busca. Na fala a regex continua exigindo a palavra inteira. O Cursor
    sozinho é a exceção: na escrita, só a maiúscula separa o produto do
    cursor da tela (config.CURSOR_ESCRITA), então ele é procurado no texto
    original.
    """
    colado = re.sub(r"\s+", "", texto).lower()
    return [
        nome
        for nome, variacoes in config.FERRAMENTAS_IA.items()
        if any(v.replace(" ", "") in colado for v in variacoes)
        or (nome == "Cursor" and RE_CURSOR_ESCRITA.search(texto))
    ]


def _descrever(tela: dict[str, Any], linhas: list[tuple[str, float]]) -> dict[str, Any]:
    """Palavras, tipo e ferramentas de uma tela, a partir das linhas que o OCR leu."""
    tipo, fracao = tipo_de_tela(linhas, tela["brilho"])
    texto = " ".join(t for t, _ in linhas)
    return {
        "numero": tela["numero"],
        "linhas": linhas,
        "palavras": len(texto.split()),
        "fracao_codigo": fracao,
        "tipo": tipo,
        "ferramentas": ferramentas_na_tela(texto),
    }


def ler_telas(palestras: list[dict[str, Any]], visual: dict[str, Any], forcar: bool = False) -> dict[str, Any]:
    """
    Passa o OCR em todas as telas guardadas pelo quadros.py.

    Só o OCR, a parte lenta, sai do cache: palestra já lida não é lida de
    novo. Tipo de tela e ferramentas são recalculados a cada execução a
    partir das linhas guardadas, então mudar uma regra não exige refazer o
    OCR. O resultado é gravado depois de cada palestra.
    """
    leitura = {} if forcar else carregar_leitura()
    tipos_tela = [t for t in config.PALETA_TELA if t != "movimento"]
    for indice, palestra in enumerate(palestras, start=1):
        telas = visual.get("palestras", {}).get(palestra["id"], {}).get("telas", [])
        if not telas:
            print(f"  [{indice}/{len(palestras)}] sem telas (rode a opção 4): {palestra['titulo'][:40]}")
            continue
        guardadas = {lida["numero"]: lida["linhas"] for lida in leitura.get(palestra["id"], [])}
        # Se a extração visual mudou as telas, o OCR guardado não serve mais.
        if set(guardadas) != {tela["numero"] for tela in telas}:
            relogio = time.time()
            guardadas = {
                tela["numero"]: ler_imagem(config.DIR_TELAS / palestra["id"] / tela["arquivo"]) for tela in telas
            }
            print(f"    OCR em {time.time() - relogio:.0f} s")
        lidas = [_descrever(tela, guardadas[tela["numero"]]) for tela in telas]
        leitura[palestra["id"]] = lidas
        with config.ARQ_LEITURA.open("w", encoding="utf-8") as arquivo:
            json.dump(leitura, arquivo, ensure_ascii=False)
        contagem = "  ".join(f"{t} {sum(l['tipo'] == t for l in lidas):>2}" for t in tipos_tela)
        citadas = sorted({f for l in lidas for f in l["ferramentas"]})
        print(
            f"  [{indice}/{len(palestras)}] {len(lidas):>3} telas  {contagem}  "
            f"na tela: {', '.join(citadas) or '-'}  ({palestra['titulo'][:28]})"
        )
    print(f"  salvo em {config.ARQ_LEITURA.name}")
    return leitura


def carregar_leitura() -> dict[str, Any]:
    if not config.ARQ_LEITURA.exists():
        return {}
    with config.ARQ_LEITURA.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ---------------------------------------------------------------------------
# Fala (Whisper)
# ---------------------------------------------------------------------------

_MODELO = None
_DISPOSITIVO = "cpu"


def _bibliotecas_cuda() -> bool:
    """
    Põe as DLLs da NVIDIA no caminho de busca do Windows (mesma função da
    etapa 2). Os pacotes nvidia-cublas-cu12 e nvidia-cudnn-cu12 do pip
    trazem as DLLs numa pasta que o Windows não procura sozinho.
    """
    if sys.platform != "win32":
        return True
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return False
    for pacote in (nvidia.cublas, nvidia.cudnn):
        pasta = Path(next(iter(pacote.__path__))) / "bin"
        if pasta.exists():
            os.add_dll_directory(str(pasta))
            os.environ["PATH"] = str(pasta) + os.pathsep + os.environ["PATH"]
    return True


def _modelo_gpu():
    """O modelo na GPU, ou None se a GPU não funcionar nesta máquina."""
    from faster_whisper import WhisperModel

    if config.DISPOSITIVO == "cpu" or not _bibliotecas_cuda():
        return None
    try:
        modelo = WhisperModel(config.MODELO_WHISPER, device="cuda", compute_type=config.TIPO_COMPUTO_GPU)
        # A falta de DLL só aparece na primeira transcrição: o teste é aqui,
        # com 1 segundo de silêncio, antes de a demo começar.
        segmentos, _ = modelo.transcribe(np.zeros(config.TAXA_AMOSTRAGEM, dtype=np.float32), language="en")
        list(segmentos)
        return modelo
    except (RuntimeError, ValueError) as erro:
        print(f"  [GPU indisponível: {str(erro)[:80]}]")
        return None


def carregar_modelo():
    global _MODELO, _DISPOSITIVO
    if _MODELO is None:
        from faster_whisper import WhisperModel

        print(f"  carregando o Whisper '{config.MODELO_WHISPER}'...")
        _MODELO = _modelo_gpu()
        _DISPOSITIVO = "cuda" if _MODELO is not None else "cpu"
        if _MODELO is None:
            _MODELO = WhisperModel(
                config.MODELO_WHISPER,
                device="cpu",
                compute_type=config.TIPO_COMPUTO_CPU,
                cpu_threads=config.THREADS,
            )
        print(f"  rodando na {'GPU' if _DISPOSITIVO == 'cuda' else 'CPU'}")
    return _MODELO


def audio(palestra: dict[str, Any]) -> np.ndarray:
    """O áudio do vídeo como vetor de amostras a 16 kHz, mono (o formato do Whisper)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(coleta.caminho_video(palestra)), sampling_rate=config.TAXA_AMOSTRAGEM)


def transcrever(sinal: np.ndarray, vocabulario: str | None = config.PROMPT_VOCABULARIO) -> list[dict[str, Any]]:
    """Transcreve um vetor de amostras e devolve os segmentos com tempo (igual à etapa 2)."""
    segmentos, _ = carregar_modelo().transcribe(
        sinal,
        language="en",
        beam_size=config.FEIXE,
        vad_filter=True,
        hotwords=vocabulario,
    )
    return [
        {
            "inicio": round(s.start, 2),
            "fim": round(s.end, 2),
            "texto": s.text.strip(),
            "logprob": round(s.avg_logprob, 3),
        }
        for s in segmentos
    ]


def tocar(sinal: np.ndarray, nome: str = "trecho.wav") -> None:
    """
    Grava o trecho em WAV de 16 bits e toca pelas caixas de som.

    `winsound` vem com o Python no Windows. SND_ASYNC devolve o controle na
    hora, então dá para falar por cima enquanto o trecho toca.
    """
    import wave

    destino = config.DIR_TRECHOS / nome
    with wave.open(str(destino), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)  # 2 bytes = 16 bits por amostra
        arquivo.setframerate(config.TAXA_AMOSTRAGEM)
        arquivo.writeframes((np.clip(sinal, -1, 1) * 32767).astype(np.int16).tobytes())
    if sys.platform != "win32":
        print(f"  (reprodução automática só no Windows; arquivo em {destino})")
        return
    import winsound

    winsound.PlaySound(str(destino), winsound.SND_FILENAME | winsound.SND_ASYNC)
    input("  tocando... enter para parar ")
    winsound.PlaySound(None, 0)


def _arquivo_transcricao(palestra: dict[str, Any]) -> Path:
    return config.DIR_TRANSCRICOES / f"{palestra['id']}.json"


def transcrever_todas(palestras: list[dict[str, Any]], forcar: bool = False) -> None:
    """Transcreve a fala de todas as palestras com vídeo, pulando as já feitas."""
    for indice, palestra in enumerate(palestras, start=1):
        destino = _arquivo_transcricao(palestra)
        if destino.exists() and not forcar:
            print(f"  [{indice}/{len(palestras)}] cache   {palestra['titulo'][:50]}")
            continue
        if not coleta.video_disponivel(palestra):
            print(f"  [{indice}/{len(palestras)}] sem vídeo, pulada")
            continue
        sinal = audio(palestra)
        relogio = time.time()
        segmentos = transcrever(sinal)
        gasto = time.time() - relogio
        duracao = len(sinal) / config.TAXA_AMOSTRAGEM
        with destino.open("w", encoding="utf-8") as arquivo:
            json.dump(
                {
                    "id": palestra["id"],
                    "modelo": config.MODELO_WHISPER,
                    "dispositivo": _DISPOSITIVO,
                    "vocabulario": config.PROMPT_VOCABULARIO,
                    "duracao_s": round(duracao, 1),
                    "tempo_transcricao_s": round(gasto, 1),
                    "segmentos": segmentos,
                },
                arquivo,
                ensure_ascii=False,
                indent=1,
            )
        palavras = sum(len(s["texto"].split()) for s in segmentos)
        print(
            f"  [{indice}/{len(palestras)}] {duracao / 60:>3.0f} min em {gasto:>4.0f} s "
            f"({duracao / gasto:>4.1f}x tempo real)  {palavras:>5} palavras  {palestra['titulo'][:34]}"
        )


def carregar_transcricoes(palestras: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    transcricoes = {}
    for palestra in palestras:
        destino = _arquivo_transcricao(palestra)
        if destino.exists():
            with destino.open(encoding="utf-8") as arquivo:
                transcricoes[palestra["id"]] = json.load(arquivo)
    return transcricoes


# ---------------------------------------------------------------------------
# Legenda oficial (WebVTT)
# ---------------------------------------------------------------------------

# Marca de tempo do VTT: "01:02:03.456" ou só "02:03.456" (sem a hora).
RE_TEMPO_VTT = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})")


def _segundos(marca: str) -> float:
    horas, minutos, segundos, milesimos = RE_TEMPO_VTT.search(marca).groups()
    return int(horas or 0) * 3600 + int(minutos) * 60 + int(segundos) + int(milesimos) / 1000


def ler_legenda(palestra: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Lê a legenda oficial. O WebVTT é texto puro em blocos separados por
    linha em branco: uma linha "início --> fim" e, embaixo, o que foi dito.
    """
    caminho = coleta.caminho_legenda(palestra)
    if not caminho.exists():
        return []
    legenda = []
    for bloco in caminho.read_text(encoding="utf-8").replace("\r", "").split("\n\n"):
        linhas = bloco.strip().split("\n")
        for posicao, linha in enumerate(linhas):
            if "-->" in linha:
                inicio, fim = linha.split("-->")
                texto = " ".join(linhas[posicao + 1:]).strip()
                if texto:
                    legenda.append({"inicio": _segundos(inicio), "fim": _segundos(fim), "texto": texto})
                break
    return legenda


# ---------------------------------------------------------------------------
# Demonstração ao vivo
# ---------------------------------------------------------------------------


def _destacar(texto: str, padrao: re.Pattern[str]) -> str:
    return padrao.sub(lambda m: f"[{m.group(0).upper()}]", texto)


def demonstrar(palestras: list[dict[str, Any]], visual: dict[str, Any]) -> None:
    """
    A mesma palestra lida de três jeitos, no mesmo instante: o OCR lê o
    slide na tela, o Whisper transcreve a fala e a legenda oficial mostra o
    que a FOSDEM publicou para aqueles segundos.
    """
    import conteudo
    import quadros

    palestra = quadros.palestra_demo(palestras)
    if palestra is None:
        print("  Nenhum vídeo no disco. Rode a coleta primeiro (opção 1).")
        return
    inicio = float(config.DEMO_SEGUNDO)
    fim = inicio + config.TRECHO_DEMO
    minuto, segundo = divmod(int(inicio), 60)
    print(f"\n  PALESTRA: {palestra['titulo'][:66]}")
    print(f"  TRECHO:   {config.TRECHO_DEMO} s a partir de {minuto}:{segundo:02d}")

    print("\n  1. A TELA (OCR)")
    relogio = time.time()
    slide = quadros.recortar(quadros.quadro_em(coleta.caminho_video(palestra), inicio), config.REGIAO_SLIDE)
    linhas = ler_imagem(slide)
    print(f"     {len(linhas)} linhas lidas em {time.time() - relogio:.1f} s")
    for texto, confianca in linhas[:12]:
        print(f"     {confianca:.2f}  {_destacar(texto, conteudo.RE_QUALQUER_FERRAMENTA)[:80]}")
    brilho = float(quadros.miniatura(slide, config.TAMANHO_SLIDE)[quadros.mascara_slide()].mean())
    tipo, fracao = tipo_de_tela(linhas, brilho)
    print(f"     tipo de tela: {tipo} ({fracao:.0%} das linhas com cara de código)")
    print(f"     ferramentas na tela: {ferramentas_na_tela(' '.join(t for t, _ in linhas)) or 'nenhuma'}")

    print("\n  2. A FALA (Whisper com a lista de vocabulário)")
    sinal = audio(palestra)[int(inicio * config.TAXA_AMOSTRAGEM):int(fim * config.TAXA_AMOSTRAGEM)]
    carregar_modelo()
    relogio = time.time()
    segmentos = transcrever(sinal)
    print(f"     {config.TRECHO_DEMO} s de áudio transcritos em {time.time() - relogio:.1f} s")
    for s in segmentos:
        print(f"     [{inicio + s['inicio']:>6.1f}s] {_destacar(s['texto'], conteudo.RE_QUALQUER_FERRAMENTA)}")

    print("\n  3. A LEGENDA OFICIAL DA FOSDEM (sem lista de vocabulário)")
    for item in ler_legenda(palestra):
        if item["fim"] > inicio and item["inicio"] < fim:
            print(f"     [{item['inicio']:>6.1f}s] {_destacar(item['texto'], conteudo.RE_QUALQUER_FERRAMENTA)}")

    if input("\n  ouvir o trecho? [s/n] ").strip().lower() == "s":
        tocar(sinal)


if __name__ == "__main__":
    import quadros

    eps = coleta.selecionar()
    demonstrar(eps, quadros.carregar_analise())
