"""
Etapa 2 - TRANSCRIÇÃO DA FALA COM WHISPER.

Whisper é um modelo de reconhecimento automático de fala (ASR, automatic
speech recognition) publicado pela OpenAI em 2022 e treinado com 680 mil
horas de áudio da internet que já tinha legenda.

Como ele funciona, em três passos:
1. O áudio é cortado em janelas de 30 segundos e cada janela vira um
   espectrograma Mel de 80 bandas, a mesma ideia do sinal.py.
2. Um transformer ENCODER lê o espectrograma e o resume.
3. Um transformer DECODER escreve o texto um pedaço de palavra (token) por
   vez, como um modelo de linguagem que "ouve" o encoder. Junto com o texto,
   ele escreve marcas de tempo: em que segundo cada frase começa e termina.

Usamos o faster-whisper, uma reimplementação do mesmo modelo que roda até 4
vezes mais rápido em CPU, com pesos em 8 bits (int8).

Parâmetros e por que cada um:
- modelo "small" (244 milhões de parâmetros): o "tiny" errava nome de
  ferramenta demais, e o "large" levaria horas por episódio em CPU.
- GPU quando existe: o mesmo modelo roda 20x o tempo real na placa de
  vídeo, contra pouco mais de 1x na CPU. Sem placa NVIDIA, cai para a CPU.
- beam_size=5 (beam search): em vez de escolher a palavra mais provável a
  cada passo, mantém as 5 melhores frases parciais e escolhe no fim.
- vad_filter=True: um detector de voz (Silero VAD) remove os trechos sem
  fala ANTES do Whisper. Sem ele, o modelo inventa texto em silêncio e
  em vinheta musical, o efeito conhecido como alucinação.
- hotwords: lista de vocabulário injetada em todas as janelas de 30 s.
  O modelo passa a preferir essas grafias quando o som é ambíguo.

Onde aprender: https://github.com/openai/whisper (artigo e cartão do modelo)
e https://github.com/SYSTRAN/faster-whisper
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

import config

_MODELO = None  # o modelo ocupa ~500 MB de memória, então carrega uma vez só
_DISPOSITIVO = "cpu"


def _bibliotecas_cuda() -> bool:
    """
    Põe as DLLs da NVIDIA no caminho de busca do Windows.

    Na GPU, o CTranslate2 (o motor do faster-whisper) procura as DLLs
    cublas64_12 e cudnn64_9. Os pacotes nvidia-cublas-cu12 e
    nvidia-cudnn-cu12 do pip trazem essas DLLs, mas numa pasta que o Windows
    não procura sozinho.
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
        # A falta de DLL só aparece na primeira transcrição, então o teste
        # transcreve 1 segundo de silêncio aqui, antes de a demo começar.
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


def transcrever(sinal: np.ndarray, vocabulario: str | None = config.PROMPT_VOCABULARIO) -> list[dict[str, Any]]:
    """
    Transcreve um vetor de amostras a 16 kHz e devolve os segmentos.

    Segmento é o pedaço de fala que o Whisper fecha com marca de tempo, em
    geral de 2 a 15 segundos. Às vezes ele fecha só no fim da janela e
    devolve 30 s com várias frases, por isso o conteudo.py corta em frases.
    """
    modelo = carregar_modelo()
    segmentos, _ = modelo.transcribe(
        sinal,
        language="en",
        beam_size=config.FEIXE,
        vad_filter=True,
        hotwords=vocabulario,
    )
    # `segmentos` é um gerador: a transcrição só acontece de fato enquanto
    # este laço consome os itens.
    return [
        {
            "inicio": round(s.start, 2),
            "fim": round(s.end, 2),
            "texto": s.text.strip(),
            "logprob": round(s.avg_logprob, 3),      # confiança média do modelo
            "sem_fala": round(s.no_speech_prob, 3),  # chance de não ser fala
        }
        for s in segmentos
    ]


def _arquivo(episodio: dict[str, Any]):
    return config.DIR_TRANSCRICOES / f"{episodio['id']}.json"


def transcrever_todos(episodios: list[dict[str, Any]], forcar: bool = False) -> None:
    """Transcreve todos os episódios com áudio, pulando os já transcritos."""
    import sinal

    for indice, episodio in enumerate(episodios, start=1):
        destino = _arquivo(episodio)
        caminho = config.DIR_AUDIO / episodio["arquivo"]
        if destino.exists() and not forcar:
            print(f"  [{indice:>2}/{len(episodios)}] cache   {episodio['podcast'][:40]}")
            continue
        if not caminho.exists():
            print(f"  [{indice:>2}/{len(episodios)}] sem áudio, pulado")
            continue

        audio = sinal.carregar(caminho)
        relogio = time.time()
        segmentos = transcrever(audio)
        gasto = time.time() - relogio
        duracao = len(audio) / config.TAXA_AMOSTRAGEM

        with destino.open("w", encoding="utf-8") as arquivo:
            json.dump(
                {
                    "id": episodio["id"],
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
            f"  [{indice:>2}/{len(episodios)}] {duracao / 60:>4.0f} min em {gasto / 60:>4.1f} min "
            f"({duracao / gasto:>4.1f}x tempo real)  {palavras:>6} palavras  {episodio['podcast'][:30]}"
        )


def carregar_transcricoes(episodios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Lê do disco as transcrições já feitas, indexadas pelo id do episódio."""
    transcricoes = {}
    for episodio in episodios:
        destino = _arquivo(episodio)
        if destino.exists():
            with destino.open(encoding="utf-8") as arquivo:
                transcricoes[episodio["id"]] = json.load(arquivo)
    return transcricoes


def _destacar(texto: str, padrao: re.Pattern[str]) -> str:
    """Põe colchetes em volta do nome de ferramenta, para saltar aos olhos."""
    return padrao.sub(lambda m: f"[{m.group(0).upper()}]", texto)


def demonstrar(episodios: list[dict[str, Any]]) -> None:
    """
    Transcreve ao vivo o trecho da demo, duas vezes.

    Primeiro sem a lista de vocabulário, depois com ela. A diferença entre
    as duas versões é o diário de bordo acontecendo na frente da turma.
    """
    import conteudo
    import sinal

    escolha = sinal.trecho_demo(episodios)
    if escolha is None:
        print("  Nenhum áudio no disco. Rode a coleta primeiro (opção 1).")
        return
    episodio, inicio = escolha
    trecho = sinal.carregar(config.DIR_AUDIO / episodio["arquivo"], inicio, config.TRECHO_DEMO)
    minuto, segundo = divmod(int(inicio), 60)
    print(f"\n  TRECHO: {episodio['podcast']}, {config.TRECHO_DEMO} s a partir de {minuto}:{segundo:02d}")

    carregar_modelo()
    for rotulo, vocabulario in (("SEM lista de vocabulário", None), ("COM lista de vocabulário", config.PROMPT_VOCABULARIO)):
        relogio = time.time()
        segmentos = transcrever(trecho, vocabulario)
        gasto = time.time() - relogio
        print(f"\n  {rotulo}  ({gasto:.1f} s para {config.TRECHO_DEMO} s de áudio)")
        print("  " + "-" * 66)
        for s in segmentos:
            texto = _destacar(s["texto"], conteudo.RE_QUALQUER_FERRAMENTA)
            print(f"  [{inicio + s['inicio']:>6.1f}s] {texto}")
        achadas = conteudo.ferramentas_em(" ".join(s["texto"] for s in segmentos))
        print(f"  ferramentas reconhecidas pela regex: {achadas or 'nenhuma'}")

    if input("\n  ouvir o trecho? [s/n] ").strip().lower() == "s":
        sinal.tocar(sinal.salvar_wav(trecho, "trecho_demo.wav"))


if __name__ == "__main__":
    import coleta

    demonstrar(coleta.selecionar())
