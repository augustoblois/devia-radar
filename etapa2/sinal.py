"""
Etapa 2 - ANÁLISE DO SINAL DE ÁUDIO COM librosa.

Para o computador, som é uma lista de números. O microfone mede a pressão
do ar milhares de vezes por segundo e guarda cada medida. Este módulo
transforma essa lista em características que dizem COMO a pessoa fala:
quanto ela pausa, com que energia, em que altura de voz.

Vocabulário mínimo (definido aqui porque aparece o tempo todo no código):

- Amostra (sample): uma medida da pressão do ar, um número entre -1 e +1.
- Taxa de amostragem (sample rate): quantas amostras por segundo. Aqui,
  16.000. Pelo teorema de Nyquist, uma taxa T representa frequências até
  T/2, então 16 kHz cobre até 8 kHz, onde está quase toda a voz humana.
- Quadro (frame): um pedaço curto do sinal, aqui 25 ms. A voz muda o tempo
  todo, mas dentro de 25 ms ela parece parada, e dá para medir.
- Salto (hop): a distância entre o início de um quadro e o do próximo, 10 ms.
  Os quadros se sobrepõem, então nenhum som cai "entre" dois quadros.
- RMS (root mean square): a energia do quadro, raiz da média dos quadrados
  das amostras. Em decibéis (dB), 20 x log10(RMS). 0 dB é o máximo possível,
  e cada -6 dB é metade da amplitude.
- FFT (transformada rápida de Fourier): decompõe um quadro nas frequências
  que o formam. Aplicada quadro a quadro chama-se STFT, e o resultado
  desenhado é o espectrograma: tempo no eixo x, frequência no eixo y.
- Escala Mel: escala de frequência que imita o ouvido. A gente distingue
  bem 100 Hz de 200 Hz, mas mal 7.100 Hz de 7.200 Hz. A fórmula é
  mel = 2595 x log10(1 + f / 700).
- MFCC (Mel-frequency cepstral coefficients): 13 números por quadro que
  resumem o formato do espectro Mel. São a "impressão digital" do timbre, e
  foram por décadas a entrada padrão do reconhecimento de fala.
- ZCR (zero crossing rate): quantas vezes o sinal troca de sinal no quadro.
  Vogal cruza pouco (som periódico), "s" e "f" cruzam muito (som de ruído).
- Centróide espectral: o "centro de gravidade" das frequências. Voz abafada
  tem centróide baixo, voz brilhante ou chiada tem centróide alto.
- F0 (frequência fundamental): a altura da voz, o que o ouvido chama de
  grave ou agudo. Estimada pelo algoritmo YIN, que procura de quanto em
  quanto tempo o sinal se repete.

Onde aprender: https://librosa.org/doc/latest/tutorial.html
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import librosa
import numpy as np

import config

QUADROS_POR_SEGUNDO = config.TAXA_AMOSTRAGEM // config.SALTO  # 100


def _decodificar_inteiro(caminho: Path) -> np.ndarray:
    """
    Decodifica o arquivo inteiro com o PyAV, a biblioteca que o Whisper usa.

    Ele converte para 16 kHz mono em pedaços, enquanto lê. Um episódio de
    1 hora ocupa ~230 MB de memória assim. O librosa.load leria o MP3
    inteiro na taxa original (44.100 Hz, estéreo) antes de reamostrar, e
    passaria de 1 GB.
    """
    from faster_whisper.audio import decode_audio

    return decode_audio(str(caminho), sampling_rate=config.TAXA_AMOSTRAGEM)


def carregar(caminho: Path, inicio: float = 0.0, duracao: float | None = None) -> np.ndarray:
    """
    Lê o áudio como um vetor de amostras a 16 kHz, mono.

    Episódio inteiro: decodificação em pedaços (ver acima). Trecho curto: o
    librosa pula direto para o segundo pedido e reamostra só o trecho, o que
    deixa a demo instantânea. Se o librosa não abrir o formato (M4A), o
    trecho sai do arquivo decodificado inteiro.
    """
    if duracao is None and inicio == 0:
        return _decodificar_inteiro(caminho)
    try:
        sinal, _ = librosa.load(
            caminho, sr=config.TAXA_AMOSTRAGEM, mono=True, offset=inicio, duration=duracao
        )
        return sinal
    except (RuntimeError, OSError):
        sinal = _decodificar_inteiro(caminho)
        primeira = int(inicio * config.TAXA_AMOSTRAGEM)
        ultima = None if duracao is None else primeira + int(duracao * config.TAXA_AMOSTRAGEM)
        return sinal[primeira:ultima]


def _pausas(sinal: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Separa som de silêncio.

    `librosa.effects.split` marca como silêncio todo quadro mais de
    TOP_DB_SILENCIO decibéis abaixo do quadro mais alto. Devolve os
    intervalos COM som, e a pausa é o buraco entre dois intervalos.
    """
    intervalos = librosa.effects.split(
        sinal,
        top_db=config.TOP_DB_SILENCIO,
        frame_length=config.JANELA,
        hop_length=config.SALTO,
    )
    pausas = []
    for (_, fim_anterior), (inicio_seguinte, _) in itertools.pairwise(intervalos):
        duracao = (inicio_seguinte - fim_anterior) / config.TAXA_AMOSTRAGEM
        if duracao >= config.PAUSA_LONGA:
            pausas.append(
                (fim_anterior / config.TAXA_AMOSTRAGEM, inicio_seguinte / config.TAXA_AMOSTRAGEM)
            )
    return intervalos, pausas


def quadros(sinal: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula as características quadro a quadro (100 quadros por segundo)."""
    rms = librosa.feature.rms(y=sinal, frame_length=config.JANELA, hop_length=config.SALTO)[0]
    energia_db = librosa.amplitude_to_db(rms, ref=1.0)  # dBFS: 0 é o teto
    zcr = librosa.feature.zero_crossing_rate(
        sinal, frame_length=config.JANELA, hop_length=config.SALTO
    )[0]
    centroide = librosa.feature.spectral_centroid(
        y=sinal,
        sr=config.TAXA_AMOSTRAGEM,
        n_fft=config.N_FFT,
        hop_length=config.SALTO,
        win_length=config.JANELA,
    )[0]
    mel = librosa.feature.melspectrogram(
        y=sinal,
        sr=config.TAXA_AMOSTRAGEM,
        n_fft=config.N_FFT,
        hop_length=config.SALTO,
        win_length=config.JANELA,
        n_mels=config.N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=config.N_MFCC)
    f0 = librosa.yin(
        sinal,
        fmin=config.F0_MIN,
        fmax=config.F0_MAX,
        sr=config.TAXA_AMOSTRAGEM,
        frame_length=config.JANELA_F0,
        hop_length=config.SALTO,
    )

    # O YIN devolve uma altura para TODO quadro, até para o silêncio e para
    # o "s" de uma palavra, onde não existe altura nenhuma. Quadro sonoro é
    # o que tem energia (acima do limiar de silêncio) e cruza o zero poucas
    # vezes (som periódico, de vogal). Só esses entram na estatística de F0.
    # O YIN também "gruda" nos limites da faixa quando não acha período, e
    # esses valores de borda saem fora.
    sonoro = (
        (energia_db > energia_db.max() - config.TOP_DB_SILENCIO)
        & (zcr < 0.15)
        & (f0 > config.F0_MIN * 1.02)
        & (f0 < config.F0_MAX * 0.98)
    )
    return {
        "energia_db": energia_db,
        "zcr": zcr,
        "centroide": centroide,
        "mel_db": mel_db,
        "mfcc": mfcc,
        "f0": f0,
        "sonoro": sonoro,
    }


def resumir(sinal: np.ndarray, q: dict[str, np.ndarray]) -> dict[str, Any]:
    """Transforma milhares de quadros em um punhado de números por episódio."""
    duracao = len(sinal) / config.TAXA_AMOSTRAGEM
    intervalos, pausas = _pausas(sinal)
    com_som = sum(fim - inicio for inicio, fim in intervalos) / config.TAXA_AMOSTRAGEM
    f0_voz = q["f0"][q["sonoro"]]
    energia_voz = q["energia_db"][q["sonoro"]]

    return {
        "duracao_s": round(duracao, 1),
        "silencio_pct": round((1 - com_som / max(duracao, 1e-9)) * 100, 1),
        "pausas_longas": len(pausas),
        "pausas_por_min": round(len(pausas) / max(duracao / 60, 1e-9), 2),
        "pausa_media_s": round(float(np.mean([f - i for i, f in pausas])) if pausas else 0.0, 2),
        "energia_db_media": round(float(np.mean(energia_voz)), 2) if energia_voz.size else None,
        "energia_db_desvio": round(float(np.std(energia_voz)), 2) if energia_voz.size else None,
        "f0_mediana_hz": round(float(np.median(f0_voz)), 1) if f0_voz.size else None,
        "f0_desvio_hz": round(float(np.std(f0_voz)), 1) if f0_voz.size else None,
        "quadros_sonoros_pct": round(float(q["sonoro"].mean()) * 100, 1),
        "centroide_medio_hz": round(float(np.mean(q["centroide"])), 1),
        "zcr_media": round(float(np.mean(q["zcr"])), 4),
        "mfcc_media": [round(float(v), 2) for v in q["mfcc"].mean(axis=1)],
    }


def curvas_por_segundo(q: dict[str, np.ndarray]) -> dict[str, list[float | None]]:
    """
    Média de energia e mediana de F0 a cada segundo.

    Guardar os 100 quadros por segundo de 14 episódios daria dezenas de
    megabytes. Um valor por segundo basta para desenhar a linha do tempo e
    para cruzar a voz com o sentimento de cada frase transcrita.
    """
    total = len(q["energia_db"]) // QUADROS_POR_SEGUNDO
    energia, f0 = [], []
    for segundo in range(total):
        fatia = slice(segundo * QUADROS_POR_SEGUNDO, (segundo + 1) * QUADROS_POR_SEGUNDO)
        energia.append(round(float(np.mean(q["energia_db"][fatia])), 1))
        voz = q["f0"][fatia][q["sonoro"][fatia]]
        f0.append(round(float(np.median(voz)), 1) if voz.size >= 10 else None)
    return {"energia_db": energia, "f0_hz": f0}


def analisar(episodios: list[dict[str, Any]], forcar: bool = False) -> dict[str, Any]:
    """
    Extrai as características de todos os episódios com áudio no disco.

    Episódio já analisado sai do cache (acustica.json), então a opção roda
    na hora durante a apresentação. O resultado é gravado depois de cada
    episódio, e interromper no meio não perde o que já foi feito.
    """
    resultado = {} if forcar else carregar_analise()
    # Episódio que saiu da amostra sai também do cache.
    atuais = {episodio["id"] for episodio in episodios}
    for chave in ("episodios", "curvas"):
        resultado[chave] = {i: v for i, v in resultado.get(chave, {}).items() if i in atuais}
    for indice, episodio in enumerate(episodios, start=1):
        caminho = config.DIR_AUDIO / episodio["arquivo"]
        if episodio["id"] not in resultado["episodios"]:
            if not caminho.exists():
                print(f"  [{indice:>2}] sem áudio no disco, pulado: {episodio['titulo'][:50]}")
                continue
            sinal = carregar(caminho)
            q = quadros(sinal)
            resultado["episodios"][episodio["id"]] = resumir(sinal, q)
            resultado["curvas"][episodio["id"]] = curvas_por_segundo(q)
            with config.ARQ_ACUSTICA.open("w", encoding="utf-8") as arquivo:
                json.dump(resultado, arquivo, ensure_ascii=False)
        resumo = resultado["episodios"][episodio["id"]]
        print(
            f"  [{indice:>2}/{len(episodios)}] {resumo['duracao_s'] / 60:>4.0f} min  "
            f"pausas/min {resumo['pausas_por_min']:>5.1f}  "
            f"F0 {resumo['f0_mediana_hz']:>5} Hz  {episodio['podcast'][:34]}"
        )
    print(f"  salvo em {config.ARQ_ACUSTICA.name}")
    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_ACUSTICA.exists():
        return {}
    with config.ARQ_ACUSTICA.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def salvar_wav(sinal: np.ndarray, nome: str) -> Path:
    """Grava um trecho em WAV de 16 bits, o formato que o Windows toca sozinho."""
    destino = config.DIR_TRECHOS / nome
    inteiros = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(destino), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)  # 2 bytes = 16 bits por amostra
        arquivo.setframerate(config.TAXA_AMOSTRAGEM)
        arquivo.writeframes(inteiros.tobytes())
    return destino


def tocar(caminho: Path) -> None:
    """
    Toca um WAV pelas caixas de som, sem biblioteca extra.

    `winsound` vem com o Python no Windows. SND_ASYNC devolve o controle na
    hora, então dá para falar por cima enquanto o trecho toca.
    """
    if sys.platform != "win32":
        print(f"  (reprodução automática só no Windows; arquivo em {caminho})")
        return
    import winsound

    winsound.PlaySound(str(caminho), winsound.SND_FILENAME | winsound.SND_ASYNC)
    input("  tocando... enter para parar ")
    winsound.PlaySound(None, 0)


def abrir(caminho: Path) -> None:
    """Abre um arquivo no programa padrão do sistema (o visualizador de imagem)."""
    if sys.platform == "win32":
        os.startfile(caminho)
    else:
        print(f"  arquivo em {caminho}")


def trecho_demo(episodios: list[dict[str, Any]]) -> tuple[dict[str, Any], float] | None:
    """Escolhe o episódio e o segundo de início do trecho usado nas demos."""
    candidatos = [e for e in episodios if (config.DIR_AUDIO / e["arquivo"]).exists()]
    if not candidatos:
        return None
    for episodio in candidatos:
        if episodio["id"] == config.TRECHO_DEMO_EPISODIO:
            return episodio, config.TRECHO_DEMO_INICIO
    return candidatos[0], config.TRECHO_DEMO_INICIO


def preparar_trecho(episodios: list[dict[str, Any]]):
    """Carrega o trecho da demo e calcula tudo que o gráfico 02 desenha."""
    escolha = trecho_demo(episodios)
    if escolha is None:
        return None
    episodio, inicio = escolha
    sinal = carregar(config.DIR_AUDIO / episodio["arquivo"], inicio, config.TRECHO_DEMO)
    _, pausas = _pausas(sinal)
    return sinal, quadros(sinal), pausas, episodio, inicio


def demonstrar(episodios: list[dict[str, Any]]) -> None:
    """
    Mostra na tela o caminho do som até os números.

    É a forma mais direta de explicar na apresentação o que a librosa faz:
    o mesmo trecho de 20 segundos, primeiro como lista de amostras, depois
    como energia, pausas, altura da voz e MFCC.
    """
    trecho = preparar_trecho(episodios)
    if trecho is None:
        print("  Nenhum áudio no disco. Rode a coleta primeiro (opção 1).")
        return
    sinal, q, pausas, episodio, inicio = trecho
    resumo = resumir(sinal, q)

    minuto, segundo = divmod(int(inicio), 60)
    print(f"\n  TRECHO: {episodio['podcast']}, a partir de {minuto}:{segundo:02d}")
    print(f"          {episodio['titulo'][:70]}")

    print("\n  1. O SOM COMO LISTA DE NÚMEROS")
    print(f"     taxa de amostragem ...... {config.TAXA_AMOSTRAGEM:,} amostras por segundo".replace(",", "."))
    print(f"     amostras no trecho ...... {len(sinal):,}".replace(",", "."))
    meio = len(sinal) // 2
    # float() antes do round: o float32 da librosa, convertido direto,
    # aparece como -0.032099999487400055 em vez de -0.0321.
    print(f"     8 amostras do meio ...... {[round(float(x), 4) for x in sinal[meio:meio + 8]]}")
    print(f"     menor e maior valor ..... {sinal.min():+.3f} e {sinal.max():+.3f}")

    print("\n  2. QUADROS DE 25 ms, A CADA 10 ms")
    print(f"     quadros no trecho ....... {q['energia_db'].size} (100 por segundo)")
    print(f"     energia média da voz .... {resumo['energia_db_media']} dB")
    print(f"     silêncio ................ {resumo['silencio_pct']}% do trecho")
    for inicio_p, fim_p in pausas[:5]:
        print(f"       pausa de {fim_p - inicio_p:.1f} s em {inicio_p:.1f} s")

    print("\n  3. FREQUÊNCIA (FFT, MEL E MFCC)")
    print(f"     centróide espectral ..... {resumo['centroide_medio_hz']:.0f} Hz")
    print(f"     taxa de cruzamento zero . {resumo['zcr_media']:.3f}")
    print(f"     espectrograma Mel ....... {q['mel_db'].shape[0]} bandas x {q['mel_db'].shape[1]} quadros")
    print(f"     MFCC .................... {q['mfcc'].shape[0]} coeficientes x {q['mfcc'].shape[1]} quadros")
    print(f"     MFCC médio (5 primeiros)  {resumo['mfcc_media'][:5]}")

    print("\n  4. ALTURA DA VOZ (YIN)")
    print(f"     F0 mediana .............. {resumo['f0_mediana_hz']} Hz")
    print(f"     variação (desvio) ....... {resumo['f0_desvio_hz']} Hz")
    print(f"     quadros com voz ......... {resumo['quadros_sonoros_pct']}%")

    import visualizacao

    figura = visualizacao.figura_trecho(sinal, q, pausas, episodio, inicio)
    abrir(figura)

    wav = salvar_wav(sinal, "trecho_demo.wav")
    if input("\n  ouvir o trecho? [s/n] ").strip().lower() == "s":
        tocar(wav)


if __name__ == "__main__":
    import coleta

    demonstrar(coleta.selecionar())
