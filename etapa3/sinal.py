"""
Etapa 3 - O SOM DA PALESTRA COM librosa.

O arquivo de vídeo da FOSDEM traz duas trilhas: a de imagem, que o
quadros.py analisa, e a de som, que este módulo analisa. Para o computador,
som é uma lista de números. O microfone mede a pressão do ar milhares de
vezes por segundo e guarda cada medida. Este módulo transforma essa lista
em características que dizem COMO o palestrante fala: quanto ele pausa,
com que energia, em que altura de voz.

Vocabulário mínimo (definido aqui porque aparece o tempo todo no código):

- Amostra (sample): uma medida da pressão do ar, um número entre -1 e +1.
- Taxa de amostragem (sample rate): quantas amostras por segundo. Aqui,
  16.000. Pelo teorema de Nyquist, uma taxa T representa frequências até
  T/2, então 16 kHz cobre até 8 kHz, onde está quase toda a voz humana.
- Quadro de áudio (frame): um pedaço curto do sinal, aqui 25 ms. A voz muda
  o tempo todo, mas dentro de 25 ms ela parece parada, e dá para medir. Não
  confundir com o quadro de vídeo do quadros.py, que é uma imagem.
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
import time
from typing import Any

import librosa
import numpy as np

import coleta
import config
import leitura

QUADROS_POR_SEGUNDO = config.TAXA_AMOSTRAGEM // config.SALTO  # 100

# ---------------------------------------------------------------------------
# Medidas quadro a quadro
# ---------------------------------------------------------------------------


def caracteristicas(sinal: np.ndarray, com_mel: bool = False) -> dict[str, np.ndarray]:
    """
    Calcula as características quadro a quadro (100 quadros por segundo).

    A STFT é feita uma vez só e serve a duas medidas: o espectrograma Mel
    (e dele o MFCC) e o centróide. O espectrograma inteiro só volta quando
    `com_mel` é True (a demo desenha ele); na palestra inteira, só o MFCC.
    """
    rms = librosa.feature.rms(y=sinal, frame_length=config.JANELA, hop_length=config.SALTO)[0]
    # dBFS: 0 é o teto. top_db=None tira o piso automático (80 dB abaixo do
    # máximo DESTE trecho), que mudaria de um bloco para o outro.
    energia_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
    zcr = librosa.feature.zero_crossing_rate(sinal, frame_length=config.JANELA, hop_length=config.SALTO)[0]
    espectro = np.abs(librosa.stft(sinal, n_fft=config.N_FFT, hop_length=config.SALTO, win_length=config.JANELA))
    centroide = librosa.feature.spectral_centroid(S=espectro, sr=config.TAXA_AMOSTRAGEM, n_fft=config.N_FFT)[0]
    mel = librosa.feature.melspectrogram(S=espectro**2, sr=config.TAXA_AMOSTRAGEM, n_mels=config.N_MELS)
    # Escala absoluta e sem piso, pelo mesmo motivo da energia: o padrão
    # corta tudo 80 dB abaixo do máximo DESTE trecho, e com a palestra em
    # blocos 39% dos quadros de MFCC mudavam (diário de bordo).
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=config.N_MFCC)
    f0 = librosa.yin(
        sinal,
        fmin=config.F0_MIN,
        fmax=config.F0_MAX,
        sr=config.TAXA_AMOSTRAGEM,
        frame_length=config.JANELA_F0,
        hop_length=config.SALTO,
    )
    medidas = {"energia_db": energia_db, "zcr": zcr, "centroide": centroide, "mfcc": mfcc, "f0": f0}
    if com_mel:
        medidas["mel_db"] = mel_db
    return medidas


def limiar_silencio(energia_db: np.ndarray) -> float:
    """
    O nível de energia (dB) abaixo do qual o quadro é silêncio.

    O critério do librosa.effects.split é ficar TOP_DB_SILENCIO dB abaixo
    do quadro mais alto, e ele supõe uma sala quieta. Com ruído de fundo
    alto, esse limiar cai dentro do próprio ruído: os quadros de uma pausa
    oscilam em volta dele, a pausa se quebra em pedaços curtos e quase
    nenhuma chega a PAUSA_LONGA (diário de bordo). Por isso o limiar fica
    também pelo menos MARGEM_RUIDO dB acima do piso de ruído, e vale o mais
    alto dos dois. O piso é o percentil 5 da energia: só os 5% de quadros
    mais quietos da palestra ficam abaixo dele.
    """
    pelo_pico = float(energia_db.max()) - config.TOP_DB_SILENCIO
    pelo_piso = float(np.percentile(energia_db, 5)) + config.MARGEM_RUIDO
    return max(pelo_pico, pelo_piso)


def com_som(energia_db: np.ndarray) -> np.ndarray:
    """Quadros com som: acima do limiar de silêncio."""
    return energia_db > limiar_silencio(energia_db)


def sonoros(medidas: dict[str, np.ndarray]) -> np.ndarray:
    """
    Marca os quadros com voz de verdade.

    O YIN devolve uma altura para TODO quadro, até para o silêncio e para
    o "s" de uma palavra, onde não existe altura nenhuma. Quadro sonoro é
    o que tem som (acima do limiar de silêncio) e cruza o zero poucas
    vezes (som periódico, de vogal). Só esses entram na estatística de F0.
    O YIN também "gruda" nos limites da faixa quando não acha período, e
    esses valores de borda saem fora.
    """
    f0 = medidas["f0"]
    return (
        com_som(medidas["energia_db"])
        & (medidas["zcr"] < config.ZCR_MAX_VOZ)
        & (f0 > config.F0_MIN * 1.02)
        & (f0 < config.F0_MAX * 0.98)
    )


def caracteristicas_palestra(sinal: np.ndarray) -> dict[str, np.ndarray]:
    """
    As características da palestra inteira, medidas um minuto por vez.

    O YIN monta uma janela de 1.024 amostras para cada quadro, e a memória
    cresce com o tamanho do áudio: 689 MB em 5 minutos, 1,4 GB em 10. A
    palestra de 50 minutos, de uma vez só, pediria perto de 7 GB (ver o
    diário de bordo). Em blocos de um minuto, o pico fica em 221 MB.

    Cada bloco devolve um quadro a mais, centrado no começo do bloco
    seguinte, e esse quadro sai: a contagem continua 100 por segundo, e o
    quadro N da palestra é o mesmo que a conta de uma vez só daria. Sobra
    de menos de 1 segundo no fim vai junto com o último bloco.
    """
    tamanho = config.BLOCO * config.TAXA_AMOSTRAGEM
    cortes = list(range(0, len(sinal), tamanho))
    if len(cortes) > 1 and len(sinal) - cortes[-1] < config.TAXA_AMOSTRAGEM:
        cortes.pop()
    cortes.append(len(sinal))
    partes = []
    for inicio, fim in itertools.pairwise(cortes):
        bloco = caracteristicas(sinal[inicio:fim])
        if fim < len(sinal):
            bloco = {chave: valores[..., :-1] for chave, valores in bloco.items()}
        partes.append(bloco)
    medidas = {chave: np.concatenate([p[chave] for p in partes], axis=-1) for chave in partes[0]}
    medidas["sonoro"] = sonoros(medidas)
    return medidas


# ---------------------------------------------------------------------------
# Resumo, pausas e curvas
# ---------------------------------------------------------------------------


def pausas(energia_db: np.ndarray) -> list[tuple[float, float]]:
    """
    Pausas longas: silêncio de PAUSA_LONGA segundos ou mais entre dois sons.

    O silêncio é o de limiar_silencio, aplicado na energia já medida
    quadro a quadro, para não ler a palestra inteira de novo. O silêncio
    antes do primeiro som e depois do último não é pausa: ninguém falou
    ainda, ou já parou.
    """
    silencio = ~com_som(energia_db)
    bordas = np.diff(np.concatenate(([0], silencio.astype(np.int8), [0])))
    inicios, fins = np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)
    minimo = config.PAUSA_LONGA * QUADROS_POR_SEGUNDO
    return [
        (round(a / QUADROS_POR_SEGUNDO, 2), round(b / QUADROS_POR_SEGUNDO, 2))
        for a, b in zip(inicios, fins)
        if b - a >= minimo and a > 0 and b < len(silencio)
    ]


def resumir(medidas: dict[str, np.ndarray], lista_pausas: list[tuple[float, float]], duracao: float) -> dict[str, Any]:
    """Transforma milhares de quadros em um punhado de números por palestra."""
    f0_voz = medidas["f0"][medidas["sonoro"]]
    energia_voz = medidas["energia_db"][medidas["sonoro"]]
    return {
        "duracao_s": round(duracao, 1),
        "silencio_pct": round(float(1 - com_som(medidas["energia_db"]).mean()) * 100, 1),
        "pausas_longas": len(lista_pausas),
        "pausas_por_min": round(len(lista_pausas) / max(duracao / 60, 1e-9), 2),
        "pausa_media_s": round(float(np.mean([f - i for i, f in lista_pausas])) if lista_pausas else 0.0, 2),
        "energia_db_media": round(float(np.mean(energia_voz)), 2) if energia_voz.size else None,
        "energia_db_desvio": round(float(np.std(energia_voz)), 2) if energia_voz.size else None,
        "f0_mediana_hz": round(float(np.median(f0_voz)), 1) if f0_voz.size else None,
        "f0_desvio_hz": round(float(np.std(f0_voz)), 1) if f0_voz.size else None,
        "quadros_sonoros_pct": round(float(medidas["sonoro"].mean()) * 100, 1),
        "centroide_medio_hz": round(float(np.mean(medidas["centroide"])), 1),
        "zcr_media": round(float(np.mean(medidas["zcr"])), 4),
        "mfcc_media": [round(float(v), 2) for v in medidas["mfcc"].mean(axis=1)],
    }


def curvas_por_segundo(medidas: dict[str, np.ndarray]) -> dict[str, list[float | None]]:
    """
    Energia e altura da voz a cada segundo.

    Guardar os 100 quadros por segundo das 8 palestras daria dezenas de
    megabytes. Um valor por segundo basta para desenhar a linha do tempo e
    para cruzar a voz com a tela e com cada frase. A energia é a média só
    dos quadros com som: num segundo que mistura fala e pausa, ela mede o
    volume da fala, e a pausa é contada à parte. Segundo com menos de
    MIN_QUADROS quadros com som (ou com voz, para a F0) fica sem valor.
    """
    som = com_som(medidas["energia_db"])
    total = len(medidas["energia_db"]) // QUADROS_POR_SEGUNDO
    energia, f0 = [], []
    for segundo in range(total):
        fatia = slice(segundo * QUADROS_POR_SEGUNDO, (segundo + 1) * QUADROS_POR_SEGUNDO)
        volume = medidas["energia_db"][fatia][som[fatia]]
        voz = medidas["f0"][fatia][medidas["sonoro"][fatia]]
        energia.append(round(float(volume.mean()), 1) if volume.size >= config.MIN_QUADROS else None)
        f0.append(round(float(np.median(voz)), 1) if voz.size >= config.MIN_QUADROS else None)
    return {"energia_db": energia, "f0_hz": f0}


# ---------------------------------------------------------------------------
# Todas as palestras
# ---------------------------------------------------------------------------


def analisar(palestras: list[dict[str, Any]], forcar: bool = False) -> dict[str, Any]:
    """
    Mede o som de todas as palestras com vídeo no disco.

    Palestra já medida sai do cache (acustica.json), então a opção roda na
    hora durante a apresentação. O resultado é gravado depois de cada
    palestra, e interromper no meio não perde o que já foi feito.
    """
    resultado = {} if forcar else carregar_analise()
    resultado.setdefault("palestras", {})
    for indice, palestra in enumerate(palestras, start=1):
        rotulo = f"  [{indice}/{len(palestras)}]"
        if palestra["id"] not in resultado["palestras"]:
            if not coleta.video_disponivel(palestra):
                print(f"{rotulo} sem vídeo no disco, pulada: {palestra['titulo'][:50]}")
                continue
            relogio = time.time()
            sinal = leitura.audio(palestra)
            medidas = caracteristicas_palestra(sinal)
            lista = pausas(medidas["energia_db"])
            resultado["palestras"][palestra["id"]] = {
                "resumo": resumir(medidas, lista, len(sinal) / config.TAXA_AMOSTRAGEM),
                "curvas": curvas_por_segundo(medidas),
                "pausas": lista,
            }
            with config.ARQ_ACUSTICA.open("w", encoding="utf-8") as arquivo:
                json.dump(resultado, arquivo, ensure_ascii=False)
            print(f"{rotulo} medida em {time.time() - relogio:.0f} s")
        r = resultado["palestras"][palestra["id"]]["resumo"]
        print(
            f"{rotulo} {r['duracao_s'] / 60:>3.0f} min  silêncio {r['silencio_pct']:>4.1f}%  "
            f"pausas/min {r['pausas_por_min']:>4.1f}  F0 {r['f0_mediana_hz'] or 0:>5.1f} Hz  "
            f"{palestra['titulo'][:30]}"
        )
    print(f"  salvo em {config.ARQ_ACUSTICA.name}")
    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_ACUSTICA.exists():
        return {}
    with config.ARQ_ACUSTICA.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ---------------------------------------------------------------------------
# Demonstração ao vivo
# ---------------------------------------------------------------------------


def preparar_trecho(palestras: list[dict[str, Any]]):
    """O trecho das demos (o mesmo das opções 3 e 7) e tudo que o gráfico 03 desenha."""
    import quadros

    palestra = quadros.palestra_demo(palestras)
    if palestra is None:
        return None
    inicio = float(config.DEMO_SEGUNDO)
    primeira = int(inicio * config.TAXA_AMOSTRAGEM)
    sinal = leitura.audio(palestra)[primeira:primeira + config.TRECHO_DEMO * config.TAXA_AMOSTRAGEM]
    medidas = caracteristicas(sinal, com_mel=True)
    medidas["sonoro"] = sonoros(medidas)
    return sinal, medidas, pausas(medidas["energia_db"]), palestra, inicio


def demonstrar(palestras: list[dict[str, Any]]) -> None:
    """
    Mostra na tela o caminho do som até os números.

    É a forma mais direta de explicar na apresentação o que o librosa faz:
    o mesmo trecho de 20 segundos, primeiro como lista de amostras, depois
    como energia, pausas, frequências e altura da voz.
    """
    import quadros

    trecho = preparar_trecho(palestras)
    if trecho is None:
        print("  Nenhum vídeo no disco. Rode a coleta primeiro (opção 1).")
        return
    sinal, medidas, lista, palestra, inicio = trecho
    resumo = resumir(medidas, lista, len(sinal) / config.TAXA_AMOSTRAGEM)

    minuto, segundo = divmod(int(inicio), 60)
    print(f"\n  PALESTRA: {palestra['titulo'][:66]}")
    print(f"  TRECHO:   {config.TRECHO_DEMO} s a partir de {minuto}:{segundo:02d} (o mesmo quadro da opção 3)")

    print("\n  1. O SOM COMO LISTA DE NÚMEROS")
    print(f"     taxa de amostragem ...... {config.TAXA_AMOSTRAGEM:,} amostras por segundo".replace(",", "."))
    print(f"     amostras no trecho ...... {len(sinal):,}".replace(",", "."))
    meio = len(sinal) // 2
    # float() antes do round: o float32 da librosa, convertido direto,
    # aparece como -0.032099999487400055 em vez de -0.0321.
    print(f"     8 amostras do meio ...... {[round(float(x), 4) for x in sinal[meio:meio + 8]]}")
    print(f"     menor e maior valor ..... {sinal.min():+.3f} e {sinal.max():+.3f}")

    print("\n  2. QUADROS DE 25 ms, A CADA 10 ms")
    print(f"     quadros no trecho ....... {medidas['energia_db'].size} (100 por segundo)")
    print(f"     energia média da voz .... {resumo['energia_db_media']} dB")
    print(f"     silêncio ................ {resumo['silencio_pct']}% do trecho")
    for inicio_p, fim_p in lista[:5]:
        print(f"       pausa de {fim_p - inicio_p:.1f} s em {inicio_p:.1f} s")

    print("\n  3. FREQUÊNCIA (FFT, MEL E MFCC)")
    print(f"     centróide espectral ..... {resumo['centroide_medio_hz']:.0f} Hz")
    print(f"     taxa de cruzamento zero . {resumo['zcr_media']:.3f}")
    print(f"     espectrograma Mel ....... {medidas['mel_db'].shape[0]} bandas x {medidas['mel_db'].shape[1]} quadros")
    print(f"     MFCC .................... {medidas['mfcc'].shape[0]} coeficientes x {medidas['mfcc'].shape[1]} quadros")
    print(f"     MFCC médio (5 primeiros)  {resumo['mfcc_media'][:5]}")

    print("\n  4. ALTURA DA VOZ (YIN)")
    print(f"     F0 mediana .............. {resumo['f0_mediana_hz']} Hz")
    print(f"     variação (desvio) ....... {resumo['f0_desvio_hz']} Hz")
    print(f"     quadros com voz ......... {resumo['quadros_sonoros_pct']}%")

    import visualizacao

    quadros.abrir(visualizacao.figura_som(sinal, medidas, lista, palestra, inicio))

    if input("\n  ouvir o trecho? [s/n] ").strip().lower() == "s":
        leitura.tocar(sinal, "trecho_som.wav")


if __name__ == "__main__":
    demonstrar(coleta.selecionar())
