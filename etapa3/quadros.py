"""
Etapa 3 - QUADROS: o vídeo virando números (PyAV + OpenCV).

Para o computador, vídeo é uma pilha de imagens mostradas rápido. Cada
imagem é um quadro (frame), e cada quadro é uma matriz de pixels. Uma
palestra de 25 minutos a 25 quadros por segundo tem 37.500 quadros. Este
módulo olha UM quadro por segundo e mede o que acontece na tela.

Vocabulário mínimo (aparece o tempo todo no código):

- Pixel: um ponto da imagem. Em tons de cinza, um número de 0 (preto) a
  255 (branco). Em cor, três números. O OpenCV guarda na ordem azul, verde,
  vermelho (BGR), ao contrário do RGB de quase todo o resto.
- Codec: o jeito de comprimir o vídeo. O AV1 guarda um quadro completo de
  tempos em tempos (o quadro-chave) e, entre eles, só o que mudou. Por isso
  não dá para pular direto para um quadro: é preciso decodificar desde o
  último quadro-chave. Quem decodifica é o PyAV, o FFmpeg dentro do Python.
- Região de interesse (ROI, region of interest): o pedaço do quadro que
  importa. A FOSDEM monta todas as gravações com o mesmo desenho: slide no
  canto superior esquerdo, câmera do palestrante no canto inferior direito.
- Diferença entre quadros: subtrair, pixel a pixel, o quadro deste segundo
  do quadro do segundo anterior e tirar a média do valor absoluto. Perto de
  zero, nada mudou. Alta no slide, o slide trocou. Alta na câmera, alguém
  se mexeu.
- Bordas (detector de Canny, 1986): marca os pixels onde o brilho muda de
  repente, como o contorno de uma letra. Tela de código e de texto miúdo
  tem muita borda; foto e diagrama simples têm pouca.
- Detector de rosto: o YuNet (2023) é uma rede neural de 227 KB que devolve
  a caixa de cada rosto com uma confiança de 0 a 1. Ele substituiu o Haar
  cascade (Viola e Jones, 2001), que só acha rosto grande e de frente.

Onde aprender: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

import coleta
import config

# ---------------------------------------------------------------------------
# Leitura do vídeo
# ---------------------------------------------------------------------------


def percorrer(caminho: Path, passo: float = config.AMOSTRAGEM) -> Iterator[tuple[float, np.ndarray]]:
    """
    Devolve (segundo, quadro) a cada `passo` segundos, do começo ao fim.

    O PyAV decodifica todos os quadros, porque o AV1 não deixa pular, mas
    só converte em matriz do numpy o primeiro de cada segundo. Com
    thread_type="AUTO" ele usa todos os núcleos da CPU e passa de 900
    quadros por segundo; o leitor do próprio OpenCV fazia 49.
    """
    with av.open(str(caminho)) as conteiner:
        fluxo = conteiner.streams.video[0]
        fluxo.thread_type = "AUTO"
        proximo = 0.0
        for quadro in conteiner.decode(fluxo):
            if quadro.time is None or quadro.time < proximo:
                continue
            yield round(quadro.time, 2), quadro.to_ndarray(format="bgr24")
            proximo = (int(quadro.time / passo) + 1) * passo


def quadro_em(caminho: Path, segundo: float) -> np.ndarray:
    """Um quadro só: pula para o quadro-chave anterior e decodifica até o segundo pedido."""
    with av.open(str(caminho)) as conteiner:
        fluxo = conteiner.streams.video[0]
        fluxo.thread_type = "AUTO"
        conteiner.seek(int(segundo / fluxo.time_base), stream=fluxo, backward=True)
        for quadro in conteiner.decode(fluxo):
            if quadro.time is not None and quadro.time >= segundo:
                return quadro.to_ndarray(format="bgr24")
    raise ValueError(f"o vídeo acaba antes de {segundo} s")


# ---------------------------------------------------------------------------
# Medidas de um quadro
# ---------------------------------------------------------------------------


def recortar(quadro: np.ndarray, regiao: tuple[float, float, float, float]) -> np.ndarray:
    """Recorta a região dada em fração da largura e da altura do quadro."""
    altura, largura = quadro.shape[:2]
    x0, y0, x1, y1 = regiao
    return quadro[int(y0 * altura):int(y1 * altura), int(x0 * largura):int(x1 * largura)]


def miniatura(imagem: np.ndarray, tamanho: tuple[int, int]) -> np.ndarray:
    """
    Versão cinza e pequena, para comparar um segundo com o outro.

    INTER_AREA tira a média dos pixels que viram um só, o que apaga o
    chuvisco da compressão. Sem isso, um slide parado "mudaria" um pouco
    a cada segundo.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cinza, tamanho, interpolation=cv2.INTER_AREA)


def mascara_slide() -> np.ndarray:
    """
    Pixels da miniatura do slide que entram na conta da diferença.

    A câmera fica POR CIMA do canto inferior direito do slide. Sem esta
    máscara, o palestrante andando naquele canto pareceria troca de slide.
    """
    largura, altura = config.TAMANHO_SLIDE
    sx0, sy0, sx1, sy1 = config.REGIAO_SLIDE
    cx0, cy0 = config.REGIAO_CAMERA[:2]
    mascara = np.ones((altura, largura), dtype=bool)
    linha = int((cy0 - sy0) / (sy1 - sy0) * altura)
    coluna = int((cx0 - sx0) / (sx1 - sx0) * largura)
    mascara[linha:, coluna:] = False
    return mascara


def mascara_moldura() -> np.ndarray:
    """Pixels FORA do slide e da câmera: a moldura fixa da FOSDEM."""
    largura, altura = config.TAMANHO_FUNDO
    mascara = np.ones((altura, largura), dtype=bool)
    for x0, y0, x1, y1 in (config.REGIAO_SLIDE, config.REGIAO_CAMERA):
        mascara[
            max(int(y0 * altura) - 1, 0):int(y1 * altura) + 2,
            max(int(x0 * largura) - 1, 0):int(x1 * largura) + 2,
        ] = False
    return mascara


def densidade_bordas(slide: np.ndarray) -> float:
    """Fração dos pixels do slide que o Canny marca como borda (0 a 1)."""
    reduzido = cv2.resize(slide, config.TAMANHO_BORDAS, interpolation=cv2.INTER_AREA)
    bordas = cv2.Canny(cv2.cvtColor(reduzido, cv2.COLOR_BGR2GRAY), config.CANNY_BAIXO, config.CANNY_ALTO)
    return float((bordas > 0).mean())


def detector_rosto() -> cv2.FaceDetectorYN:
    """Carrega o YuNet. O tamanho de entrada é ajustado a cada imagem."""
    return cv2.FaceDetectorYN.create(str(config.MODELO_ROSTO), "", (320, 320), config.CONFIANCA_ROSTO)


def rostos(detector: cv2.FaceDetectorYN, imagem: np.ndarray) -> np.ndarray:
    """
    Rostos achados na imagem, um por linha.

    Cada linha tem 15 números: x, y, largura e altura da caixa, 5 pontos do
    rosto (olhos, nariz, cantos da boca) e, por último, a confiança.
    """
    detector.setInputSize((imagem.shape[1], imagem.shape[0]))
    _, achados = detector.detect(imagem)
    return achados if achados is not None else np.empty((0, 15), dtype=np.float32)


# ---------------------------------------------------------------------------
# Uma palestra inteira
# ---------------------------------------------------------------------------


def _jpg(imagem: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def medir(caminho: Path) -> tuple[dict[str, list], np.ndarray, dict[int, bytes]]:
    """
    Uma passada pelo vídeo, um quadro por segundo.

    Devolve três coisas:
    - a série de medidas, um valor por segundo;
    - a moldura de cada segundo, para conferir depois se o quadro segue o
      desenho padrão da FOSDEM;
    - um JPG do slide de cada trecho que ficou parado TELA_MINIMA segundos,
      indexado pelo segundo em que o trecho começou. O OCR lê esses JPGs.
    """
    m_slide, m_moldura = mascara_slide(), mascara_moldura()
    detector = detector_rosto()
    serie: dict[str, list] = {c: [] for c in ("t", "dif_slide", "dif_camera", "brilho", "bordas", "rostos")}
    molduras, candidatas = [], {}
    anterior_slide = anterior_camera = None
    estavel_desde = 0

    for indice, (segundo, quadro) in enumerate(percorrer(caminho)):
        slide = recortar(quadro, config.REGIAO_SLIDE)
        camera = recortar(quadro, config.REGIAO_CAMERA)
        mini_slide = miniatura(slide, config.TAMANHO_SLIDE)
        mini_camera = miniatura(camera, config.TAMANHO_CAMERA)

        dif_slide = 0.0 if anterior_slide is None else float(cv2.absdiff(mini_slide, anterior_slide)[m_slide].mean())
        dif_camera = 0.0 if anterior_camera is None else float(cv2.absdiff(mini_camera, anterior_camera).mean())
        anterior_slide, anterior_camera = mini_slide, mini_camera

        # Slide mudou: começa um trecho novo. Trecho que chega a TELA_MINIMA
        # segundos parado vira candidato a tela, e o quadro de agora (igual
        # ao do começo do trecho, porque nada mudou) é guardado.
        if dif_slide > config.LIMIAR_TROCA:
            estavel_desde = indice
        elif indice - estavel_desde + 1 == config.TELA_MINIMA:
            candidatas[estavel_desde] = _jpg(slide)

        serie["t"].append(segundo)
        serie["dif_slide"].append(round(dif_slide, 2))
        serie["dif_camera"].append(round(dif_camera, 2))
        serie["brilho"].append(round(float(mini_slide[m_slide].mean()), 1))
        serie["bordas"].append(round(densidade_bordas(slide), 4))
        serie["rostos"].append(len(rostos(detector, camera)))
        molduras.append(miniatura(quadro, config.TAMANHO_FUNDO)[m_moldura])

    return serie, np.stack(molduras), candidatas


def dentro_da_moldura(molduras: np.ndarray) -> np.ndarray:
    """
    Marca os segundos em que o quadro segue o desenho padrão da FOSDEM.

    A mediana, pixel a pixel, de todos os segundos é a moldura típica da
    palestra, porque a moldura fixa aparece na grande maioria deles. Segundo
    que foge dela é vinheta de abertura, tela de patrocinador ou slide em
    tela cheia: ali o recorte de slide e câmera não vale.
    """
    pilha = molduras.astype(np.int16)
    tipica = np.median(pilha, axis=0)
    return np.abs(pilha - tipica).mean(axis=1) <= config.LIMIAR_MOLDURA


def telas(dif_slide: list[float], valido: np.ndarray) -> list[tuple[int, int]]:
    """
    Corta a série em telas: trechos em que o slide ficou parado.

    Cada troca (diferença acima do LIMIAR_TROCA) abre um trecho novo.
    Trecho com pelo menos TELA_MINIMA segundos é uma tela. Trecho mais
    curto é animação, rolagem ou vídeo passando: conta como "tela em
    movimento". Devolve pares (primeiro segundo, segundo depois do último).
    """
    cortes = [0] + [i for i, d in enumerate(dif_slide) if i > 0 and d > config.LIMIAR_TROCA]
    cortes.append(len(dif_slide))
    return [
        (inicio, fim)
        for inicio, fim in itertools.pairwise(cortes)
        if fim - inicio >= config.TELA_MINIMA and valido[inicio:fim].mean() > 0.5
    ]


def resumir(serie: dict[str, list], valido: np.ndarray, trechos: list[tuple[int, int]]) -> dict[str, Any]:
    """Transforma milhares de segundos em um punhado de números por palestra."""
    total = len(valido)
    minutos = max(float(valido.sum()) * config.AMOSTRAGEM / 60, 1e-9)
    em_tela = np.zeros(total, dtype=bool)
    for inicio, fim in trechos:
        em_tela[inicio:fim] = True
    duracoes = [(fim - inicio) * config.AMOSTRAGEM for inicio, fim in trechos]

    def media(chave: str) -> float:
        return float(np.mean(np.array(serie[chave], dtype=float)[valido])) if valido.any() else 0.0

    brilho = np.array(serie["brilho"])[valido]
    com_rosto = np.array(serie["rostos"])[valido] > 0
    return {
        "duracao_s": round(total * config.AMOSTRAGEM, 1),
        "moldura_pct": round(float(valido.mean()) * 100, 1),
        "telas": len(trechos),
        "telas_por_min": round(len(trechos) / minutos, 2),
        "tela_mediana_s": round(float(np.median(duracoes)), 1) if duracoes else None,
        "tela_movimento_pct": round(float((valido & ~em_tela).sum() / max(valido.sum(), 1)) * 100, 1),
        "tela_escura_pct": round(float((brilho < config.LIMIAR_ESCURO).mean()) * 100, 1) if brilho.size else 0.0,
        "brilho_medio": round(media("brilho"), 1),
        "bordas_media": round(media("bordas"), 4),
        "rosto_pct": round(float(com_rosto.mean()) * 100, 1) if com_rosto.size else 0.0,
        "movimento_camera": round(media("dif_camera"), 2),
    }


def analisar_palestra(palestra: dict[str, Any]) -> dict[str, Any]:
    """Mede a palestra, corta as telas e grava um JPG de cada uma."""
    serie, molduras, candidatas = medir(coleta.caminho_video(palestra))
    valido = dentro_da_moldura(molduras)
    trechos = telas(serie["dif_slide"], valido)

    pasta = config.DIR_TELAS / palestra["id"]
    pasta.mkdir(parents=True, exist_ok=True)
    for antigo in pasta.glob("*.jpg"):
        antigo.unlink()
    lista = []
    for numero, (inicio, fim) in enumerate(trechos, start=1):
        arquivo = pasta / f"tela_{numero:03d}.jpg"
        arquivo.write_bytes(candidatas[inicio])
        lista.append({
            "numero": numero,
            "inicio": serie["t"][inicio],
            "fim": round(serie["t"][fim - 1] + config.AMOSTRAGEM, 2),
            "arquivo": arquivo.name,
            "brilho": round(float(np.mean(serie["brilho"][inicio:fim])), 1),
            "bordas": round(float(np.mean(serie["bordas"][inicio:fim])), 4),
        })

    serie["valido"] = [int(v) for v in valido]
    return {"resumo": resumir(serie, valido, trechos), "serie": serie, "telas": lista}


def analisar(palestras: list[dict[str, Any]], forcar: bool = False) -> dict[str, Any]:
    """
    Analisa todas as palestras com vídeo no disco.

    Palestra já analisada sai do cache. O resultado é gravado depois de
    cada palestra, então interromper no meio não perde o que já foi feito.
    """
    resultado = carregar_analise() if not forcar else {}
    resultado.setdefault("palestras", {})
    for indice, palestra in enumerate(palestras, start=1):
        rotulo = f"  [{indice}/{len(palestras)}]"
        if not coleta.video_disponivel(palestra):
            print(f"{rotulo} sem vídeo no disco, pulada: {palestra['titulo'][:50]}")
            continue
        if palestra["id"] in resultado["palestras"]:
            r = resultado["palestras"][palestra["id"]]["resumo"]
        else:
            resultado["palestras"][palestra["id"]] = analisar_palestra(palestra)
            r = resultado["palestras"][palestra["id"]]["resumo"]
            with config.ARQ_VISUAL.open("w", encoding="utf-8") as arquivo:
                json.dump(resultado, arquivo, ensure_ascii=False)
        print(
            f"{rotulo} {r['duracao_s'] / 60:>3.0f} min  {r['telas']:>3} telas  "
            f"rosto {r['rosto_pct']:>5.1f}%  escura {r['tela_escura_pct']:>5.1f}%  "
            f"moldura {r['moldura_pct']:>5.1f}%  {palestra['titulo'][:30]}"
        )
    print(f"  salvo em {config.ARQ_VISUAL.name}")
    return resultado


def carregar_analise() -> dict[str, Any]:
    if not config.ARQ_VISUAL.exists():
        return {}
    with config.ARQ_VISUAL.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def abrir(caminho: Path) -> None:
    """Abre um arquivo no programa padrão do sistema (o visualizador de imagem)."""
    if sys.platform == "win32":
        os.startfile(caminho)
    else:
        print(f"  arquivo em {caminho}")


# ---------------------------------------------------------------------------
# Demonstração ao vivo
# ---------------------------------------------------------------------------


def palestra_demo(palestras: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A palestra das demos (config.DEMO_PALESTRA), ou a primeira com vídeo."""
    com_video = [p for p in palestras if coleta.video_disponivel(p)]
    for palestra in com_video:
        if palestra["id"] == config.DEMO_PALESTRA:
            return palestra
    return com_video[0] if com_video else None


def preparar_demo(palestras: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Lê dois quadros seguidos da palestra da demo e calcula tudo que o gráfico 02 desenha."""
    palestra = palestra_demo(palestras)
    if palestra is None:
        return None
    caminho = coleta.caminho_video(palestra)
    segundo = config.DEMO_SEGUNDO
    quadro = quadro_em(caminho, segundo)
    anterior = quadro_em(caminho, segundo - config.AMOSTRAGEM)

    slide, camera = recortar(quadro, config.REGIAO_SLIDE), recortar(quadro, config.REGIAO_CAMERA)
    m_slide = mascara_slide()
    mini, mini_anterior = miniatura(slide, config.TAMANHO_SLIDE), miniatura(recortar(anterior, config.REGIAO_SLIDE), config.TAMANHO_SLIDE)
    reduzido = cv2.resize(slide, config.TAMANHO_BORDAS, interpolation=cv2.INTER_AREA)

    # O Haar entra só aqui, para a turma ver lado a lado por que ele saiu.
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cinza_camera = cv2.equalizeHist(cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY))
    return {
        "palestra": palestra,
        "segundo": segundo,
        "quadro": quadro,
        "slide": slide,
        "camera": camera,
        "dif_slide": float(cv2.absdiff(mini, mini_anterior)[m_slide].mean()),
        "brilho": float(mini[m_slide].mean()),
        "bordas": cv2.Canny(cv2.cvtColor(reduzido, cv2.COLOR_BGR2GRAY), config.CANNY_BAIXO, config.CANNY_ALTO),
        "rostos_yunet": rostos(detector_rosto(), camera),
        "rostos_haar": haar.detectMultiScale(cinza_camera, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)),
    }


def demonstrar(palestras: list[dict[str, Any]]) -> None:
    """
    Mostra na tela o caminho de um quadro de vídeo até os números.

    O mesmo segundo de uma palestra, primeiro como matriz de pixels, depois
    recortado em slide e câmera, comparado com o segundo anterior, passado
    pelo Canny e pelos dois detectores de rosto.
    """
    demo = preparar_demo(palestras)
    if demo is None:
        print("  Nenhum vídeo no disco. Rode a coleta primeiro (opção 1).")
        return
    palestra, quadro = demo["palestra"], demo["quadro"]
    with av.open(str(coleta.caminho_video(palestra))) as conteiner:
        fluxo = conteiner.streams.video[0]
        codec, fps, quadros_total = fluxo.codec_context.name, float(fluxo.average_rate), fluxo.frames
        duracao = float(conteiner.duration / av.time_base)

    minuto, segundo = divmod(int(demo["segundo"]), 60)
    print(f"\n  PALESTRA: {palestra['titulo'][:66]}")
    print(f"            FOSDEM {palestra['ano']}, quadro em {minuto}:{segundo:02d}")

    print("\n  1. O VÍDEO COMO PILHA DE IMAGENS")
    print(f"     codec ................... {codec}")
    print(f"     quadros por segundo ..... {fps:.0f}")
    total = quadros_total or int(duracao * fps)
    print(f"     quadros na palestra ..... {total:,} em {duracao / 60:.0f} min".replace(",", "."))
    print(f"     um quadro (matriz) ...... {quadro.shape[0]} linhas x {quadro.shape[1]} colunas x {quadro.shape[2]} cores")
    y, x = quadro.shape[0] // 2, quadro.shape[1] // 2
    print(f"     pixel do centro (B,G,R) . {quadro[y, x].tolist()}")

    print("\n  2. REGIÕES DE INTERESSE")
    print(f"     slide ................... {demo['slide'].shape[1]} x {demo['slide'].shape[0]} pixels")
    print(f"     câmera .................. {demo['camera'].shape[1]} x {demo['camera'].shape[0]} pixels")

    print("\n  3. DIFERENÇA PARA O SEGUNDO ANTERIOR")
    print(f"     no slide ................ {demo['dif_slide']:.2f} (limiar de troca: {config.LIMIAR_TROCA})")
    print(f"     o slide trocou? ......... {'sim' if demo['dif_slide'] > config.LIMIAR_TROCA else 'não'}")
    print(f"     brilho médio do slide ... {demo['brilho']:.0f} de 255 ({'escuro' if demo['brilho'] < config.LIMIAR_ESCURO else 'claro'})")

    print("\n  4. BORDAS (Canny)")
    print(f"     pixels de borda ......... {(demo['bordas'] > 0).mean() * 100:.1f}% do slide")

    print("\n  5. ROSTOS NA CÂMERA")
    print(f"     Haar (2001) ............. {len(demo['rostos_haar'])} rosto(s)")
    print(f"     YuNet (2023) ............ {len(demo['rostos_yunet'])} rosto(s)", end="")
    if len(demo["rostos_yunet"]):
        print(f", confiança {demo['rostos_yunet'][0][-1]:.2f}")
    else:
        print()

    import visualizacao

    abrir(visualizacao.figura_quadro(demo))


if __name__ == "__main__":
    demonstrar(coleta.selecionar())
