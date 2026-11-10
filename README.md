# DevIA Radar

Analisador de como desenvolvedores discutem **Inteligência Artificial no
desenvolvimento de software**, em três etapas de dado não estruturado: o texto
de issues do GitHub, o significado desse mesmo texto e o áudio e o vídeo de
palestras.

Projeto da disciplina **Dados Não Estruturados** (UFPB), feito em dupla por
**Augusto Blois** e **Pedro Flávio Nogueira Ribeiro de Luna**.
Tema 1 da lista de temas, dentro do grande tema de Engenharia de Software.

**Pergunta de pesquisa:** como desenvolvedores discutem o uso de Inteligência
Artificial no desenvolvimento de software? O que eles escrevem em issues
confirma ou contradiz o que falam em palestras?

---

## Etapas

| Etapa | Formato | Fonte | Técnicas | Situação |
|---|---|---|---|---|
| [1](etapa1/) | Texto | 979 issues do GitHub (Search API) | Regex, spaCy, VADER | **pronta** |
| [2](etapa2/) | Texto: semântica e sentimento | as mesmas 979 issues | Pipeline do spaCy, embeddings (sentence-transformers), KMeans, transformer (RoBERTa) | **pronta** |
| [3](etapa3/) | Áudio e vídeo | 8 palestras da FOSDEM (agenda em XML) | PyAV, OpenCV, librosa, RapidOCR, Whisper, regex, spaCy, VADER | **pronta** |

O tema é o mesmo nas três etapas, como o enunciado sugere. A fonte de texto
não precisa ser a mesma fonte de áudio e vídeo.

O enunciado mudou depois da etapa 1: a etapa 2, que era de áudio (podcasts),
passou a ser de análise semântica e de sentimentos, e o áudio foi para a
etapa 3, junto com o vídeo. O trabalho antigo da etapa 2 está guardado na tag
`arquivo/etapa2-audio`. O `docs/pergunta-de-pesquisa.pdf` é a primeira versão
da pergunta, de setembro, de antes dessa mudança.

## O que as três etapas mostram

| Etapa | O que mede | Resultado principal |
|---|---|---|
| 1 · texto | VADER, issue inteira | 65% das issues positivas e 31% negativas. Claude (305 issues) e ChatGPT (302) são as ferramentas mais citadas. |
| 2 · sentido | transformer, frase a frase | 85% das frases são neutras e, quando há tom, 78% é queixa. Parte do positivo da etapa 1 vinha de jargão: "commit" e "acceptance" são elogio para o VADER. |
| 3 · áudio e vídeo | VADER frase a frase, voz e tela | 9% das frases faladas são negativas, contra 25% das escritas nas issues, com o mesmo medidor. A voz quase não acompanha o sentimento da frase. |

Juntando as três: o escrito reclama mais que o falado, e a medida muda o
retrato. O mesmo texto que o VADER lia como 65% positivo, issue a issue, sai
85% neutro no transformer, frase a frase.

---

## Instalação

```powershell
git clone https://github.com/augustoblois/devia-radar.git
cd devia-radar
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

O download dos modelos do spaCy é obrigatório. A biblioteca instala sem os
modelos de idioma, que vêm separados: o `sm` serve às etapas 1 e 3, e o `md`,
que traz vetores de palavra, à etapa 2.

A etapa 2 baixa dois modelos do Hugging Face na primeira execução (cerca de
600 MB). A etapa 3 baixa o vídeo das palestras (715 MB), e o Whisper baixa o
modelo dele (cerca de 500 MB) na primeira transcrição. Com placa de vídeo
NVIDIA, a transcrição fica bem mais rápida: o README da etapa 3 explica como
ligar.

## Execução

Cada etapa é um programa independente, com menu próprio:

```powershell
cd etapa1    # ou etapa2, etapa3
python main.py
```

O README de cada etapa explica as opções do menu, a teoria e os resultados.

---

## Estrutura

```
devia-radar/
├── README.md              este arquivo
├── requirements.txt       dependências das três etapas
├── docs/
│   └── pergunta-de-pesquisa.pdf
├── etapa1/                texto: issues do GitHub
├── etapa2/                semântica e sentimento: as mesmas issues
├── etapa3/                áudio e vídeo: palestras da FOSDEM
└── atividades/            exercícios avulsos da disciplina (regex, Turing)
```

As três etapas têm a mesma organização:

```
etapaN/
├── README.md              opções, teoria e resultados da etapa
├── main.py                menu do programa
├── config.py              parâmetros centrais da etapa
├── *.py                   um módulo por técnica
├── diario-de-bordo.md     erros da implementação e as correções
├── dados/                 cache da coleta e resultados das análises
├── saida/                 gráficos PNG
└── apresentacao/          slides em PDF entregues no SIGAA
```

Os roteiros de fala e os arquivos editáveis dos slides (`.pptx`) ficam em
`etapaN/apresentacao/`, só no computador da dupla (fora do Git).
