# DevIA Radar

Analisador de como desenvolvedores discutem **Inteligência Artificial no
desenvolvimento de software**, em três formatos de dado não estruturado:
texto, áudio e vídeo.

Projeto da disciplina **Dados Não Estruturados** (UFPB), feito em dupla por
**Augusto Blois** e **Pedro Flávio Nogueira Ribeiro de Luna**.
Tema 1 da lista de temas, dentro do grande tema de Engenharia de Software.

**Pergunta de pesquisa:** como desenvolvedores discutem o uso de Inteligência
Artificial no desenvolvimento de software? O que eles escrevem em issues
confirma ou contradiz o que falam em podcasts e palestras?

---

## Etapas

| Etapa | Formato | Fonte | Técnicas | Situação |
|---|---|---|---|---|
| [1](etapa1/) | Texto | 979 issues do GitHub (Search API) | Regex, spaCy, VADER | **pronta** |
| [2](etapa2/) | Áudio | 14 episódios de podcast (iTunes Search API) | librosa, Whisper, regex, spaCy, VADER | **pronta** |
| [3](etapa3/) | Vídeo | 8 palestras da FOSDEM (agenda em XML) | PyAV, OpenCV, RapidOCR, Whisper, regex, spaCy, VADER | **pronta** |

O tema é o mesmo nas três etapas, como o enunciado sugere. A fonte de texto
não precisa ser a mesma fonte de áudio e vídeo.

## O que as três etapas mostram

As três etapas medem o sentimento do mesmo jeito (frase a frase, com o mesmo
VADER) e contam as ferramentas com o mesmo dicionário:

| Etapa | Amostra | Positivas | Neutras | Negativas | Ferramenta mais citada |
|---|---|---|---|---|---|
| texto | 979 issues do GitHub | 34% | 41% | 25% | ChatGPT (30%) |
| fala | 14 episódios de podcast, 10 h | 30% | 61% | 9% | Claude (41%) |
| vídeo | 8 palestras da FOSDEM, 3,3 h | 25% | 66% | 9% | ChatGPT (51%) |

A resposta à pergunta é "em parte as duas coisas":

- **O tom muda com o lugar.** Um quarto das frases das issues é negativo,
  contra 9% na fala dos podcasts e das palestras. Issue é onde se relata
  defeito; podcast é conversa, e palestra é apresentação preparada.
- **As ferramentas são as mesmas, em outra ordem.** O ChatGPT lidera no texto
  e o Claude lidera na fala dos podcasts. No vídeo, a ordem segue o tema de
  duas palestras (ver o [README da etapa 3](etapa3/README.md)).

---

## Instalação

```powershell
git clone https://github.com/augustoblois/devia-radar.git
cd devia-radar
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

O download do modelo do spaCy é obrigatório. A biblioteca instala sem o
modelo de idioma, que vem separado.

As etapas 2 e 3 baixam o áudio (670 MB) e o vídeo (715 MB) na primeira
execução, e o Whisper baixa o modelo dele (cerca de 500 MB) na primeira
transcrição. Com placa de vídeo NVIDIA, a transcrição fica bem mais rápida:
o README da etapa 2 explica como ligar.

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
├── etapa2/                áudio: podcasts
├── etapa3/                vídeo: palestras da FOSDEM
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
