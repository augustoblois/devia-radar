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
| [1](etapa1/) | Texto | Issues do GitHub (Search API) | Regex, spaCy, VADER | **pronta** |
| 2 | Áudio | Podcasts | em construção | em construção |
| 3 | Vídeo | Palestras | em construção | em construção |

O tema é o mesmo nas três etapas, como o enunciado sugere. A fonte de texto
não precisa ser a mesma fonte de áudio e vídeo.

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

## Execução

Cada etapa é um programa independente, com menu próprio:

```powershell
cd etapa1
python main.py
```

O README de cada etapa explica as opções do menu, a teoria e os resultados.

---

## Estrutura

```
dados-nao-estruturados/
├── README.md              este arquivo
├── requirements.txt       dependências das três etapas
├── docs/
│   └── pergunta-de-pesquisa.pdf
├── etapa1/                texto: issues do GitHub
│   ├── README.md          opções, teoria e resultados da etapa
│   ├── main.py            menu do programa
│   ├── *.py               um módulo por técnica
│   ├── diario-de-bordo.md erros da implementação e as correções
│   ├── dados/             cache da coleta e resultados das análises
│   ├── saida/             gráficos PNG
│   └── apresentacao/      slides em PDF entregues no SIGAA
└── atividades/            exercícios avulsos da disciplina (regex, Turing)
```

Os roteiros de fala e os arquivos editáveis dos slides (`.pptx`) ficam em
`etapaN/apresentacao/`, só no computador da dupla (fora do Git).
