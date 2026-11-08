# Diário de bordo · Etapa 1 (DevIA Radar)

Registro honesto dos erros que apareceram durante a implementação e de como
cada um foi resolvido. A professora pede esse conteúdo em **um slide**, então
o slide leva os 4 destacados com ⭐ e o resto fica de reserva para pergunta.

---

## ⭐ 1. Bots de CI invertiam a conclusão do trabalho

**O que aconteceu.** Na primeira rodada completa, a análise de sentimento
dizia que as ferramentas de IA eram mal avaliadas: Claude com média −0,05,
GitHub Copilot −0,10, Gemini −0,35. O bigrama mais frequente de todo o corpus
era `workflow failure`, com 239 ocorrências.

**Diagnóstico.** `workflow failure` não é desenvolvedor discutindo IA. É robô
de integração contínua abrindo issue automática quando o build quebra. A busca
da API trouxe centenas desses relatórios porque o texto deles menciona Copilot
e outras ferramentas.

**Correção.** Filtro por autor e por título no módulo `limpeza.py`:

```python
RE_AUTOR_BOT = re.compile(r"(\[bot\]$|^dependabot|-(?:bot|ci)$)", re.IGNORECASE)
```

**Resultado.** 198 issues descartadas, e as médias viraram positivas:
Claude +0,15, Copilot +0,12, Gemini +0,32.

**Lição.** Dado não estruturado coletado de API traz conteúdo gerado por
máquina misturado com conteúdo humano. Sem separar os dois, a conclusão do
estudo sai invertida.

---

## ⭐ 2. O spaCy classificava `###` como valor monetário

**O que aconteceu.** No ranking de entidades nomeadas, o tipo `MONEY` liderava
com `###` (1.404 vezes), `####` (250) e `## Submission Process` (34).

**Diagnóstico.** `#` é símbolo de moeda em alguns países, e o modelo de NER
interpretou os títulos de markdown das issues como valores. O erro não estava
no spaCy, estava na limpeza, que deixava a sintaxe de markdown passar.

**Correção.** Duas frentes. Regex removendo markdown antes do PLN
(`RE_TITULO_MD`, `RE_ENFASE`, `RE_MARCADOR`, `RE_SIMBOLO`) e uma lista branca
de tipos de entidade em `config.ENTIDADES_RELEVANTES`, que deixa de fora
`MONEY`, `CARDINAL`, `ORDINAL`, `PERCENT`, `TIME` e `QUANTITY`.

**Lição.** O modelo de PLN só é tão bom quanto o pré-processamento que vem
antes dele.

---

## ⭐ 3. O programa morria no meio da execução por causa do terminal

**O que aconteceu.**

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 143-144
```

O programa rodava a coleta, a limpeza, o PLN inteiro e morria na hora de
imprimir as frases de exemplo.

**Diagnóstico.** O console do Windows usa a codificação `cp1252`, que não
consegue escrever travessão, aspas curvas nem emoji. As issues do GitHub estão
cheias desses caracteres. Rodar o mesmo código no Linux não reproduzia o erro.

**Correção.** Em `config.py`, antes de qualquer `print`:

```python
_saida.reconfigure(encoding="utf-8", errors="replace")
```

O `errors="replace"` troca o caractere impossível por `?` em vez de derrubar
o programa.

**Lição.** Erro de codificação de caractere não aparece no teste com dado de
exemplo, só com dado real. E aparece justamente na apresentação.

---

## ⭐ 4. A ordem das expressões regulares mudava o resultado

**O que aconteceu.** Duas vezes, e o sintoma foi silencioso nas duas: nenhuma
exceção, só resultado errado.

**Caso A, ordem de remoção.** A limpeza removia URLs antes dos blocos de
código. Como bloco de código contém URL, HTML e crase dentro dele, sobravam
fragmentos de código soltos no texto final. A correção foi remover o bloco de
código primeiro, porque ele é o padrão mais externo.

**Caso B, ordem das alternativas.** A regex de ferramentas usa `|`, que para
na primeira alternativa que casa. Com `copilot|github copilot`, o texto
"github copilot" era rotulado só como `copilot`, e o rótulo específico se
perdia. A correção foi ordenar do termo mais longo para o mais curto:

```python
ordenados = sorted(termos, key=len, reverse=True)
```

**Lição.** Em regex, ordem é lógica, não estilo.

---

## 5. Rate limit da GitHub API (HTTP 403)

A Search API aceita 10 consultas por minuto sem autenticação. A coleta
estourava o teto na quarta consulta e morria.

Resolvido com três medidas: pausa configurável entre requisições
(`PAUSA_SEM_TOKEN = 7.0`), detecção do status 403 com espera de 60 segundos, e
busca automática de token no GitHub CLI já instalado na máquina
(`gh auth token`). Com token, o teto sobe para 30 consultas por minuto.

## 6. Textos longos diluíam o sentimento

O `compound` do VADER é uma soma normalizada. Em issue de 40 mil caracteres,
as partes positivas e negativas se cancelam e o resultado tende a zero, o que
jogava tudo para "neutro".

Resolvido truncando em 2.000 caracteres antes de pontuar. Em issue, a opinião
aparece nos primeiros parágrafos, e o resto costuma ser log colado.

## 7. Erros do modelo que foram documentados em vez de corrigidos

Três resultados errados que decidimos manter e explicar, porque corrigir na
marra seria esconder a limitação real da ferramenta:

| Erro do modelo | Exemplo | Por que acontece |
|---|---|---|
| NER marca produto como lugar | `Cursor` e `AI` viram `GPE` (país/cidade) | O modelo `en_core_web_sm` foi treinado em notícia e blog, não em texto técnico. Nome curto e capitalizado parece topônimo |
| Lematização errada | `coding` vira `cod` | O lematizador erra quando a palavra é ambígua entre substantivo e verbo em contexto técnico |
| VADER não entende jargão | `critical`, `fatal`, `abort` puxam para negativo | São palavras técnicas neutras em issue de software, mas o léxico do VADER foi construído com linguagem de rede social |

Para resolver de verdade seria preciso treinar um modelo no domínio de
engenharia de software, o que fica fora do escopo da etapa.

## 8. Regex encontrava linguagem de programação dentro do código

Uma issue com trecho em Kotlin era contada como issue sobre Java, porque a
linha `User::class.java` casava com o padrão de linguagens.

O problema era onde a extração rodava, e não o padrão em si. Ferramenta de IA
e linguagem de programação agora são extraídas do texto **sem** os blocos de
código, enquanto os sinais estruturais (tem código, tem stack trace, versões)
continuam saindo do texto bruto, porque é neles que essa informação vive.

## 9. Templates de issue inflavam a frequência dos termos

Repositórios com template automático (`kburson/ai-task-manager` sozinho tem 41
issues) repetiam os mesmos cabeçalhos dezenas de vezes, e termos como
`submission process` subiam no ranking sem significar nada.

Resolvido com deduplicação por hash do texto limpo em `limpeza.processar`.
