# Diário de bordo · Etapa 3 (DevIA Radar)

Registro honesto dos erros que apareceram durante a implementação e de como
cada um foi resolvido. A professora pede esse conteúdo em **um slide**, então
o slide leva os 4 destacados com ⭐ e o resto fica de reserva para pergunta.

---

## ⭐ 1. O classificador achava 3 telas de código em 320

**O que aconteceu.** A primeira versão do tipo de tela marcava como "código"
a tela em que muitas linhas do OCR tinham cara de código: chave, ponto e
vírgula, `def`, `import`. Nas 320 telas das 8 palestras, saíram 3. Só a
palestra de Swift, que é programação ao vivo, passa mais da metade do tempo
num editor.

**Diagnóstico.** O editor daquela palestra mostrava YAML, número de linha e
nome de arquivo, e a regex não conhecia nenhum dos três. Além disso, a
pergunta estava mal feita: o que separa slide de captura de ferramenta é a
quantidade de texto. Slide é feito para ser lido do fundo da sala e tem
poucas linhas de letra grande; editor, terminal e site têm dezenas de linhas
miúdas.

**Correção.** Duas etapas em `leitura.tipo_de_tela`:

1. 50 linhas de OCR ou mais (30, se a tela for escura) é captura de
   ferramenta. Nenhum slide rotulado à mão passou de 44 linhas.
2. Dentro da captura, fundo escuro ou 15% das linhas com cara de código é
   "código"; o resto é "interface" (GitHub, n8n, navegador).

A regex ganhou chave de YAML e JSON, número de linha, nome de arquivo e
prompt de terminal (`config.py` e `leitura.RE_LINHA_CODIGO`).

**Resultado.** 48 acertos nas 48 telas rotuladas à mão que serviram para
escolher os limites, e 29 em 32 telas sorteadas que não serviram. Os 3 erros
foram capturas curtas lidas como slide, então o tempo de captura medido é um
piso. Agora são 63 telas de código e 61 de interface.

**Lição.** Regra escolhida olhando um conjunto de exemplos precisa ser
conferida em exemplos que ficaram de fora. Acertar 48 de 48 nos exemplos da
própria escolha não prova nada sozinho.

---

## ⭐ 2. A lista de vocabulário acerta "cloth" e inventa "Claude striked"

**O que aconteceu.** A FOSDEM publica uma legenda para cada palestra. Ela
também sai de um reconhecimento automático de fala, mas sem lista de
vocabulário. Comparando com a nossa transcrição no mesmo intervalo de tempo,
44 nomes de ferramenta aparecem só na nossa, e nenhum aparece só na legenda.

**Diagnóstico.** Conferimos os 44 à mão:

| Caso | Quantos | Exemplo |
|---|---|---|
| a legenda errou o nome | 33 | "with the cloth on the hard work testing" (era Claude, 11 vezes na mesma palestra); "CHAPGPT", "catch easy", "Chagipiti", "Olama" |
| só grafia diferente | 3 | "open AI", "chatGPTAPI" colado |
| não dá para decidir sem ouvir | 2 | |
| **nós erramos, por causa da lista** | **6** | "OpenAPI" virou "OpenAI" 5 vezes; "CrowdStrike" virou "Claude striked" |

A palestra de Swift usa o OpenAPI, um formato de especificação de API. Como
"OpenAI" está na lista, o Whisper escreveu "the OpenAI document" onde o
palestrante disse "OpenAPI document". Na palestra de Self-Healing Rollouts, "CrowdStrike"
(a empresa cuja atualização derrubou máquinas Windows em 2024) virou
"Claude striked". É o mesmo efeito colateral do "Claude Codex" da etapa 2.

**Correção.** A contagem continua automática, e o gráfico 05 diz "nomes que
só a nossa transcrição escreveu", em vez de "achou". As 6 menções inventadas
ficam na conta (5 do ChatGPT e 1 do Claude) e estão registradas aqui.

**Resultado.** A lista ganha de lavada (33 contra 6), mas não sai de graça. A
opção 5 mostra o caso bom ao vivo: aos 0:16 da palestra de testes com Claude,
o slide diz "Claude", a nossa transcrição escreve "Claude" e a legenda oficial
escreve "cloth". Para mostrar o caso ruim, a opção 9 com o padrão `striked`.

**Lição.** Uma ajuda ao modelo puxa os sons parecidos para as palavras dela.
É preciso medir o ganho e o custo, nos dois sentidos.

---

## ⭐ 3. O detector de rosto clássico não achava o palestrante

**O que aconteceu.** Para medir quanto tempo o palestrante aparece, o
primeiro detector foi o Haar cascade, que vem dentro do OpenCV. Em 96
quadros de teste da câmera, ele achou rosto em 20.

**Diagnóstico.** O Haar (Viola e Jones, 2001) procura um padrão de claro e
escuro de rosto grande e de frente. Na gravação da FOSDEM, a câmera é um
quadro pequeno no canto, e o palestrante anda, olha para a tela e fica de
perfil.

**Correção.** O YuNet (2023), uma rede neural de 227 KB que roda pelo próprio
OpenCV (`modelos/face_detection_yunet_2023mar.onnx`). Nos mesmos 96 quadros:
79 rostos. No quadro do gráfico 02, o YuNet acha o rosto com confiança 0,91 e
o Haar não acha nada.

**Efeito colateral.** O OpenCV 5 removeu o Haar cascade. Como a demonstração
mostra os dois lado a lado, o `requirements.txt` fixa o OpenCV abaixo da
versão 5 (testado na 4.13).

**Lição.** Técnica clássica serve de linha de base, mas precisa ser medida
no dado real antes de entrar na análise.

---

## ⭐ 4. O "Gemini" escrito na tela era o botão do navegador

**O que aconteceu.** O OCR achou o Gemini em 34 telas, e os palestrantes
disseram "Gemini" 4 vezes. Parecia que as palestras mostravam muito o Gemini
sem falar dele.

**Diagnóstico.** Conferindo o texto em volta de cada achado, 31 das 34 telas
são da mesma palestra (Self-Healing Rollouts) e mostram o botão "Gemini" que
o Chrome põe no canto da janela. Toda captura do navegador traz o botão. Só
3 são de fato o modelo: duas telas em que o sistema do palestrante mostra o
voto do "gemini2.5-flash", e um slide com a lista de modelos. A mesma
conferência achou um falso positivo do Sourcegraph Cody: "Cody Martin", um
jogador de basquete numa tabela de exemplo da palestra de Swift. O Cursor na
tela é real: é o menu do aplicativo Cursor, que o palestrante usa e nunca
cita.

**Correção.** A contagem continua, porque é o que a plateia vê, mas a
leitura muda: na tela, o número mede quantas vezes a plateia viu o nome
escrito. O assunto da palestra se mede melhor na fala.

**Lição.** O OCR lê tudo o que está na tela, inclusive os botões da
ferramenta que o palestrante usa para mostrar outra coisa.

---

## 5. A seleção encheu de palestra sobre filtro de spam

**O que aconteceu.** A primeira regra de seleção procurava termo de IA no
título e termo de desenvolvimento de software no resumo. A amostra veio com
palestras sobre filtro de spam com LLM e servidor de inferência.

**Diagnóstico.** "code" e "open source" aparecem no resumo de quase toda
palestra da FOSDEM, que é uma conferência de software livre.

**Correção.** O termo de desenvolvimento precisa estar no **título**, e
título sobre treinar ou servir modelo ("training", "dataset", "inference",
"serving") sai da amostra (`coleta.RE_IA_COMO_PRODUTO`).

**Resultado.** Das 1.944 palestras com vídeo das duas edições, 27 têm termo
de IA no título e 8 passam na regra nova, todas entre 15 e 50 minutos. A
amostra é a lista inteira das que passam. Entre as 19 que ficaram fora estão
servidor de LLM, dataset de treino, o filtro de spam e uma palestra sobre
mouses antigos, que entrou pela palavra "cursor".

---

## 6. O leitor de vídeo do OpenCV era lento demais

**O que aconteceu.** O `cv2.VideoCapture` decodificava 49 quadros por
segundo. Uma palestra de 25 minutos a 30 quadros por segundo tem 45 mil
quadros: 15 minutos só para ler cada palestra.

**Correção.** O PyAV, o FFmpeg dentro do Python, com `thread_type="AUTO"`
para usar todos os núcleos (`quadros.percorrer`): 924 quadros por segundo. O
AV1 não deixa pular quadro, então tudo é decodificado, mas só um quadro por
segundo vira matriz do numpy.

---

## 7. O palestrante andando parecia troca de slide

**O que aconteceu.** A troca de slide é detectada pela diferença de brilho
entre um segundo e o anterior. Nas primeiras séries, apareciam "trocas" com
o slide parado.

**Diagnóstico.** A câmera do palestrante fica **por cima** do canto inferior
direito do slide. Quando ele se mexia, aquele canto mudava.

**Correção.** Uma máscara zera o canto da câmera antes da conta
(`quadros.mascara_slide`).

---

## 8. Vinheta, patrocinador e slide em tela cheia

**O que aconteceu.** Algumas gravações começam com vinheta, mostram tela de
patrocinador ou trocam para o slide em tela cheia. Nesses segundos, o recorte
de slide e câmera pega a coisa errada.

**Correção.** A moldura fixa da FOSDEM (logotipo e faixa, fora do slide e da
câmera) é comparada com a moldura típica da palestra, a mediana de todos os
segundos. Segundo que foge dela sai da análise (`quadros.dentro_da_moldura`).
Na palestra de segurança da cadeia de software, 16,6% dos segundos saíram
assim.

---

## 9. O OCR colava as palavras

**O que aconteceu.** O OCR leu "usingClaudeCode" num slide, e a regex com
palavra inteira não achava o Claude.

**Correção.** Na tela, o texto e os termos do dicionário perdem os espaços
antes da comparação (`leitura.ferramentas_na_tela`). Na fala, a regex
continua exigindo a palavra inteira, porque lá o espaço é confiável.

---

## 10. As duas correções da etapa 2 também valem aqui

O "like" saiu do léxico do VADER e a unidade passou a ser a frase, como na
etapa 2, para o painel das três etapas comparar a mesma coisa. Nas palestras
o "like" aparece 5,3 vezes a cada mil palavras (16 nos podcasts, 0,5 nas
issues), então o efeito é menor:

| Medida | Positivo | Neutro | Negativo |
|---|---|---|---|
| trecho do Whisper, com "like" | 28,7% | 62,2% | 9,2% |
| frase, com "like" | 27,5% | 63,1% | 9,3% |
| frase, sem "like" (atual) | 24,5% | 66,0% | 9,5% |

---

## 11. Limites que ficaram

| Limite | Efeito |
|---|---|
| 8 palestras | Das 38 frases que citam o ChatGPT, 34 são da palestra que constrói um proxy do ChatGPT; das 25 do Claude, 19 são da palestra de testes com Claude. No vídeo, a participação das ferramentas reflete o tema dessas duas palestras. |
| Tempo de captura é um piso | 3 das 32 telas de conferência eram capturas curtas lidas como slide. |
| Tela apagada vira "visual" | A palestra do Firefox começa com 93 s de projetor preto. Sem texto, o classificador chama de "visual" (0,8% do tempo total). |
| Câmera em plano aberto | Na mesma palestra, a câmera filma a sala de longe e o rosto aparece em 1,1% do tempo. O palestrante está lá, mas pequeno demais para o detector. |
| A legenda oficial também é automática | A comparação mede o efeito da lista de vocabulário: as duas transcrições são automáticas. |
| Tempo da frase estimado | Como na etapa 2, o tempo de cada frase sai da posição dos caracteres no trecho do Whisper. |
