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

## ⭐ 2. A lista de nomes acerta "cloth" e inventa "Claude striked"

**O que aconteceu.** A FOSDEM publica uma legenda para cada palestra. Ela
também sai de um reconhecimento automático de fala, mas sem lista de
nomes. Comparando com a nossa transcrição no mesmo intervalo de tempo,
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
"Claude striked".

**Correção.** A contagem continua automática, e o gráfico 06 diz "nomes que
só a nossa transcrição escreveu", em vez de "achou". As 6 menções inventadas
ficam na conta (5 do ChatGPT e 1 do Claude) e estão registradas aqui.

**Resultado.** A lista ganha de lavada (33 contra 6), mas não sai de graça. A
opção 7 mostra o caso bom ao vivo: aos 0:16 da palestra de testes com Claude,
o slide diz "Claude", a nossa transcrição escreve "Claude" e a legenda oficial
escreve "cloth". Para mostrar o caso ruim, a opção 11 com o padrão `striked`.

**Lição.** Uma ajuda ao modelo puxa os sons parecidos para as palavras dela.
É preciso medir o ganho e o custo, nos dois sentidos.

---

## 3. O detector de rosto clássico não achava o palestrante

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

## 10. As regras de texto que a fala e as issues dividem

O painel final (gráfico 11) põe o texto das issues do GitHub, da etapa 1, ao
lado da fala das palestras. Para as duas fontes serem comparáveis, elas
passam pelas mesmas três regras, e cada regra foi medida nas duas.

**O "like" saiu do léxico do VADER.** O VADER dá +1,5 a "like", pensando no
verbo "gostar". Nas palestras, a palavra aparece 140 vezes em 26.499 palavras
(5,3 a cada mil). O spaCy marcou a classe gramatical de cada "like" das
frases avaliadas: das 125 ocorrências, 93 são preposição ("like this"), 18
são interjeição (a muleta "it's like, you know"), 2 são adjetivo e só 12 são
o verbo. Com a palavra no léxico, 72 das 117 frases que têm "like" saíam
positivas só por causa dela. Nas issues, o "like" é raro (0,5 a cada mil
palavras), e o efeito é pequeno.

**A unidade é a frase.** A etapa 1 deu uma nota para cada issue inteira
(mediana de 311 palavras). O compound soma a nota de todas as palavras e
espreme a soma entre −1 e +1, então texto longo vai para os extremos: só 4%
das issues saíram neutras. Cortadas em frases pela mesma regra da fala
(`conteudo.RE_FRASE`, 6 palavras ou mais), as 979 issues viram 30.625 frases
avaliadas. Na fala, o trecho do Whisper já é curto, e cortar em frases muda
pouco.

| Medida | Positivo | Neutro | Negativo |
|---|---|---|---|
| palestras: trecho do Whisper, com "like" | 28,7% | 62,2% | 9,2% |
| palestras: frase, com "like" | 27,5% | 63,1% | 9,3% |
| palestras: frase, sem "like" (atual) | 24,5% | 66,0% | 9,5% |
| issues: issue inteira, com "like" (etapa 1) | 65,5% | 4,0% | 30,5% |
| issues: frase, com "like" | 34,4% | 41,1% | 24,6% |
| issues: frase, sem "like" (atual) | 33,9% | 41,5% | 24,7% |

**O Cursor conta pela maiúscula no texto escrito.** "Cursor" sozinho é o
produto e também palavra comum (o cursor da tela, o cursor de paginação). A
etapa 1 só contava "cursor ai", "cursor editor" e "cursor ide", e isso acha o
produto em 98 frases das issues. No texto escrito, a maiúscula é escolha do
autor, então a regra nova conta "Cursor" e deixa "cursor" de fora: 450
frases. Nas 15 conferidas entre as 352 que só a regra nova acha, as 15 são o
produto ("Editor: Cursor", "install Specflow as a Cursor plugin"). Das 12
primeiras com "cursor" minúsculo, 9 são paginação ou posição
("cursor-paginated", "the supplied cursor"). Na fala, quem escolhe a
maiúscula é o Whisper, então ali vale qualquer caixa. Só que nas 8 palestras
ninguém disse "cursor", nem na nossa transcrição nem na legenda oficial: o
Cursor das palestras está só na tela (entrada 4).

**Lição.** Comparar duas fontes exige as mesmas regras nas duas, e cada regra
precisa ser conferida no dado de cada uma.

---

## 11. O slide visual parecia deixar a fala mais emotiva

**O que aconteceu.** Separando as frases pelo tipo de tela do mesmo segundo,
o slide visual (foto, meme, diagrama) saiu com 33% de frases positivas e 17%
de negativas, contra 25% e 10% no slide de texto. Parecia que a imagem na
tela puxava a emoção de quem fala.

**Diagnóstico.** 141 das 193 frases ditas com slide visual são de uma
palestra só, a do Rspamd, que usa muito meme e foto. Essa palestra é mais
emotiva com qualquer tela: no slide de texto, tem 35% de frases positivas e
13% de negativas, e as outras sete palestras, 24% e 9%. O número junto
misturava o efeito da tela com o jeito de falar de uma pessoa. Dentro do
próprio Rspamd, o slide visual e o de texto têm quase a mesma fração de
frases positivas (33% e 35%).

**Correção.** A leitura mudou, e o slide de resultados diz isso. Sem o
Rspamd, sobram 52 frases com slide visual: 31% positivas e 4% negativas,
contra 24% e 9% no slide de texto. Com tão poucas frases, não dá para dizer
que a tela muda o tom.

**Lição.** Com 8 palestras, uma palestra pode responder por quase toda uma
categoria. Antes de comparar categorias, contar de onde vêm as frases de
cada uma.

---

## 12. O YIN pedia mais memória do que o computador tem

**O que aconteceu.** A primeira versão do `sinal.py` media a palestra inteira
de uma vez. O pico de memória (medido com o `tracemalloc`) cresce com a
duração: 216 MB em 1 minuto de áudio, 689 MB em 5 e 1,4 GB em 10. No ritmo
dos 5 para os 10 minutos (138 MB por minuto), a palestra do Firefox, de 50
minutos, pediria perto de 7 GB.

**Diagnóstico.** O `librosa.yin` monta uma janela de 1.024 amostras para cada
quadro, e são 100 quadros por segundo. A STFT e o Mel, juntos, gastam bem
menos: 93 MB em 5 minutos.

**Correção.** `sinal.caracteristicas_palestra` mede um minuto por vez e junta
os pedaços. Cada bloco devolve um quadro a mais, centrado no começo do bloco
seguinte, e esse quadro sai na junção.

**Resultado.** A palestra do Firefox inteira, em blocos: pico de 221 MB e 31
s. Para conferir a junção, 3 minutos de áudio foram medidos dos dois jeitos.
Dos 18.078 quadros, 6 diferem na energia e no MFCC (um de cada lado das duas
junções e o da própria junção) e 14 na F0, cuja janela é maior. A marcação de
quadro sonoro é idêntica.

---

## 13. O decibel mudava de um bloco para o outro

**O que aconteceu.** Na conferência dos blocos (entrada 12), a energia de
513 quadros não batia com a conta de uma vez só. Corrigida a energia, o MFCC
ainda diferia em 7.074 quadros (39%).

**Diagnóstico.** O `librosa.amplitude_to_db` e o `librosa.power_to_db`
cortam, por padrão, tudo que fica 80 dB abaixo do máximo do trecho que
receberam (`top_db=80`). Com a palestra em blocos, cada minuto tem o seu
máximo, e o piso muda de um bloco para o outro.

**Correção.** `top_db=None` nas duas conversões (`sinal.caracteristicas`). No
desenho do espectrograma (gráfico 03), o piso de 80 dB volta, só na escala de
cor.

**Resultado.** 6 quadros diferentes nas duas medidas, os das junções.

---

## 14. O YIN dá altura até para o silêncio

**O que aconteceu.** O `librosa.yin` devolve uma F0 para todo quadro, até
para a pausa e para o "s" de uma palavra, onde não existe altura nenhuma. Na
palestra de testes com Claude, 5,6% dos quadros saem grudados nos limites da
faixa (a menos de 2% de 65 ou de 400 Hz): 1,3% dos quadros com som e 14,5%
dos de silêncio.

**Correção.** Só entra na F0 o quadro sonoro (`sinal.sonoros`): acima do
limiar de silêncio, cruzando o zero poucas vezes (som de vogal) e longe dos
limites da faixa.

**Resultado.** Sem o filtro, a F0 mediana cai em todas as palestras, de 1,1
Hz (Self-Healing Rollouts) a 17,4 Hz (testes com Claude: 208,6 contra 226,0
Hz).

**Conferência.** Essa palestra tem F0 mediana de 226 Hz, e as outras sete
ficam entre 106 e 149 Hz. Parecia erro de oitava, o YIN achando metade do
período. O histograma dos quadros sonoros tem um pico só, entre 180 e 260 Hz
(61% dos quadros), e menos de 10% abaixo de 120 Hz: a voz de quem fala nessa
palestra é mais aguda. Como a voz entra na análise em escore z dentro de cada
palestra, a diferença entre vozes não pesa no cruzamento com a tela.

---

## ⭐ 15. A sala barulhenta escondia as pausas

**O que aconteceu.** A palestra do Rspamd saiu com 0,3 pausa longa por minuto
e 10,5% de silêncio. As outras sete ficaram entre 2,2 e 9,6 pausas por
minuto.

**Diagnóstico.** O silêncio seguia o critério do `librosa.effects.split`: 35
dB abaixo do quadro mais alto. Esse critério supõe uma sala quieta. No
Rspamd, os 5% de quadros mais quietos ficam abaixo de −49,8 dB, e o limiar
caía em −47,2 dB, dentro do próprio ruído de fundo. Os quadros de uma pausa
oscilavam em volta dele, e a pausa se quebrava em pedaços menores que 0,7 s.
Com o limiar de −48 a −38 dB, de 2 em 2, o Rspamd ia de 0,1 a 5,0 pausas por
minuto. Na palestra de Swift, com o ruído de fundo 18 dB mais baixo, a mesma
varredura ficava entre 9,1 e 10,1.

**Correção.** O limiar fica também pelo menos 10 dB acima do piso de ruído
(o percentil 5 da energia), e vale o mais alto dos dois
(`sinal.limiar_silencio`). Nas seis palestras de sala quieta, nada muda: a
regra antiga já ficava mais de 10 dB acima do ruído.

**Resultado.** Rspamd: 31,2% de silêncio e 4,3 pausas por minuto. A palestra
de Drag, Drop, and Deploy, a segunda mais barulhenta (piso em −51,7 dB), foi
de 2,2 para 4,0 pausas por minuto.

**Lição.** Um limiar relativo ao pico mede o microfone junto com a pessoa.
Antes de contar silêncio, é preciso olhar o ruído de fundo de cada gravação.

---

## 16. A voz por tipo de tela misturava pessoas

**O que aconteceu.** Juntando as 8 palestras, a altura da voz ficava 0,21
desvio-padrão acima do normal com slide visual na tela, e as pausas longas
iam de 5,1 por minuto no slide de texto para 8,3 com código.

**Diagnóstico.** A palestra do Firefox começa com 93 s de projetor apagado,
que o classificador lê como slide visual (entrada 17). Nesse trecho, quem
fala é outra pessoa, anunciando o palestrante ("Hello. So our next speaker
is..."), com F0 mediana de 224 Hz contra 112 Hz da palestra. Já nas pausas,
12 dos 20 minutos de código são da palestra de Swift, que pausa muito com
qualquer tela: 10,8 pausas por minuto no slide de texto.

**Correção.** A leitura mudou. Sem os 93 s do Firefox, o slide visual fica em
+0,06. Nas pausas, a comparação vale dentro de cada palestra: a captura de
ferramenta tem mais pausas que o slide de texto na de Self-Healing Rollouts
(6,6 no código e 8,9 na interface, contra 4,6) e na de Drag, Drop, and
Deploy (4,8 contra 3,3), e menos na de Swift (9,5 contra 10,8).

**Lição.** É a mesma da entrada 11, agora no som: com 8 palestras, uma pessoa
pode responder por quase toda uma categoria.

---

## 17. Limites que ficaram

| Limite | Efeito |
|---|---|
| 8 palestras | Das 38 frases que citam o ChatGPT, 34 são da palestra que constrói um proxy do ChatGPT; das 25 do Claude, 19 são da palestra de testes com Claude. No vídeo, a participação das ferramentas reflete o tema dessas duas palestras. |
| Tempo de captura é um piso | 3 das 32 telas de conferência eram capturas curtas lidas como slide. |
| Tela apagada vira "visual" | A palestra do Firefox começa com 93 s de projetor preto. Sem texto, o classificador chama de "visual" (0,8% do tempo total). |
| Câmera em plano aberto | Na mesma palestra, a câmera filma a sala de longe e o rosto aparece em 1,1% do tempo. O palestrante está lá, mas pequeno demais para o detector. |
| A legenda oficial também é automática | A comparação mede o efeito da lista de nomes: as duas transcrições são automáticas. |
| Tempo da frase estimado | O tempo de cada frase sai da posição dos caracteres no trecho do Whisper, e a voz da frase vem dos segundos que esse tempo cobre. |
| Uma voz por palestra | O som não separa as pessoas: quem anuncia o palestrante e quem pergunta da plateia entram na voz da palestra (entrada 16). |
| Pausa depende do limiar | No Rspamd, com o limiar em −42 ou −38 dB, em vez de −39,8, a contagem vai de 4,3 para 2,7 ou 5,0 pausas por minuto (entrada 15). |
