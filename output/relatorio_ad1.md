# Avaliação a Distância 1 - Tópicos Especiais

**Aluno:** Fernando Paladini

## Repositório

Link do repositório pessoal no GitHub: [https://github.com/paladini/sri](https://github.com/paladini/sri)

Pasta de saída com o relatório e artefatos gerados: [`output/`](./)

## Execução dos notebooks

Antes da execução dos Notebooks principais, foi criado e executado o notebook `0_0_PrepararDocumentosCSV_trainjson_v1.ipynb`. Esse notebook adapta o dataset escolhido para o formato de entrada esperado pelos notebooks disponibilizados pelo professor, gerando o arquivo `data/documentos.csv` com as colunas `id` e `documento`.

Os notebooks disponibilizados pelo professor foram executados em 24/05/2026 na ordem solicitada:

1. `1_1_Segmentacao_Limpeza_v1.ipynb`
2. `1_2_GerarPOS_v1.ipynb`
3. `1_3_NER_spaCy_v1.ipynb`
4. `2_1_AnaliseDados_v1.ipynb`

Os arquivos de dados gerados pelos notebooks ficam na pasta `data/`. Os relatórios, imagens e exports ficam centralizados na pasta [`output/`](./). Os principais arquivos adicionados ao repositório foram:

| Arquivo | Descrição |
|---|---|
| `0_0_PrepararDocumentosCSV_trainjson_v1.ipynb` | Notebook criado para adaptar o dataset escolhido ao formato esperado pela sequência principal. |
| `data/dataset.csv` | Documentos limpos e segmentados em sentenças. |
| `data/datasetpos.csv` | Saída de POS Tagging gerada com spaCy. |
| `data/datasetner.csv` | Saída de NER gerada com spaCy. |
| `output/relatorio_ad1.md` | Relatório principal da Avaliação à Distância 1. |
| `output/relatorio_ad1_export/2_1_AnaliseDados_v1.md` | Export completo do notebook de análise com tabelas e gráficos. |
| `output/relatorio_ad1_export/2_1_AnaliseDados_v1_files/` | Imagens PNG geradas pelo notebook de análise. |

Essa organização mantém a raiz do repositório semelhante ao SRI original, com notebooks na raiz e a pasta `data/` com os arquivos processados, enquanto os materiais finais de entrega ficam agrupados em `output/`.

## Descrição do texto utilizado

O corpus utilizado foi o dataset de notícias do GovBR disponível no Hugging Face: [divergente/noticias-govbr-ptbr-1](https://huggingface.co/datasets/divergente/noticias-govbr-ptbr-1).

Como os notebooks `1_1` até `2_1`, disponibilizados pelo professor, esperam um arquivo `data/documentos.csv` com as colunas `id` e `documento`, foi criado o notebook `0_0_PrepararDocumentosCSV_trainjson_v1.ipynb` para adaptar o dataset escolhido a esse formato. O script lê o arquivo de entrada do dataset (`train.json`), extrai o campo textual utilizado na análise, normaliza os registros e gera o CSV consumido pelas etapas seguintes.

Após a adaptação, o corpus utilizado no processamento ficou preparado no arquivo `data/documentos.csv`. Ele possui 500 documentos curtos, armazenados nas colunas `id` e `documento`. Cada documento corresponde a um título textual em português extraído de notícias do portal GovBR. A base utilizada contém principalmente títulos relacionados a governo, controle público, Polícia Federal, CGU, operações, investigações, Covid-19 e ações institucionais.

Depois do pré-processamento, o arquivo `data/dataset.csv` manteve 500 documentos e 567 sentenças. O processamento linguístico foi feito com spaCy, usando o modelo `pt_core_news_lg`; a análise também utiliza o tokenizador BERT `neuralmind/bert-base-portuguese-cased`.

Resumo geral da base:

| Métrica | Valor |
|---|---:|
| Documentos | 500 |
| Sentenças | 567 |
| Entidades nomeadas reconhecidas | 1032 |
| Classes NER encontradas | LOC, MISC, ORG, PER |

## Exemplo de POS Tagging

Sentença utilizada:

> Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto

| Token | POS | Lema |
|---|---|---|
| Brasil | PROPN | Brasil |
| inicia | VERB | iniciar |
| construção | NOUN | construção |
| do | ADP | de o |
| 5 | PROPN | 5 |
| Plano | PROPN | Plano |
| de | ADP | de |
| Ação | PROPN | Ação |
| Nacional | PROPN | Nacional |
| de | ADP | de |
| Governo | PROPN | Governo |
| Aberto | PROPN | Aberto |

Verbo identificado na sentença: `inicia`.

## Exemplo de NER

Sentença utilizada:

> Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto

| Entidade | Classe | Início | Fim |
|---|---|---:|---:|
| Brasil | LOC | 0 | 6 |
| 5 Plano de Ação Nacional de Governo Aberto | MISC | 28 | 70 |

## Tabelas da análise de dados

### Distribuição de POS no corpus

| POS | Ocorrências |
|---|---:|
| PROPN | 1612 |
| NOUN | 1382 |
| ADP | 1353 |
| VERB | 725 |
| ADJ | 336 |
| NUM | 178 |
| PUNCT | 136 |
| CCONJ | 132 |
| DET | 112 |
| ADV | 62 |
| SCONJ | 55 |
| AUX | 19 |
| PRON | 16 |
| SYM | 5 |

### Distribuição de NER no corpus

| Classe | Ocorrências | Exemplo mais frequente |
|---|---:|---|
| LOC | 492 | Polícia Federal |
| ORG | 378 | PF |
| MISC | 151 | BPFRON |
| PER | 11 | LENI NCIA |

### Estatísticas por documento

| Estatística | Sentenças | Palavras | Tokens BERT | Palavras sem stopwords | Verbos | Substantivos | Entidades |
|---|---:|---:|---:|---:|---:|---:|---:|
| Média | 1.13 | 12.25 | 17.19 | 8.52 | 1.45 | 2.76 | 2.06 |
| Desvio padrão | 0.45 | 3.61 | 5.25 | 2.40 | 0.71 | 1.40 | 0.90 |
| Mínimo | 1.00 | 3.00 | 3.00 | 2.00 | 0.00 | 0.00 | 0.00 |
| Mediana | 1.00 | 12.00 | 17.00 | 8.00 | 1.00 | 3.00 | 2.00 |
| Máximo | 4.00 | 27.00 | 40.00 | 19.00 | 4.00 | 7.00 | 6.00 |

### Estatísticas por sentença

| Estatística | Palavras | Tokens BERT | Palavras sem stopwords | Verbos | Substantivos | Entidades |
|---|---:|---:|---:|---:|---:|---:|
| Média | 10.80 | 15.16 | 7.51 | 1.28 | 2.44 | 1.82 |
| Desvio padrão | 4.53 | 6.28 | 2.96 | 0.77 | 1.50 | 0.89 |
| Mínimo | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Mediana | 11.00 | 15.00 | 8.00 | 1.00 | 2.00 | 2.00 |
| Máximo | 23.00 | 38.00 | 17.00 | 4.00 | 7.00 | 5.00 |

### Estatísticas por janela

| Estatística | Palavras | Tokens BERT | Palavras janela 3 | Palavras janela 5 | Tokens janela 3 | Tokens janela 5 |
|---|---:|---:|---:|---:|---:|---:|
| Média | 10.80 | 15.16 | 12.36 | 12.52 | 17.41 | 17.64 |
| Desvio padrão | 4.53 | 6.28 | 3.92 | 3.83 | 5.72 | 5.60 |
| Mínimo | 1.00 | 1.00 | 2.00 | 3.00 | 2.00 | 3.00 |
| Mediana | 11.00 | 15.00 | 12.00 | 12.00 | 17.00 | 17.00 |
| Máximo | 23.00 | 38.00 | 27.00 | 27.00 | 40.00 | 40.00 |

As tabelas completas exportadas diretamente do notebook estão em [`relatorio_ad1_export/2_1_AnaliseDados_v1.md`](relatorio_ad1_export/2_1_AnaliseDados_v1.md).

## Gráficos gerados pelo notebook de análise

Todos os gráficos abaixo foram gerados pelo notebook `2_1_AnaliseDados_v1.ipynb`, a partir dos arquivos `data/dataset.csv`, `data/datasetpos.csv` e `data/datasetner.csv`.

### Por documento

![Boxplot das estatísticas gerais por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_122_2.png)

Este boxplot resume as medidas gerais por documento, como quantidade de sentenças, palavras, tokens BERT, palavras sem stopwords, verbos, substantivos e entidades. A maior parte dos documentos é curta, com mediana de 12 palavras, 17 tokens BERT e 1 sentença por documento.

![Boxplot de POS Tagging por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_127_1.png)

O gráfico compara a distribuição das classes gramaticais por documento. As classes mais presentes são `PROPN`, `NOUN` e `ADP`, o que faz sentido para títulos de notícias, pois eles concentram nomes próprios, substantivos e preposições.

![Distribuição de POS Tagging por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_129_0.png)

Este gráfico mostra a frequência total das classes POS no corpus. A predominância de `PROPN`, `NOUN`, `ADP` e `VERB` é coerente com um corpus formado por títulos jornalísticos institucionais.

![Boxplot de NER por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_134_0.png)

O boxplot de NER apresenta a quantidade de entidades por classe em cada documento. As classes `LOC` e `ORG` aparecem com mais frequência, resultado esperado para notícias do GovBR que citam órgãos, localidades, operações e instituições.

![Distribuição de NER por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_136_0.png)

Este gráfico sintetiza a distribuição das entidades nomeadas. A classe `LOC` foi a mais frequente, seguida de `ORG`, `MISC` e `PER`; isso mostra que o corpus contém mais referências geográficas e institucionais do que nomes de pessoas.

### Por sentença

![Boxplot das estatísticas gerais por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_144_0.png)

O gráfico apresenta as estatísticas no nível das sentenças. Como os documentos são títulos curtos, as sentenças também têm distribuição concentrada: média de 10,80 palavras e 15,16 tokens BERT por sentença.

![Boxplot de POS Tagging por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_148_0.png)

Este boxplot mostra como as classes POS se distribuem por sentença. `PROPN`, `NOUN` e `ADP` continuam concentrando os maiores valores, confirmando o padrão observado na análise por documento.

![Distribuição de POS Tagging por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_150_2.png)

O gráfico agrega as ocorrências de POS nas sentenças. Ele reforça que a estrutura linguística dos títulos é formada principalmente por nomes próprios, substantivos, preposições e verbos.

![Gráfico POS por sentença - ADP](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_153_2.png)

Este gráfico detalha a ocorrência de `ADP` por sentença. A frequência de preposições é compatível com títulos que conectam órgãos, ações, locais e complementos, como em expressões do tipo "de", "em" e "para".

![Gráfico POS por sentença - PROPN](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_154_2.png)

O gráfico de `PROPN` destaca a presença de nomes próprios nas sentenças. Esse resultado é esperado em notícias governamentais, que mencionam instituições, programas, operações e localidades específicas.

![Gráfico POS por sentença - NOUN](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_155_2.png)

A distribuição de `NOUN` mostra a frequência de substantivos comuns. Essa classe é importante no corpus porque os títulos descrevem ações, objetos de investigação, políticas públicas e temas administrativos.

![Gráfico POS por sentença - VERB](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_156_2.png)

O gráfico de `VERB` indica que a maioria das sentenças contém poucos verbos, geralmente um ou dois. Isso é coerente com títulos jornalísticos, que costumam expressar uma ação principal de forma direta.

![Gráfico POS por sentença - ADJ](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_157_2.png)

O gráfico de `ADJ` mostra menor frequência de adjetivos em comparação com substantivos e nomes próprios. O resultado combina com o estilo informativo dos títulos, que tende a priorizar fatos e entidades.

![Gráfico POS por sentença - NUM](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_158_2.png)

O gráfico de `NUM` evidencia ocorrências numéricas em parte das sentenças. Isso aparece em títulos com datas, valores, números de operações, planos, edições ou quantidades.

![Boxplot de NER por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_162_0.png)

O boxplot de NER por sentença mostra que cada sentença normalmente possui poucas entidades, com mediana de 2 entidades. Isso é compatível com sentenças curtas oriundas de títulos.

![Distribuição de NER por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_164_2.png)

Este gráfico mostra a frequência das classes NER nas sentenças. `LOC` e `ORG` seguem como as classes mais relevantes, indicando presença forte de lugares e organizações.

![Gráfico NER por sentença - LOC](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_167_2.png)

O gráfico de `LOC` detalha as entidades de localização por sentença. A concentração dessa classe é coerente com notícias que citam estados, cidades, países, órgãos tratados pelo modelo como localização e áreas de atuação.

![Gráfico NER por sentença - ORG](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_168_2.png)

O gráfico de `ORG` mostra a distribuição de organizações por sentença. A presença de órgãos como PF e CGU explica a frequência dessa classe no corpus.

![Gráfico NER por sentença - MISC](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_169_2.png)

O gráfico de `MISC` reúne entidades diversas que não se encaixam diretamente como pessoa, organização ou localização. Essa classe aparece em nomes de operações, programas e expressões institucionais.

### Distribuições por documento

![Quantidade de documentos por quantidade de sentenças](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_183_2.png)

Este gráfico confirma que a maioria dos documentos possui apenas uma sentença. Isso faz sentido porque o corpus processado utiliza títulos de notícias, que normalmente são textos curtos.

![Quantidade de documentos por quantidade de palavras](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_185_2.png)

O gráfico mostra a distribuição dos documentos pela quantidade de palavras. A maior concentração fica em documentos curtos, com mediana de 12 palavras.

![Quantidade de documentos por quantidade de tokens](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_187_2.png)

Este gráfico mostra a distribuição por tokens BERT. A mediana de 17 tokens e o máximo de 40 no dataset processado indicam que os documentos estão muito abaixo do limite de 512 tokens.

![Quantidade de documentos por quantidade de palavras sem stopwords](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_189_2.png)

O gráfico mostra a quantidade de palavras restantes após a remoção de stopwords. A redução em relação ao total de palavras é esperada, pois títulos em português contêm preposições e artigos frequentes.

![Quantidade de documentos por quantidade de locuções verbais](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_191_2.png)

Este gráfico apresenta a distribuição de locuções verbais por documento. Como os textos são títulos, a maior parte tem poucas locuções verbais.

![Quantidade de documentos por quantidade de verbos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_193_2.png)

O gráfico mostra a quantidade de verbos por documento. A concentração em poucos verbos confirma que os títulos descrevem uma ação principal, como "apura", "deflagra", "apoia" ou "inicia".

![Quantidade de documentos por quantidade de verbos e auxiliares](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_195_2.png)

Este gráfico soma verbos principais e auxiliares. A distribuição é próxima à de verbos porque os títulos do corpus usam poucos auxiliares.

![Quantidade de documentos por quantidade de substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_197_2.png)

O gráfico mostra a quantidade de substantivos por documento. A média de 2,76 substantivos por documento é coerente com títulos que nomeiam objetos, ações, instituições e temas.

![Quantidade de documentos por quantidade de auxiliares e substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_199_2.png)

Este gráfico combina verbos auxiliares e substantivos. A distribuição acompanha principalmente os substantivos, pois auxiliares aparecem pouco no corpus.

![Quantidade de documentos por quantidade de entidades reconhecidas](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_201_2.png)

O gráfico mostra quantas entidades foram reconhecidas por documento. A mediana de 2 entidades por documento é adequada para títulos curtos que geralmente citam pelo menos uma instituição, localidade ou operação.

![Distribuição do comprimento dos documentos tokenizados](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_204_0.png)

Este gráfico apresenta o comprimento dos documentos após a tokenização. Ele confirma que a base está dentro do limite exigido, sem documentos longos.

![Histograma do comprimento dos documentos tokenizados](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_205_0.png)

O histograma reforça a concentração dos documentos em faixas baixas de tokens. Esse padrão é esperado porque a análise foi feita sobre títulos de notícias.

### Distribuições por sentença

![Quantidade de sentenças por quantidade de palavras](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_208_2.png)

Este gráfico mostra a distribuição das sentenças por quantidade de palavras. A maior parte das sentenças tem tamanho curto ou intermediário, com média de 10,80 palavras.

![Quantidade de sentenças por quantidade de tokens](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_210_2.png)

O gráfico apresenta a quantidade de tokens BERT por sentença. A distribuição acompanha a de palavras, mas com valores um pouco maiores por causa da tokenização subword do BERT.

![Quantidade de sentenças por quantidade de palavras sem stopwords](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_212_2.png)

Este gráfico mostra a distribuição após remoção de stopwords. A redução do número de palavras evidencia a presença de termos funcionais comuns em português.

![Quantidade de sentenças por quantidade de locuções verbais](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_214_2.png)

O gráfico indica que as sentenças possuem poucas locuções verbais. Isso é compatível com títulos que tendem a ser sintaticamente enxutos.

![Quantidade de sentenças por quantidade de verbos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_216_2.png)

Este gráfico mostra a quantidade de verbos por sentença. A concentração em um ou dois verbos confirma o caráter objetivo das sentenças analisadas.

![Quantidade de sentenças por quantidade de verbos e auxiliares](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_218_2.png)

O gráfico soma verbos e auxiliares por sentença. O comportamento permanece próximo ao gráfico de verbos, indicando baixa incidência de auxiliares.

![Quantidade de sentenças por quantidade de substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_220_2.png)

Este gráfico evidencia a distribuição de substantivos por sentença. A classe aparece com frequência relevante porque os títulos nomeiam temas, ações, instituições e objetos de notícia.

![Quantidade de sentenças por quantidade de auxiliares e substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_222_2.png)

O gráfico combina auxiliares e substantivos por sentença. Como há poucos auxiliares, o padrão observado é explicado majoritariamente pela presença de substantivos.

![Quantidade de sentenças por quantidade de entidades](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_224_2.png)

Este gráfico mostra a quantidade de entidades por sentença. A concentração em poucas entidades por sentença é esperada para títulos curtos, mas a presença recorrente de entidades confirma que o corpus é adequado para análise de NER.
