# Avaliação a Distância 1 - Tópicos Especiais

**Aluno:** Fernando Paladini

## Repositório

Link do repositório pessoal no GitHub: [https://github.com/paladini/sri](https://github.com/paladini/sri)

## Execução dos notebooks

Antes da sequência principal da avaliação, foi criado o notebook `0_0_PrepararDocumentosCSV_trainjson_v1.ipynb`. Esse notebook adapta o dataset escolhido para o formato de entrada esperado pelos notebooks disponibilizados pelo professor, gerando o arquivo `data/documentos.csv` com as colunas `id` e `documento`.

Os notebooks disponibilizados pelo professor foram executados em 24/05/2026, a partir do ambiente virtual `.venv`, na ordem solicitada:

1. `1_1_Segmentacao_Limpeza_v1.ipynb`
2. `1_2_GerarPOS_v1.ipynb`
3. `1_3_NER_spaCy_v1.ipynb`
4. `2_1_AnaliseDados_v1.ipynb`

Os arquivos gerados ficam na pasta `data/`. Os novos arquivos adicionados ao repositório foram:

| Arquivo | Descrição |
|---|---|
| `0_0_PrepararDocumentosCSV_trainjson_v1.ipynb` | Notebook criado para adaptar o dataset escolhido ao formato esperado pela sequência principal. |
| `data/dataset.csv` | Documentos limpos e segmentados em sentenças. |
| `data/datasetpos.csv` | Saída de POS Tagging gerada com spaCy. |
| `data/datasetner.csv` | Saída de NER gerada com spaCy. |
| `relatorio_ad1_export/2_1_AnaliseDados_v1.md` | Export completo do notebook de análise com tabelas e gráficos. |
| `relatorio_ad1_export/2_1_AnaliseDados_v1_files/` | Imagens PNG geradas pelo notebook de análise. |

Além deste documento de relatório da Avaliação à Distância 1.

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

### Por documento

![Boxplot das estatísticas gerais por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_122_2.png)

![Boxplot de POS Tagging por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_127_1.png)

![Distribuição de POS Tagging por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_129_0.png)

![Boxplot de NER por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_134_0.png)

![Distribuição de NER por documento](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_136_0.png)

### Por sentença

![Boxplot das estatísticas gerais por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_144_0.png)

![Boxplot de POS Tagging por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_148_0.png)

![Distribuição de POS Tagging por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_150_2.png)

![Gráfico POS por sentença - ADP](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_153_2.png)

![Gráfico POS por sentença - PROPN](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_154_2.png)

![Gráfico POS por sentença - NOUN](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_155_2.png)

![Gráfico POS por sentença - VERB](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_156_2.png)

![Gráfico POS por sentença - ADJ](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_157_2.png)

![Gráfico POS por sentença - NUM](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_158_2.png)

![Boxplot de NER por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_162_0.png)

![Distribuição de NER por sentença](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_164_2.png)

![Gráfico NER por sentença - LOC](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_167_2.png)

![Gráfico NER por sentença - ORG](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_168_2.png)

![Gráfico NER por sentença - MISC](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_169_2.png)

### Distribuições por documento

![Quantidade de documentos por quantidade de sentenças](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_183_2.png)

![Quantidade de documentos por quantidade de palavras](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_185_2.png)

![Quantidade de documentos por quantidade de tokens](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_187_2.png)

![Quantidade de documentos por quantidade de palavras sem stopwords](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_189_2.png)

![Quantidade de documentos por quantidade de locuções verbais](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_191_2.png)

![Quantidade de documentos por quantidade de verbos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_193_2.png)

![Quantidade de documentos por quantidade de verbos e auxiliares](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_195_2.png)

![Quantidade de documentos por quantidade de substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_197_2.png)

![Quantidade de documentos por quantidade de auxiliares e substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_199_2.png)

![Quantidade de documentos por quantidade de entidades reconhecidas](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_201_2.png)

![Distribuição do comprimento dos documentos tokenizados](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_204_0.png)

![Histograma do comprimento dos documentos tokenizados](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_205_0.png)

### Distribuições por sentença

![Quantidade de sentenças por quantidade de palavras](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_208_2.png)

![Quantidade de sentenças por quantidade de tokens](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_210_2.png)

![Quantidade de sentenças por quantidade de palavras sem stopwords](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_212_2.png)

![Quantidade de sentenças por quantidade de locuções verbais](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_214_2.png)

![Quantidade de sentenças por quantidade de verbos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_216_2.png)

![Quantidade de sentenças por quantidade de verbos e auxiliares](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_218_2.png)

![Quantidade de sentenças por quantidade de substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_220_2.png)

![Quantidade de sentenças por quantidade de auxiliares e substantivos](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_222_2.png)

![Quantidade de sentenças por quantidade de entidades](relatorio_ad1_export/2_1_AnaliseDados_v1_files/2_1_AnaliseDados_v1_224_2.png)