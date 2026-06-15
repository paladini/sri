# Avaliação a Distância 2 - Projeção de Embeddings

**Aluno:** Fernando Paladini

## Enunciado da AD2

Com o texto analisado na Avaliação a Distância 1, realizar a projeção de embeddings dos notebooks:

1. [`3_2_1_GerarArquivosProjecaoEmbeddingsDocumento_v1.ipynb`](https://github.com/paladini/sri/blob/main/3_2_1_GerarArquivosProjecaoEmbeddingsDocumento_v1.ipynb)
2. [`3_2_2_GerarArquivosProjecaoEmbeddingsToken_v1.ipynb`](https://github.com/paladini/sri/blob/main/3_2_2_GerarArquivosProjecaoEmbeddingsToken_v1.ipynb)
3. [`3_2_3_GerarArquivosProjecaoEmbeddingsToken_Documento_v1.ipynb`](https://github.com/paladini/sri/blob/main/3_2_3_GerarArquivosProjecaoEmbeddingsToken_Documento_v1.ipynb)
4. [`3_2_4_GerarArquivosProjecaoEmbeddingsSentenca_Documento_v1.ipynb`](https://github.com/paladini/sri/blob/main/3_2_4_GerarArquivosProjecaoEmbeddingsSentenca_Documento_v1.ipynb)

Os notebooks originais estão disponíveis em [https://github.com/osmarbraz/sri](https://github.com/osmarbraz/sri).

O PDF deve conter printscreens das projeções de documento, tokens, tokens e documento, sentença e documento, identificando os itens projetados, além do link do repositório pessoal com notebooks na raiz e arquivos gerados na pasta `projecao/`. O dataset não deve replicar o exemplo dos notebooks.

## Repositório

Link do repositório pessoal no GitHub: [https://github.com/paladini/sri](https://github.com/paladini/sri)

Os notebooks estão na raiz do repositório e os arquivos gerados foram organizados na pasta `projecao/`.

## Dataset e execução

Foi reutilizado o texto analisado na Avaliação a Distância 1: o dataset de notícias GovBR `divergente/noticias-govbr-ptbr-1`, adaptado para `data/documentos.csv` com 500 documentos curtos em português. Esse corpus é diferente do CSTNews usado nos notebooks de exemplo do professor.

Considerando a observação recebida na AD1, mantive o mesmo corpus para preservar a continuidade exigida na AD2, mas registro que documentos com mais de uma sentença tendem a oferecer mais contexto semântico para análises futuras.

Os quatro notebooks solicitados foram executados e geraram os arquivos na pasta `projecao/`.

## Arquivos gerados

| Projeção | Registros | Metadados | Pontos |
|---|---|---:|---:|
| Documento | [`projecao/documento/records_documento_768_base_CLS.tsv`](https://github.com/paladini/sri/blob/main/projecao/documento/records_documento_768_base_CLS.tsv) | [`projecao/documento/meta_documento_768_base_CLS.tsv`](https://github.com/paladini/sri/blob/main/projecao/documento/meta_documento_768_base_CLS.tsv) | 500 |
| Tokens | [`projecao/token/DOALL_records_token_768_base_POOL.tsv`](https://github.com/paladini/sri/blob/main/projecao/token/DOALL_records_token_768_base_POOL.tsv) | [`projecao/token/DOALL_meta_token_768_base_POOL.tsv`](https://github.com/paladini/sri/blob/main/projecao/token/DOALL_meta_token_768_base_POOL.tsv) | 6.123 |
| Tokens e documento | [`projecao/token_documento/DOALL_records_token_documento_768_base_POOL.tsv`](https://github.com/paladini/sri/blob/main/projecao/token_documento/DOALL_records_token_documento_768_base_POOL.tsv) | [`projecao/token_documento/DOALL_meta_token_documento_768_base_POOL.tsv`](https://github.com/paladini/sri/blob/main/projecao/token_documento/DOALL_meta_token_documento_768_base_POOL.tsv) | 6.623 |
| Sentença e documento | [`projecao/sentenca_documento/DOALL_records_sentenca_documento_768_base.tsv`](https://github.com/paladini/sri/blob/main/projecao/sentenca_documento/DOALL_records_sentenca_documento_768_base.tsv) | [`projecao/sentenca_documento/DOALL_meta_sentenca_documento_768_base.tsv`](https://github.com/paladini/sri/blob/main/projecao/sentenca_documento/DOALL_meta_sentenca_documento_768_base.tsv) | 1.067 |

## Links do Embedding Projector

- [Documento](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/paladini/sri/main/projecao/config_documento.json)
- [Tokens](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/paladini/sri/main/projecao/config_token.json)
- [Tokens e documento](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/paladini/sri/main/projecao/config_token_documento.json)
- [Sentença e documento](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/paladini/sri/main/projecao/config_sentenca_documento.json)

## Printscreens do Embedding Projector

As imagens abaixo são printscreens do TensorBoard Embedding Projector usando PCA em duas dimensões. A busca por `Brasil` foi usada para evidenciar o documento de referência, tokens associados e sentenças/documentos relacionados.

### Documento

Documento projetado: `Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto`

O print mostra os 500 documentos do corpus GovBR projetados por PCA a partir dos embeddings BERTimbau. A busca por `Brasil` seleciona 14 documentos que contêm esse termo; eles aparecem destacados em vermelho e com rótulos visíveis no gráfico. O documento de referência, `Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto`, fica próximo de outros títulos sobre ações da CGU, Polícia Federal e governo federal, indicando agrupamento semântico por tema institucional.

![Printscreen da projeção de documento](ad2_screenshots/projector_documento.png)

### Tokens

Token projetado: `Brasil`, no documento 1.

O print mostra 6.123 embeddings de tokens. A projeção está configurada com `Color by: POS-Tag`, permitindo organizar os tokens por classe morfossintática, enquanto a busca por `Brasil` destaca as ocorrências do token e variações como `Brasileiro`, `Brasileiras` e `brasileira`. O token escolhido pertence ao documento 1 e aparece em uma região compartilhada por termos de mesma família lexical.

![Printscreen da projeção de tokens](ad2_screenshots/projector_token.png)

### Tokens e documento

Token e documento projetados: `Brasil` e documento 1.

O print combina tokens e documentos no mesmo espaço vetorial, totalizando 6.623 pontos. A opção `Color by: Granularidade` diferencia os tipos de ponto disponíveis na projeção, e a busca por `Brasil` evidencia tokens relacionados ao termo pesquisado. Essa visualização permite comparar a posição do token `Brasil` com documentos do corpus que tratam de assuntos próximos.

![Printscreen da projeção de tokens e documento](ad2_screenshots/projector_token_documento.png)

### Sentença e documento

Sentença projetada: `Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto`

Documento projetado: `Brasil inicia construção do 5 Plano de Ação Nacional de Governo Aberto`

O print combina sentenças e documentos em uma projeção com 1.067 pontos. A opção `Color by: Granularidade` separa sentenças e documentos, enquanto a busca por `Brasil` seleciona sentenças/documentos relacionados ao termo. A sentença/documento de referência aparece junto de outros títulos governamentais, permitindo observar a proximidade semântica entre unidades textuais de níveis diferentes.

![Printscreen da projeção de sentença e documento](ad2_screenshots/projector_sentenca_documento.png)
