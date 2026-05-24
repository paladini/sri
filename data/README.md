# Arquivo de dados

Esta pasta guarda os arquivos de entrada e saída usados pelos notebooks da prática.

## `train.json`

O notebook [0_0_PrepararDocumentosCSV_trainjson_v1.ipynb](https://github.com/osmarbraz/sri/blob/main/0_0_PrepararDocumentosCSV_trainjson_v1.ipynb) lê este arquivo para gerar `documentos.csv`.

Você pode manter esse arquivo apenas no seu ambiente local quando quiser analisar um dataset próprio. Basta colocar o arquivo como `data/train.json` e executar o notebook 0_0 para criar o CSV consumido pelos demais notebooks.

O formato esperado é uma lista de objetos com, no mínimo, o campo `title`. O notebook também aceita um dicionário com a chave `train` ou outros formatos equivalentes em dicionário. Cada título será normalizado e gravado na coluna `documento` do arquivo final.

Exemplo:

```json
[
  {
    "date": "2024-01-01",
    "title": "Título do documento",
    "text": "Texto completo do documento"
  }
]
```

## `documentos.csv`

Este é o arquivo consumido pelos notebooks da sequência principal. Ele deve conter as colunas:

- `id`: identificador do documento.
- `documento`: texto que será processado nas etapas seguintes.

Se você gerar sua própria base, o notebook 0_0 limita a saída a 500 registros e garante no máximo 512 tokens por documento usando o tokenizer BERT PT-BR.