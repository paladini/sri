# Checklist de entrega

Este checklist resume a estrutura esperada do repositório e ajuda a conferir a entrega antes de enviar os arquivos finais.

## Estrutura

- Notebooks principais na raiz do repositório.
- Arquivos de entrada e saída tabular na pasta `data/`.
- Arquivos do Embedding Projector na pasta `projecao/`.
- Relatórios finais na pasta `output/`.

## AD1

- `data/documentos.csv` com as colunas `id` e `documento`, separadas por ponto e vírgula.
- Notebooks de segmentação, POS, NER e análise executados.
- Relatório final exportado para PDF.

## AD2

- Notebooks `3_2_1` a `3_2_4` executados.
- Arquivos `config_*.json`, `meta_*.tsv` e `records_*.tsv` atualizados em `projecao/`.
- Links do Embedding Projector conferidos com URLs públicas do repositório.
- Relatório final exportado para PDF.

## Conferências rápidas

- O repositório não deve versionar arquivos locais como `.venv/`, `tmp/`, caches Python ou screenshots intermediários.
- O dataset de entrada próprio pode permanecer fora do Git como `data/train.json`.
- Os arquivos finais devem abrir corretamente pelo GitHub e pelo visualizador de PDF.
