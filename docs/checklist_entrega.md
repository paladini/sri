# Checklist de entrega

Este checklist resume a estrutura esperada do repositorio e ajuda a conferir a entrega antes de enviar os arquivos finais.

## Estrutura

- Notebooks principais na raiz do repositorio.
- Arquivos de entrada e saida tabular na pasta `data/`.
- Arquivos do Embedding Projector na pasta `projecao/`.
- Relatorios finais na pasta `output/`.

## AD1

- `data/documentos.csv` com as colunas `id` e `documento`, separadas por ponto e virgula.
- Notebooks de segmentacao, POS, NER e analise executados.
- Relatorio final exportado para PDF.

## AD2

- Notebooks `3_2_1` a `3_2_4` executados.
- Arquivos `config_*.json`, `meta_*.tsv` e `records_*.tsv` atualizados em `projecao/`.
- Links do Embedding Projector conferidos com URLs publicas do repositorio.
- Relatorio final exportado para PDF.

## Conferencias rapidas

- O repositorio nao deve versionar arquivos locais como `.venv/`, `tmp/`, caches Python ou screenshots intermediarios.
- O dataset de entrada proprio pode permanecer fora do Git como `data/train.json`.
- Os arquivos finais devem abrir corretamente pelo GitHub e pelo visualizador de PDF.
