from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.font_manager as fm
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "ad2_figures"
PDF_PATH = OUTPUT_DIR / "relatorio_ad2.pdf"
MD_PATH = OUTPUT_DIR / "relatorio_ad2.md"
REPO_URL = "https://github.com/paladini/sri"
RAW_BASE = "https://raw.githubusercontent.com/paladini/sri/main/projecao"


def setup_fonts() -> tuple[str, str]:
    regular_path = fm.findfont("DejaVu Sans")
    bold_path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    pdfmetrics.registerFont(TTFont("DejaVuSans", regular_path))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
    return "DejaVuSans", "DejaVuSans-Bold"


def load_records(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", dtype=np.float32)


def project(records: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(records)


def wrap(text: object, width: int = 78) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def save_projection(
    coords: np.ndarray,
    meta: pd.DataFrame,
    title: str,
    subtitle: str,
    highlight_mask: np.ndarray,
    highlight_label: str,
    out_name: str,
    color_mode: str | None = None,
) -> Path:
    palette = {
        "PROPN": "#2f6f9f",
        "NOUN": "#55a868",
        "VERB": "#c44e52",
        "ADP": "#8172b2",
        "ADJ": "#ccb974",
        "NUM": "#64b5cd",
        "ADV": "#dd8452",
        "PUNCT": "#8c8c8c",
        "OTHER": "#b8b8b8",
    }

    fig, ax = plt.subplots(figsize=(11.2, 7.0), dpi=170)
    ax.grid(True, color="#e6e2d8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fbfbf8")

    if color_mode == "pos" and "POS-Tag" in meta.columns:
        top_pos = meta["POS-Tag"].value_counts().head(7).index.tolist()
        for pos in top_pos:
            mask = (meta["POS-Tag"] == pos).to_numpy()
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=13,
                alpha=0.55,
                color=palette.get(pos, palette["OTHER"]),
                label=pos,
                linewidths=0,
            )
        other = ~meta["POS-Tag"].isin(top_pos).to_numpy()
        if other.any():
            ax.scatter(
                coords[other, 0],
                coords[other, 1],
                s=10,
                alpha=0.25,
                color=palette["OTHER"],
                label="Outras",
                linewidths=0,
            )
    elif color_mode == "granularidade" and "Granularidade" in meta.columns:
        has_token_column = "Token" in meta.columns
        labels = {
            0: ("Tokens" if has_token_column else "Sentenças", "#4c72b0"),
            1: ("Documentos", "#dd8452"),
        }
        for value, (label, color) in labels.items():
            mask = (meta["Granularidade"] == value).to_numpy()
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=14 if value == 0 else 28,
                alpha=0.52 if value == 0 else 0.86,
                color=color,
                label=label,
                linewidths=0,
            )
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=16,
            alpha=0.48,
            color="#4c72b0",
            label="Documentos",
            linewidths=0,
        )

    highlighted = np.where(highlight_mask)[0]
    if len(highlighted):
        ax.scatter(
            coords[highlighted, 0],
            coords[highlighted, 1],
            s=185,
            color="#d62728",
            marker="*",
            edgecolor="white",
            linewidth=1.0,
            label="Item destacado",
            zorder=5,
        )
        x, y = coords[highlighted[0], 0], coords[highlighted[0], 1]
        annotation = highlight_label.split("\n")[0].split(":")[0][:46]
        ax.annotate(
            annotation,
            xy=(x, y),
            xytext=(18, 18),
            textcoords="offset points",
            fontsize=9,
            color="#111111",
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.1},
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d8d2c5", "alpha": 0.96},
        )

    ax.set_title(title, loc="left", pad=14, fontweight="bold")
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#d8d2c5")

    fig.text(0.055, 0.02, "Item identificado: " + highlight_label.replace("\n", " "), fontsize=8.6)
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_figures() -> dict[str, Path | str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
        }
    )

    doc_records = load_records(ROOT / "projecao/documento/records_documento_768_base_CLS.tsv")
    doc_meta = pd.read_csv(ROOT / "projecao/documento/meta_documento_768_base_CLS.tsv", sep="\t")
    doc_coords = project(doc_records)
    doc_mask = (doc_meta["Id"].astype(str) == "1").to_numpy()
    doc_text = doc_meta.loc[doc_mask, "Documento"].iloc[0]
    doc_fig = save_projection(
        doc_coords,
        doc_meta,
        "Projeção de documentos",
        "PCA 2D sobre embeddings BERTimbau [CLS] - 500 títulos GovBR",
        doc_mask,
        "Documento 1: " + wrap(doc_text, 70),
        "ad2_projecao_documento.png",
    )

    token_records = load_records(ROOT / "projecao/token/DOALL_records_token_768_base_POOL.tsv")
    token_meta = pd.read_csv(ROOT / "projecao/token/DOALL_meta_token_768_base_POOL.tsv", sep="\t")
    token_coords = project(token_records)
    token_mask = ((token_meta["Id"].astype(str) == "1") & (token_meta["Index"] == 0)).to_numpy()
    token_name = token_meta.loc[token_mask, "Token"].iloc[0]
    token_fig = save_projection(
        token_coords,
        token_meta,
        "Projeção de tokens",
        "PCA 2D sobre embeddings BERTimbau com pooling por token - 6.123 tokens",
        token_mask,
        f"Token '{token_name}' - documento 1, índice 0",
        "ad2_projecao_token.png",
        color_mode="pos",
    )

    token_doc_records = load_records(ROOT / "projecao/token_documento/DOALL_records_token_documento_768_base_POOL.tsv")
    token_doc_meta = pd.read_csv(ROOT / "projecao/token_documento/DOALL_meta_token_documento_768_base_POOL.tsv", sep="\t")
    token_doc_coords = project(token_doc_records)
    token_doc_mask = (
        (token_doc_meta["Id"].astype(str) == "1")
        & (
            (token_doc_meta["Granularidade"] == 1)
            | ((token_doc_meta["Granularidade"] == 0) & (token_doc_meta["Index"] == 0))
        )
    ).to_numpy()
    token_doc_fig = save_projection(
        token_doc_coords,
        token_doc_meta,
        "Projeção de tokens e documento",
        "PCA 2D com tokens e documentos no mesmo espaço - 6.623 pontos",
        token_doc_mask,
        f"Token '{token_name}' e documento 1: " + wrap(doc_text, 60),
        "ad2_projecao_token_documento.png",
        color_mode="granularidade",
    )

    sent_doc_records = load_records(ROOT / "projecao/sentenca_documento/DOALL_records_sentenca_documento_768_base.tsv")
    sent_doc_meta = pd.read_csv(ROOT / "projecao/sentenca_documento/DOALL_meta_sentenca_documento_768_base.tsv", sep="\t")
    sent_col = sent_doc_meta.columns[0]
    sent_doc_coords = project(sent_doc_records)
    sent_doc_mask = (
        (sent_doc_meta["Id"].astype(str) == "1")
        & (
            (sent_doc_meta["Granularidade"] == 1)
            | ((sent_doc_meta["Granularidade"] == 0) & (sent_doc_meta["Index"] == 0))
        )
    ).to_numpy()
    sentence = sent_doc_meta[(sent_doc_meta["Id"].astype(str) == "1") & (sent_doc_meta["Granularidade"] == 0)][sent_col].iloc[0]
    sent_doc_fig = save_projection(
        sent_doc_coords,
        sent_doc_meta,
        "Projeção de sentença e documento",
        "PCA 2D com sentenças e documentos - 567 sentenças + 500 documentos",
        sent_doc_mask,
        "Sentença 1 e documento 1: " + wrap(sentence, 68),
        "ad2_projecao_sentenca_documento.png",
        color_mode="granularidade",
    )

    return {
        "doc_fig": doc_fig,
        "token_fig": token_fig,
        "token_doc_fig": token_doc_fig,
        "sent_doc_fig": sent_doc_fig,
        "doc_text": doc_text,
        "token_name": token_name,
        "sentence": sentence,
    }


def create_markdown(context: dict[str, Path | str]) -> None:
    md = f"""# Avaliação a Distância 2 - Projeção de Embeddings

**Aluno:** Fernando Paladini

## Repositório

Link do repositório pessoal no GitHub: [{REPO_URL}]({REPO_URL})

Os notebooks estão na raiz do repositório e os arquivos gerados foram organizados na pasta `projecao/`.

## Dataset e execução

Foi reutilizado o texto analisado na Avaliação a Distância 1: o dataset de notícias GovBR `divergente/noticias-govbr-ptbr-1`, adaptado para `data/documentos.csv` com 500 títulos em português. Esse corpus é diferente do CSTNews usado nos notebooks de exemplo do professor.

Os notebooks executados foram:

1. `3_2_1_GerarArquivosProjecaoEmbeddingsDocumento_v1.ipynb`
2. `3_2_2_GerarArquivosProjecaoEmbeddingsToken_v1.ipynb`
3. `3_2_3_GerarArquivosProjecaoEmbeddingsToken_Documento_v1.ipynb`
4. `3_2_4_GerarArquivosProjecaoEmbeddingsSentenca_Documento_v1.ipynb`

## Arquivos gerados

| Projeção | Registros | Metadados | Pontos |
|---|---|---:|---:|
| Documento | `projecao/documento/records_documento_768_base_CLS.tsv` | `projecao/documento/meta_documento_768_base_CLS.tsv` | 500 |
| Tokens | `projecao/token/DOALL_records_token_768_base_POOL.tsv` | `projecao/token/DOALL_meta_token_768_base_POOL.tsv` | 6.123 |
| Tokens e documento | `projecao/token_documento/DOALL_records_token_documento_768_base_POOL.tsv` | `projecao/token_documento/DOALL_meta_token_documento_768_base_POOL.tsv` | 6.623 |
| Sentença e documento | `projecao/sentenca_documento/DOALL_records_sentenca_documento_768_base.tsv` | `projecao/sentenca_documento/DOALL_meta_sentenca_documento_768_base.tsv` | 1.067 |

## Links do Embedding Projector

- Documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_documento.json
- Tokens: https://projector.tensorflow.org/?config={RAW_BASE}/config_token.json
- Tokens e documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_token_documento.json
- Sentença e documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_sentenca_documento.json

## Leitura das projeções

As projeções foram reduzidas para duas dimensões com PCA apenas para visualização, então elas não mostram toda a informação dos embeddings originais de 768 dimensões. Mesmo assim, os gráficos ajudam a observar a distribuição dos documentos, tokens e sentenças do corpus GovBR. Em todos os casos eu destaquei o mesmo documento de referência, usando o token `Brasil` e a primeira sentença/documento, para facilitar a comparação entre as quatro projeções pedidas.

## Projeções

### Documento

Documento projetado: `{context["doc_text"]}`

![Projeção de documento](ad2_figures/ad2_projecao_documento.png)

### Tokens

Token projetado: `{context["token_name"]}`, no documento 1.

![Projeção de tokens](ad2_figures/ad2_projecao_token.png)

### Tokens e documento

Token e documento projetados: `{context["token_name"]}` e documento 1.

![Projeção de tokens e documento](ad2_figures/ad2_projecao_token_documento.png)

### Sentença e documento

Sentença projetada: `{context["sentence"]}`

Documento projetado: `{context["doc_text"]}`

![Projeção de sentença e documento](ad2_figures/ad2_projecao_sentenca_documento.png)
"""
    MD_PATH.write_text(md, encoding="utf-8")


def make_styles(font: str, bold_font: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitlePT",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "HeadingPT",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=16,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#243447"),
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyPT",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=9.4,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallPT",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#404040"),
        )
    )
    styles.add(
        ParagraphStyle(
            "PathPT",
            parent=styles["BodyText"],
            fontName=font,
            fontSize=6.8,
            leading=8.2,
            wordWrap="CJK",
        )
    )
    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


def fig(path: Path, width: float = 16.8 * cm) -> Image:
    image = Image(str(path))
    image._restrictSize(width, 9.7 * cm)
    return image


def create_pdf(context: dict[str, Path | str]) -> None:
    font, bold_font = setup_fonts()
    styles = make_styles(font, bold_font)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="Avaliação a Distância 2 - Projeção de Embeddings",
        author="Fernando Paladini",
    )

    table_data = [
        ["Projeção", "Arquivo de registros", "Pontos"],
        ["Documento", Paragraph("projecao/documento/records_documento_768_base_CLS.tsv", styles["PathPT"]), "500"],
        ["Tokens", Paragraph("projecao/token/DOALL_records_token_768_base_POOL.tsv", styles["PathPT"]), "6.123"],
        [
            "Tokens e documento",
            Paragraph("projecao/token_documento/DOALL_records_token_documento_768_base_POOL.tsv", styles["PathPT"]),
            "6.623",
        ],
        [
            "Sentença e documento",
            Paragraph("projecao/sentenca_documento/DOALL_records_sentenca_documento_768_base.tsv", styles["PathPT"]),
            "1.067",
        ],
    ]
    table = Table(table_data, colWidths=[3.7 * cm, 10.6 * cm, 1.5 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfcfcf")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f2")]),
            ]
        )
    )

    story = [
        Paragraph("Avaliação a Distância 2", styles["TitlePT"]),
        Paragraph("Projeção de Embeddings - Tópicos Especiais em Computação: Semântica e Recuperação de Informação", styles["BodyPT"]),
        Paragraph("<b>Aluno:</b> Fernando Paladini", styles["BodyPT"]),
        Paragraph(f"<b>Repositório pessoal:</b> {REPO_URL}", styles["BodyPT"]),
        Paragraph("Dataset e execução", styles["HeadingPT"]),
        Paragraph(
            "Foi reutilizado o texto analisado na Avaliação a Distância 1: o dataset de notícias GovBR "
            "<i>divergente/noticias-govbr-ptbr-1</i>, adaptado para 500 títulos em português em "
            "<i>data/documentos.csv</i>. Esse corpus é diferente do CSTNews presente no exemplo original dos notebooks.",
            styles["BodyPT"],
        ),
        Paragraph(
            "Os quatro notebooks de projeção solicitados foram executados localmente com BERTimbau "
            "(neuralmind/bert-base-portuguese-cased). As células específicas de Colab, instalação interativa "
            "e TensorBoard foram neutralizadas para permitir execução reprodutível no ambiente local.",
            styles["BodyPT"],
        ),
        Paragraph("Arquivos gerados", styles["HeadingPT"]),
        table,
        Paragraph("Links do Embedding Projector", styles["HeadingPT"]),
        Paragraph(f"Documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_documento.json", styles["SmallPT"]),
        Paragraph(f"Tokens: https://projector.tensorflow.org/?config={RAW_BASE}/config_token.json", styles["SmallPT"]),
        Paragraph(f"Tokens e documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_token_documento.json", styles["SmallPT"]),
        Paragraph(f"Sentença e documento: https://projector.tensorflow.org/?config={RAW_BASE}/config_sentenca_documento.json", styles["SmallPT"]),
        Paragraph("Leitura das projeções", styles["HeadingPT"]),
        Paragraph(
            "As projeções foram reduzidas para duas dimensões com PCA apenas para visualização, então elas não mostram "
            "toda a informação dos embeddings originais de 768 dimensões. Mesmo assim, os gráficos ajudam a observar "
            "a distribuição dos documentos, tokens e sentenças do corpus GovBR. Em todos os casos eu destaquei o mesmo "
            "documento de referência, usando o token <i>Brasil</i> e a primeira sentença/documento, para facilitar a "
            "comparação entre as quatro projeções pedidas.",
            styles["BodyPT"],
        ),
        PageBreak(),
    ]

    sections = [
        (
            "Gráfico de projeção de documento",
            f"Documento projetado: {context['doc_text']}",
            context["doc_fig"],
        ),
        (
            "Gráfico de projeção de tokens",
            f"Token projetado: {context['token_name']} - documento 1, índice 0.",
            context["token_fig"],
        ),
        (
            "Gráfico de projeção de tokens e documento",
            f"Token e documento projetados: {context['token_name']} e documento 1.",
            context["token_doc_fig"],
        ),
        (
            "Gráfico de projeção de sentença e documento",
            f"Sentença projetada: {context['sentence']}<br/>Documento projetado: {context['doc_text']}",
            context["sent_doc_fig"],
        ),
    ]

    for idx, (heading, description, image_path) in enumerate(sections):
        story.append(
            KeepTogether(
                [
                    Paragraph(heading, styles["HeadingPT"]),
                    Paragraph(description, styles["BodyPT"]),
                    fig(Path(image_path)),
                ]
            )
        )
        if idx in {1}:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 8))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = build_figures()
    create_markdown(context)
    create_pdf(context)
    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
