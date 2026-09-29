"""
Genera un archivo .pptx con las diapositivas de la presentación SegNT.

Cada diapositiva tiene:
  - título grande
  - 3-5 bullets
  - imagen opcional (SVG convertido a PNG)

Uso:
    /home/cami/Desktop/ProyectoIntegrador/Inpactor3/.venv/bin/python \
        scripts/build_pptx.py
"""
from __future__ import annotations

from pathlib import Path

import cairosvg
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "slides_img"
PNG_DIR = ROOT / "docs" / "slides_png"
PNG_DIR.mkdir(exist_ok=True)


def svg_to_png(svg_path: Path, out_path: Path, width: int = 1600) -> Path:
    """Convierte un SVG a PNG con ancho fijo."""
    cairosvg.svg2png(url=str(svg_path), write_to=str(out_path), output_width=width)
    return out_path


SLIDES = [
    {
        "title": "Inpactor3 · SegNT",
        "subtitle": "Presencia en ventanas de 50 kb — Arabidopsis, contra Inpactor 2",
        "layout": "title",
    },
    {
        "title": "1 · Presencia en ventanas de 50 kb",
        "bullets": [
            "Arabidopsis, contra Inpactor 2",
            "La ventana mide 50 000 bases",
            "Examen: cromosoma 5, no visto en entrenamiento",
        ],
    },
    {
        "title": "2 · La ventana",
        "bullets": [
            "50 000 letras",
            "Cuatro canales: A, C, G, T",
            "La misma en Inpactor 2 y en nuestra red",
        ],
    },
    {
        "title": "3 · Inpactor 2 (la vara)",
        "bullets": [
            "Red de 50 kb: dice si hay elemento",
            "LTR_FINDER escribe inicio y fin",
            "Otra red pone el linaje",
            "109 elementos en TAIR10",
        ],
    },
    {
        "title": "4 · Nuestra red · SegNT",
        "bullets": [
            "Nucleotide Transformer, 40 M parámetros",
            "Pre-entrenado en 850 especies",
            "Una sola red: presencia + posición + linaje",
            "Cabeza pequeña de segmentación por token",
        ],
        "image": "arquitectura",
    },
    {
        "title": "5 · El examen",
        "bullets": [
            "Cromosoma 5",
            "540 ventanas",
            "23 elementos verdaderos",
            "No entra al entrenamiento",
        ],
    },
    {
        "title": "6 · Cómo se repartió",
        "bullets": [
            "Chr1, Chr2, Chr3 → entrenar (65 anotaciones)",
            "Chr4 → validar (21 anotaciones)",
            "Chr5 → examen (23 anotaciones)",
            "Mismo split que el informe interno",
        ],
        "image": "split",
    },
    {
        "title": "7 · Cómo aprende",
        "bullets": [
            "Desbalance: 65 anotaciones vs miles de vacías",
            "pos_weight=100: positivos pesan 100×",
            "Muestreo balanceado: 50% positivos por batch",
            "15 pasadas por los datos",
        ],
    },
    {
        "title": "8 · Evolución del entrenamiento",
        "bullets": [
            "Época 1:  F1 = 0.03  (dice sí a todo)",
            "Época 6:  F1 = 0.10  (empieza a discriminar)",
            "Época 11: F1 = 0.14  (mejor balance)",
            "Época 15: F1 = 0.14  (best, loss = 0.0005)",
        ],
        "image": "evolucion",
    },
    {
        "title": "9 · El resultado sobre Chr5",
        "bullets": [
            "Encontró 4.8%: 1 de 21 ventanas",
            "De lo marcado, 25% era real",
            "F1 = 8.0%",
            "IoU medio = 0 (ningún match ≥ 0.5)",
        ],
    },
    {
        "title": "10 · Comparación con el informe",
        "bullets": [
            "Aleatorio uniforme: 5%",
            "Otro transposón: 7.5%",
            "Fondo genoma real (baseline): 26.3%",
            "SegNT nuestro: 8.0%",
        ],
        "image": "comparacion",
    },
    {
        "title": "11 · Por qué no superó",
        "bullets": [
            "40 M parámetros vs 65 anotaciones train",
            "Loss cae a 0.0005: memorización",
            "Chr4 ≠ Chr5 en distribución de linajes",
            "El modelo se vuelve muy conservador",
        ],
    },
    {
        "title": "12 · Qué SÍ demuestra",
        "bullets": [
            "Pipeline reproducible en Colab (1 hora)",
            "Métricas idénticas al informe (F1 + IoU + linaje)",
            "1 acierto real: hay señal, no es azar",
            "Cambiar arquitectura no compensa falta de datos",
        ],
    },
    {
        "title": "13 · Qué falta",
        "bullets": [
            "Ampliar corpus a arroz, sorgo, maíz",
            "Congelar encoder, entrenar solo cabeza",
            "Data augmentation (reverse complement)",
            "Grid search de hiperparámetros",
        ],
    },
    {
        "title": "14 · Cierre",
        "bullets": [
            "Inpactor 2 sigue siendo la vara: 100% por definición",
            "SegNT: 8% sobre Chr5",
            "Baseline informe: 26.3%",
            "Siguiente paso: escalar los datos, no la red",
        ],
    },
]

# Colores tema
COLOR_TITLE = RGBColor(0x0F, 0x17, 0x2A)
COLOR_ACCENT = RGBColor(0x4C, 0x1D, 0x95)
COLOR_TEXT = RGBColor(0x33, 0x41, 0x55)
COLOR_MUTED = RGBColor(0x64, 0x74, 0x8B)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for i, s in enumerate(SLIDES):
        layout = prs.slide_layouts[6]  # blank
        slide = prs.slides.add_slide(layout)

        if s.get("layout") == "title":
            # Portada
            box = slide.shapes.add_textbox(
                Inches(1), Inches(2.5), Inches(11.333), Inches(1.5)
            )
            tf = box.text_frame
            tf.text = s["title"]
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.runs[0]
            run.font.size = Pt(60)
            run.font.bold = True
            run.font.color.rgb = COLOR_ACCENT

            sub_box = slide.shapes.add_textbox(
                Inches(1), Inches(4.2), Inches(11.333), Inches(1)
            )
            stf = sub_box.text_frame
            stf.text = s["subtitle"]
            sp = stf.paragraphs[0]
            sp.alignment = PP_ALIGN.CENTER
            srun = sp.runs[0]
            srun.font.size = Pt(24)
            srun.font.color.rgb = COLOR_TEXT
            continue

        # Título
        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.9)
        )
        tf = title_box.text_frame
        tf.text = s["title"]
        p = tf.paragraphs[0]
        run = p.runs[0]
        run.font.size = Pt(36)
        run.font.bold = True
        run.font.color.rgb = COLOR_TITLE

        # Bullets
        img_key = s.get("image")
        bullet_w = 6.0 if img_key else 12.0
        bullet_box = slide.shapes.add_textbox(
            Inches(0.7), Inches(1.5), Inches(bullet_w), Inches(5.5)
        )
        btf = bullet_box.text_frame
        btf.word_wrap = True
        for j, b in enumerate(s["bullets"]):
            para = btf.paragraphs[0] if j == 0 else btf.add_paragraph()
            para.text = "•  " + b
            para.space_before = Pt(12)
            r = para.runs[0]
            r.font.size = Pt(22)
            r.font.color.rgb = COLOR_TEXT

        # Imagen (si hay)
        if img_key:
            svg_path = IMG_DIR / f"{img_key}.svg"
            png_path = PNG_DIR / f"{img_key}.png"
            if not png_path.exists():
                svg_to_png(svg_path, png_path, width=1600)
            slide.shapes.add_picture(
                str(png_path), Inches(6.8), Inches(1.5),
                width=Inches(6.3),
            )

    out = ROOT / "docs" / "presentacion_segnt.pptx"
    prs.save(out)
    print(f"[ok] {out}  ({out.stat().st_size / 1024:.0f} KB · {len(SLIDES)} diapositivas)")


if __name__ == "__main__":
    build()
