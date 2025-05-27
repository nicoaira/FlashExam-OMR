#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera PDFs de hojas de respuestas OMR de 42 preguntas
Distribuidas en 2 columnas de 21 preguntas cada una.
Cada fila de pregunta va dentro de su propio recuadro aislado,
con un gap horizontal entre columnas, filas más altas para
ocupar el espacio vertical sobrante, y mayor separación
horizontal entre el número de pregunta y la primera burbuja.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def generar_hoja_respuestas(
    filename="hoja_respuestas_tema_A.pdf",
    tema="A"
):
    c = canvas.Canvas(filename, pagesize=A4)
    ancho, alto = A4

    # Márgenes
    m_x = 20 * mm
    m_y = 20 * mm

    # Parámetros ajustados
    row_form_h    = 8   * mm    # altura de fila del formulario
    row_q_h       = 9   * mm    # altura de cada recuadro de pregunta
    gap_cols      = 10  * mm    # separación entre las dos columnas de preguntas
    r_bubble      = 3.5 * mm    # radio de cada burbuja
    font_num_pt   = 11          # pt para número de pregunta
    font_q_pt     = 9           # pt para letra dentro de burbuja
    font_l_pt     = 10          # pt para texto general
    pt2mm = lambda pt: pt * 0.3528

    # Cálculo de columnas
    ancho_util    = ancho - 2*m_x
    ancho_col     = (ancho_util - gap_cols) / 2
    col_x = [
        m_x,
        m_x + ancho_col + gap_cols
    ]

    # Marcadores negros en las 4 esquinas
    mk   = 8 * mm
    off  = 5 * mm
    ty   = alto - mk - off
    by   = m_y + off
    for x in (m_x, ancho - m_x - mk):
        c.rect(x, ty, mk, mk, stroke=0, fill=1)
        c.rect(x, by, mk, mk, stroke=0, fill=1)

    # Encabezado
    intro = "Introducción a la Bioinformática 24/25 - Examen final - Biomedicina (UIC)"
    fecha = "Fecha: 27/05/2025"
    h1y   = alto - m_y
    c.setFont("Helvetica-Bold", 10)
    c.drawString(m_x, h1y, f"{intro}   {fecha}")

    h2y = h1y - 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(m_x, h2y, f"Hoja de Respuestas – Tema {tema}")

    # Tabla de formulario 2×2
    tx0    = m_x
    tx1    = tx0 + 50 * mm
    tx2    = ancho - m_x - mk
    tpy    = h2y - 25

    # Filas y columnas cerradas de la tabla
    for i in range(3):
        y = tpy - i*row_form_h
        c.line(tx0, y, tx2, y)
    for x in (tx0, tx1, tx2):
        c.line(x, tpy, x, tpy - 2*row_form_h)

    # Textos centrados verticalmente en celdas
    c.setFont("Helvetica", font_l_pt)
    cy0 = tpy - row_form_h/2
    y0  = cy0 - pt2mm(font_l_pt)/2 * mm
    c.drawString(tx0 + 2*mm, y0, "NOMBRE")
    cy1 = tpy - 1.5*row_form_h
    y1  = cy1 - pt2mm(font_l_pt)/2 * mm
    c.drawString(tx0 + 2*mm, y1, "DNI / NIE / PASAPORTE")

    # Etiquetas A–D sobre preguntas
    ly = tpy - 2*row_form_h - 20
    c.setFont("Helvetica", font_l_pt)
    for col in range(2):
        bx = col_x[col] + 20 * mm  # desplazamos la primera burbuja a 20mm
        for idx, opc in enumerate(("A","B","C","D")):
            x = bx + idx*15*mm
            c.drawString(x - 2*mm, ly, opc)

    # Preguntas y burbujas en recuadros aislados
    preguntas_por_col = 21
    for col in range(2):
        x0 = col_x[col]
        xi = x0 + 20 * mm  # mayor separación al primer bubble
        for i in range(preguntas_por_col):
            num = col*preguntas_por_col + i + 1
            y   = ly - (i+1)*row_q_h

            # recuadro aislado
            c.rect(x0, y - row_q_h/2, ancho_col, row_q_h, stroke=1, fill=0)

            # número centrado
            c.setFont("Helvetica", font_num_pt)
            y_num = y - pt2mm(font_num_pt)/2 * mm
            c.drawString(x0 + 2*mm, y_num, str(num))

            # burbujas dentro
            for j, opc in enumerate(("A","B","C","D")):
                xb = xi + j*15*mm
                c.circle(xb, y, r_bubble)
                c.setFont("Helvetica", font_q_pt)
                y_letra = y - pt2mm(font_q_pt)/2 * mm
                c.drawCentredString(xb, y_letra, opc)

    c.showPage()
    c.save()
    print(f"Generado: {filename}")

if __name__ == "__main__":
    for tema in ("A", "B"):
        generar_hoja_respuestas(f"hoja_respuestas_tema_{tema}.pdf", tema)