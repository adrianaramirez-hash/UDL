from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _texto_seguro(valor) -> str:
    if valor is None:
        return "-"

    texto = str(valor).strip()

    return texto if texto else "-"


def generar_pdf_docente(
    detalle: dict,
) -> BytesIO:
    """
    Genera el reporte PDF individual de un docente
    para el módulo Observación de Clases.
    """

    buffer = BytesIO()

    documento = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title=(
            f"Reporte de Observación - "
            f"{detalle.get('docente', 'Docente')}"
        ),
        author="SIA Intelligence - Universidad de Londres",
    )

    styles = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloSIA",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_LEFT,
        spaceAfter=5,
    )

    subtitulo = ParagraphStyle(
        "SubtituloSIA",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=14,
    )

    encabezado = ParagraphStyle(
        "EncabezadoSIA",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#075985"),
        spaceBefore=8,
        spaceAfter=8,
    )

    cuerpo = ParagraphStyle(
        "CuerpoSIA",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    cuerpo_lista = ParagraphStyle(
        "ListaSIA",
        parent=cuerpo,
        leftIndent=10,
        firstLineIndent=-5,
        spaceAfter=5,
    )

    historia = []

    historia.append(
        Paragraph(
            "SIA INTELLIGENCE · OBSERVACIÓN DE CLASES",
            subtitulo,
        )
    )

    historia.append(
        Paragraph(
            _texto_seguro(
                detalle.get("docente")
            ),
            titulo,
        )
    )

    historia.append(
        Paragraph(
            "Reporte individual de seguimiento docente",
            subtitulo,
        )
    )

    resumen_data = [
        [
            "Observaciones",
            "Promedio",
            "Clasificación",
        ],
        [
            str(
                detalle.get(
                    "observaciones",
                    0,
                )
            ),
            (
                f"{float(detalle.get('promedio', 0)):.1f}"
                if detalle.get("promedio") is not None
                else "-"
            ),
            _texto_seguro(
                detalle.get(
                    "clasificacion"
                )
            ),
        ],
    ]

    resumen_table = Table(
        resumen_data,
        colWidths=[
            5.4 * cm,
            5.4 * cm,
            5.4 * cm,
        ],
    )

    resumen_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F1F5F9"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#64748B"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, 1),
                    colors.HexColor("#0F172A"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    8,
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, 1),
                    14,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.HexColor("#E2E8F0"),
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    historia.append(
        resumen_table
    )

    historia.append(
        Spacer(
            1,
            14,
        )
    )

    # ---------------------------------------------
    # Historial
    # ---------------------------------------------

    historial = detalle.get(
        "historial",
        [],
    )

    historia.append(
        Paragraph(
            "Historial de observaciones",
            encabezado,
        )
    )

    if historial:
        historial_data = [
            [
                "Corte",
                "Servicio",
                "Asignatura",
                "Tipo",
                "Puntaje",
                "Clasificación",
            ]
        ]

        for item in historial:
            historial_data.append(
                [
                    Paragraph(
                        _texto_seguro(
                            item.get("corte")
                        ),
                        cuerpo,
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("servicio")
                        ),
                        cuerpo,
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("asignatura")
                        ),
                        cuerpo,
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("tipo")
                        ),
                        cuerpo,
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("puntaje")
                        ),
                        cuerpo,
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get(
                                "clasificacion"
                            )
                        ),
                        cuerpo,
                    ),
                ]
            )

        historial_table = Table(
            historial_data,
            repeatRows=1,
            colWidths=[
                2.2 * cm,
                3.0 * cm,
                4.1 * cm,
                3.1 * cm,
                1.5 * cm,
                2.7 * cm,
            ],
        )

        historial_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#E0F2FE"),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#075985"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.25,
                        colors.HexColor("#CBD5E1"),
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ]
            )
        )

        historia.append(
            historial_table
        )

    else:
        historia.append(
            Paragraph(
                "No hay historial registrado.",
                cuerpo,
            )
        )

    historia.append(
        Spacer(
            1,
            12,
        )
    )

    # ---------------------------------------------
    # Fortalezas
    # ---------------------------------------------

    fortalezas = detalle.get(
        "fortalezas",
        [],
    )

    historia.append(
        Paragraph(
            "Fortalezas",
            encabezado,
        )
    )

    if fortalezas:
        for item in fortalezas:
            historia.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    cuerpo_lista,
                )
            )
    else:
        historia.append(
            Paragraph(
                "Sin fortalezas registradas.",
                cuerpo,
            )
        )

    # ---------------------------------------------
    # Áreas de oportunidad
    # ---------------------------------------------

    areas = detalle.get(
        "areas_oportunidad",
        [],
    )

    historia.append(
        Paragraph(
            "Áreas de oportunidad",
            encabezado,
        )
    )

    if areas:
        for item in areas:
            historia.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    cuerpo_lista,
                )
            )
    else:
        historia.append(
            Paragraph(
                "Sin áreas de oportunidad registradas.",
                cuerpo,
            )
        )

    # ---------------------------------------------
    # Recomendaciones
    # ---------------------------------------------

    recomendaciones = detalle.get(
        "recomendaciones",
        [],
    )

    bloque_recomendaciones = [
        Paragraph(
            "Recomendaciones",
            encabezado,
        )
    ]

    if recomendaciones:
        for item in recomendaciones:
            bloque_recomendaciones.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    cuerpo_lista,
                )
            )
    else:
        bloque_recomendaciones.append(
            Paragraph(
                "Sin recomendaciones registradas.",
                cuerpo,
            )
        )

    historia.append(
        KeepTogether(
            bloque_recomendaciones
        )
    )

    historia.append(
        Spacer(
            1,
            18,
        )
    )

    historia.append(
        Paragraph(
            (
                "Documento generado automáticamente por "
                "SIA Intelligence · Universidad de Londres."
            ),
            subtitulo,
        )
    )

    documento.build(
        historia
    )

    buffer.seek(0)

    return buffer