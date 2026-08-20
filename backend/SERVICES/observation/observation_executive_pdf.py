from datetime import datetime
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


def _crear_estilos():
    styles = getSampleStyleSheet()

    return {
        "titulo": ParagraphStyle(
            "TituloEjecutivoSIA",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitulo": ParagraphStyle(
            "SubtituloEjecutivoSIA",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12,
        ),
        "encabezado": ParagraphStyle(
            "EncabezadoEjecutivoSIA",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#075985"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "cuerpo": ParagraphStyle(
            "CuerpoEjecutivoSIA",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#334155"),
        ),
        "lista": ParagraphStyle(
            "ListaEjecutivoSIA",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            leftIndent=10,
            firstLineIndent=-5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=5,
        ),
        "pequeno": ParagraphStyle(
            "PequenoEjecutivoSIA",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#475569"),
        ),
    }


def generar_pdf_ejecutivo_observacion(
    kpis: dict,
    insights: dict,
    casos_prioritarios: list[dict],
    filtros: dict,
) -> BytesIO:
    """
    Genera el reporte ejecutivo PDF del módulo
    Observación de Clases.

    Incluye:
    - filtros aplicados
    - KPIs
    - resumen ejecutivo
    - fortalezas
    - alertas
    - recomendaciones
    - casos prioritarios
    """

    buffer = BytesIO()

    documento = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Reporte Ejecutivo - Observación de Clases",
        author="SIA Intelligence - Universidad de Londres",
    )

    estilos = _crear_estilos()

    historia = []

    # -------------------------------------------------
    # Encabezado
    # -------------------------------------------------

    historia.append(
        Paragraph(
            "SIA INTELLIGENCE · CALIDAD ACADÉMICA",
            estilos["subtitulo"],
        )
    )

    historia.append(
        Paragraph(
            "Reporte Ejecutivo de Observación de Clases",
            estilos["titulo"],
        )
    )

    historia.append(
        Paragraph(
            (
                "Documento institucional para análisis, seguimiento "
                "y toma de decisiones."
            ),
            estilos["subtitulo"],
        )
    )

    fecha_generacion = datetime.now().strftime(
        "%d/%m/%Y %H:%M"
    )

    historia.append(
        Paragraph(
            f"Generado: {fecha_generacion}",
            estilos["subtitulo"],
        )
    )

    # -------------------------------------------------
    # Filtros aplicados
    # -------------------------------------------------

    historia.append(
        Paragraph(
            "Alcance del reporte",
            estilos["encabezado"],
        )
    )

    filtros_data = [
        [
            "Corte",
            "Servicio",
            "Tipo de observación",
        ],
        [
            _texto_seguro(
                filtros.get(
                    "corte",
                    "Todos los cortes",
                )
            ),
            _texto_seguro(
                filtros.get(
                    "servicio",
                    "Todos los servicios",
                )
            ),
            _texto_seguro(
                filtros.get(
                    "tipo",
                    "Todos los tipos",
                )
            ),
        ],
    ]

    filtros_table = Table(
        filtros_data,
        colWidths=[
            5.4 * cm,
            5.4 * cm,
            5.4 * cm,
        ],
    )

    filtros_table.setStyle(
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
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    historia.append(
        filtros_table
    )

    historia.append(
        Spacer(
            1,
            14,
        )
    )

    # -------------------------------------------------
    # KPIs
    # -------------------------------------------------

    historia.append(
        Paragraph(
            "Indicadores clave",
            estilos["encabezado"],
        )
    )

    kpis_data = [
        [
            "Observaciones",
            "Promedio",
            "Consolidado",
            "En proceso",
            "No consolidado",
        ],
        [
            str(
                kpis.get(
                    "observaciones",
                    0,
                )
            ),
            (
                f"{float(kpis.get('promedio', 0)):.1f}"
                if kpis.get("promedio") is not None
                else "-"
            ),
            str(
                kpis.get(
                    "consolidado",
                    0,
                )
            ),
            str(
                kpis.get(
                    "en_proceso",
                    0,
                )
            ),
            str(
                kpis.get(
                    "no_consolidado",
                    0,
                )
            ),
        ],
    ]

    kpis_table = Table(
        kpis_data,
        colWidths=[
            3.25 * cm,
            3.25 * cm,
            3.25 * cm,
            3.25 * cm,
            3.25 * cm,
        ],
    )

    kpis_table.setStyle(
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
                    7.5,
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
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    colors.HexColor("#CBD5E1"),
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
        kpis_table
    )

    historia.append(
        Spacer(
            1,
            14,
        )
    )

    # -------------------------------------------------
    # Resumen del Asesor SIA
    # -------------------------------------------------

    historia.append(
        Paragraph(
            "Lectura ejecutiva del Asesor SIA",
            estilos["encabezado"],
        )
    )

    historia.append(
        Paragraph(
            _texto_seguro(
                insights.get(
                    "resumen",
                    "Sin análisis disponible.",
                )
            ),
            estilos["cuerpo"],
        )
    )

    historia.append(
        Spacer(
            1,
            10,
        )
    )

    # -------------------------------------------------
    # Fortalezas
    # -------------------------------------------------

    fortalezas = insights.get(
        "fortalezas",
        [],
    )

    bloque_fortalezas = [
        Paragraph(
            "Fortalezas",
            estilos["encabezado"],
        )
    ]

    if fortalezas:
        for item in fortalezas:
            bloque_fortalezas.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    estilos["lista"],
                )
            )
    else:
        bloque_fortalezas.append(
            Paragraph(
                "No se identificaron fortalezas destacadas.",
                estilos["cuerpo"],
            )
        )

    historia.append(
        KeepTogether(
            bloque_fortalezas
        )
    )

    # -------------------------------------------------
    # Alertas
    # -------------------------------------------------

    alertas = insights.get(
        "alertas",
        [],
    )

    bloque_alertas = [
        Paragraph(
            "Alertas y focos de atención",
            estilos["encabezado"],
        )
    ]

    if alertas:
        for item in alertas:
            bloque_alertas.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    estilos["lista"],
                )
            )
    else:
        bloque_alertas.append(
            Paragraph(
                "No se identificaron alertas relevantes.",
                estilos["cuerpo"],
            )
        )

    historia.append(
        KeepTogether(
            bloque_alertas
        )
    )

    # -------------------------------------------------
    # Recomendaciones
    # -------------------------------------------------

    recomendaciones = insights.get(
        "recomendaciones",
        [],
    )

    bloque_recomendaciones = [
        Paragraph(
            "Recomendaciones",
            estilos["encabezado"],
        )
    ]

    if recomendaciones:
        for item in recomendaciones:
            bloque_recomendaciones.append(
                Paragraph(
                    f"• {_texto_seguro(item)}",
                    estilos["lista"],
                )
            )
    else:
        bloque_recomendaciones.append(
            Paragraph(
                "No hay recomendaciones adicionales.",
                estilos["cuerpo"],
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
            12,
        )
    )

    # -------------------------------------------------
    # Casos prioritarios
    # -------------------------------------------------

    historia.append(
        Paragraph(
            "Seguimiento prioritario",
            estilos["encabezado"],
        )
    )

    if casos_prioritarios:
        casos_data = [
            [
                "Docente",
                "Servicio",
                "Tipo",
                "Promedio",
                "Observaciones",
                "Clasificación",
            ]
        ]

        for item in casos_prioritarios:
            casos_data.append(
                [
                    Paragraph(
                        _texto_seguro(
                            item.get("docente")
                        ),
                        estilos["pequeno"],
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("servicio")
                        ),
                        estilos["pequeno"],
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("tipo")
                        ),
                        estilos["pequeno"],
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("promedio")
                        ),
                        estilos["pequeno"],
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("observaciones")
                        ),
                        estilos["pequeno"],
                    ),
                    Paragraph(
                        _texto_seguro(
                            item.get("clasificacion")
                        ),
                        estilos["pequeno"],
                    ),
                ]
            )

        casos_table = Table(
            casos_data,
            repeatRows=1,
            colWidths=[
                3.5 * cm,
                3.4 * cm,
                3.2 * cm,
                1.8 * cm,
                2.1 * cm,
                2.7 * cm,
            ],
        )

        casos_table.setStyle(
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
                        colors.HexColor("#475569"),
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
                        7.2,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "ALIGN",
                        (3, 1),
                        (4, -1),
                        "CENTER",
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
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        historia.append(
            casos_table
        )

    else:
        historia.append(
            Paragraph(
                (
                    "No se identificaron casos prioritarios "
                    "para los filtros seleccionados."
                ),
                estilos["cuerpo"],
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
            estilos["subtitulo"],
        )
    )

    documento.build(
        historia
    )

    buffer.seek(0)

    return buffer