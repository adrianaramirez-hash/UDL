from backend.CONFIG.settings import settings
from backend.FUENTES.calidad.google_sheets import (
    abrir_spreadsheet,
    crear_cliente_google_sheets,
)


def _normalizar_control(valor) -> str:
    import unicodedata

    texto = str(valor or "").strip()

    return "".join(
        caracter
        for caracter in unicodedata.normalize(
            "NFD",
            texto,
        )
        if unicodedata.category(caracter)
        != "Mn"
    ).upper()


def cargar_mapa_preguntas(
    fuente_id: str,
    anio: int | None = None,
) -> list[dict]:
    """
    Carga del MASTER las preguntas activas
    correspondientes a una fuente.

    Si se indica anio, conserva las preguntas
    cuya vigencia inicia en ese anio o antes.
    """
    client = crear_cliente_google_sheets()

    master = abrir_spreadsheet(
        client,
        settings.calidad_master_sheet_id,
    )

    worksheet = master.worksheet(
        "02_MAPA_PREGUNTAS"
    )

    registros = worksheet.get_all_records()

    fuente = fuente_id.strip().upper()

    filas = [
        fila
        for fila in registros
        if str(
            fila.get("FUENTE_ID", "")
        ).strip().upper()
        == fuente
        and _normalizar_control(
            fila.get("ACTIVA", "")
        )
        == "SI"
    ]

    if anio is None:
        return filas

    resultado = []

    for fila in filas:
        aplica_desde = str(
            fila.get(
                "APLICA_DESDE_ANIO",
                "",
            )
        ).strip()

        if not aplica_desde:
            resultado.append(fila)
            continue

        try:
            aplica_desde_num = int(
                float(aplica_desde)
            )
        except ValueError:
            continue

        if aplica_desde_num <= int(anio):
            resultado.append(fila)

    return resultado


def cargar_catalogo_escalas() -> list[dict]:
    """
    Carga las equivalencias oficiales de respuesta
    definidas en 01_CATALOGO_ESCALAS.
    """
    client = crear_cliente_google_sheets()

    master = abrir_spreadsheet(
        client,
        settings.calidad_master_sheet_id,
    )

    worksheet = master.worksheet(
        "01_CATALOGO_ESCALAS"
    )

    return worksheet.get_all_records()


def _columna_a_indice(columna: str) -> int:
    resultado = 0

    for caracter in columna.strip().upper():
        resultado = (
            resultado * 26
            + ord(caracter)
            - ord("A")
            + 1
        )

    return resultado - 1


def _es_si(valor) -> bool:
    return str(valor or "").strip().upper() in {
        "SÍ",
        "SI",
    }


def _crear_indice_escalas() -> dict:
    indice = {}

    for fila in cargar_catalogo_escalas():
        escala_id = str(
            fila.get("ESCALA_ID", "")
        ).strip()

        respuesta = str(
            fila.get("RESPUESTA_ORIGINAL", "")
        ).strip()

        if escala_id and respuesta:
            indice[
                (
                    escala_id,
                    respuesta,
                )
            ] = fila

    return indice


def normalizar_filas_preview(
    filas: list[list],
    fila_inicio: int,
    fuente_id: str,
    anio: int,
    periodo_id: str,
    version_instrumento: str,
) -> tuple[list[dict], list[dict]]:
    """
    Convierte filas nuevas de ESC_MAIN al formato
    analítico, sin escribir en el MASTER.

    Devuelve:
    registros_normalizados
    incidencias_qa
    """
    fuente = fuente_id.strip().upper()

    mapa = cargar_mapa_preguntas(
        fuente,
        anio,
    )

    escalas = _crear_indice_escalas()

    registros = []
    qa = []

    for desplazamiento, fila in enumerate(
        filas
    ):
        fila_origen = (
            fila_inicio
            + desplazamiento
        )

        marca_temporal = (
            str(fila[0]).strip()
            if len(fila) > 0
            else ""
        )

        servicio = ""
        modalidad_detalle = ""
        modalidad_agrupada = ""
        grado = ""
        ciclo_escolar = ""
        turno = ""
        edad = ""

        if fuente == "ESC_MAIN":
            servicio = (
                str(fila[1]).strip()
                if len(fila) > 1
                else ""
            )

            modalidades = [
                str(fila[i]).strip()
                for i in range(
                    2,
                    min(6, len(fila)),
                )
                if str(fila[i]).strip()
            ]

            modalidad_detalle = (
                modalidades[0]
                if modalidades
                else ""
            )

            modalidad_agrupada = (
                "Escolarizadas"
            )

        elif fuente == "ESC_2025_ARCHIVE":
            servicio = (
                str(fila[1]).strip()
                if len(fila) > 1
                else ""
            )

            ciclo_escolar = (
                str(fila[2]).strip()
                if len(fila) > 2
                else ""
            )

            turno = (
                str(fila[3]).strip()
                if len(fila) > 3
                else ""
            )

            edad = (
                str(fila[4]).strip()
                if len(fila) > 4
                else ""
            )

            modalidad_agrupada = (
                "Escolarizadas"
            )

            modalidad_detalle = (
                "Escolarizadas"
            )

        elif fuente == "VIR_MAIN":
            servicio = (
                str(fila[1]).strip()
                if len(fila) > 1
                else ""
            )

            modalidad_agrupada = (
                "Virtuales"
            )

            modalidad_detalle = (
                "Virtuales"
            )

        elif fuente == "PRE_MAIN":
            grado = (
                str(fila[2]).strip()
                if len(fila) > 2
                else ""
            )

            edad = (
                str(fila[3]).strip()
                if len(fila) > 3
                else ""
            )

            servicio = grado

            modalidad_agrupada = (
                "Prepa"
            )

            modalidad_detalle = (
                "Preparatoria"
            )

        else:
            raise ValueError(
                f"Fuente no soportada: {fuente}"
            )

        for pregunta in mapa:
            columna = str(
                pregunta.get(
                    "COLUMNA_ORIGEN",
                    "",
                )
            ).strip()

            if not columna:
                continue

            indice_columna = (
                _columna_a_indice(
                    columna
                )
            )

            if indice_columna >= len(
                fila
            ):
                continue

            respuesta_original = str(
                fila[
                    indice_columna
                ]
            ).strip()

            if not respuesta_original:
                continue

            escala_id = str(
                pregunta.get(
                    "ESCALA_ID",
                    "",
                )
            ).strip()

            respuesta_normalizada = (
                respuesta_original
            )

            valor_base = ""
            indice_100 = ""
            respuesta_valida = "SÍ"
            incluir_en_kpi = "NO"

            if escala_id:
                equivalencia = escalas.get(
                    (
                        escala_id,
                        respuesta_original,
                    )
                )

                if equivalencia is None:
                    qa.append(
                        {
                            "fila_origen": fila_origen,
                            "columna": columna,
                            "pregunta_id": pregunta.get(
                                "PREGUNTA_ID",
                                "",
                            ),
                            "escala_id": escala_id,
                            "respuesta": respuesta_original,
                            "error": "SCALE_VALUE_UNKNOWN",
                        }
                    )

                    respuesta_valida = "NO"
                    incluir_en_kpi = "NO"

                else:
                    respuesta_normalizada = (
                        equivalencia.get(
                            "RESPUESTA_NORMALIZADA",
                            respuesta_original,
                        )
                    )

                    valor_base = (
                        equivalencia.get(
                            "VALOR_BASE",
                            "",
                        )
                    )

                    indice_100 = (
                        equivalencia.get(
                            "INDICE_100",
                            "",
                        )
                    )

                    respuesta_valida = str(
                        equivalencia.get(
                            "ES_RESPUESTA_VALIDA",
                            "NO",
                        )
                    ).strip()

                    incluir_en_kpi = str(
                        equivalencia.get(
                            "INCLUIR_EN_KPI",
                            "NO",
                        )
                    ).strip()

            pregunta_id = str(
                pregunta.get(
                    "PREGUNTA_ID",
                    "",
                )
            ).strip()

            registros.append(
                {
                    "ID_REGISTRO": (
                        f"{fuente}-"
                        f"{anio}-"
                        f"{fila_origen:04d}-"
                        f"{pregunta_id}"
                    ),
                    "FUENTE_ID": fuente,
                    "ANIO": anio,
                    "MARCA_TEMPORAL": marca_temporal,
                    "FILA_ORIGEN": fila_origen,
                    "MODALIDAD_AGRUPADA": modalidad_agrupada,
                    "SERVICIO_PROGRAMA": servicio,
                    "MODALIDAD_DETALLE": modalidad_detalle,
                    "GRADO": grado,
                    "CICLO_ESCOLAR": ciclo_escolar,
                    "TURNO": turno,
                    "EDAD": edad,
                    "SECCION_ID": pregunta.get(
                        "SECCION_ID",
                        "",
                    ),
                    "SECCION_NOMBRE": pregunta.get(
                        "SECCION_NOMBRE",
                        "",
                    ),
                    "DIMENSION": pregunta.get(
                        "DIMENSION",
                        "",
                    ),
                    "SUBDIMENSION": pregunta.get(
                        "SUBDIMENSION",
                        "",
                    ),
                    "PREGUNTA_ID": pregunta_id,
                    "PREGUNTA_CANONICA_ID": pregunta.get(
                        "PREGUNTA_CANONICA_ID",
                        "",
                    ),
                    "TEXTO_CORTO": pregunta.get(
                        "TEXTO_CORTO",
                        "",
                    ),
                    "TIPO_DATO": pregunta.get(
                        "TIPO_DATO",
                        "",
                    ),
                    "ESCALA_ID": escala_id,
                    "PERSONA_EVALUADA": pregunta.get(
                        "PERSONA_EVALUADA",
                        "",
                    ),
                    "ROL_EVALUADO": pregunta.get(
                        "ROL_EVALUADO",
                        "",
                    ),
                    "RESPUESTA_ORIGINAL": respuesta_original,
                    "RESPUESTA_NORMALIZADA": respuesta_normalizada,
                    "VALOR_BASE": valor_base,
                    "INDICE_100": indice_100,
                    "RESPUESTA_VALIDA": respuesta_valida,
                    "INCLUIR_EN_KPI": incluir_en_kpi,
                    "ENTRA_PROMEDIO_SECCION": pregunta.get(
                        "ENTRA_PROMEDIO_SECCION",
                        "",
                    ),
                    "ENTRA_PROMEDIO_GENERAL": pregunta.get(
                        "ENTRA_PROMEDIO_GENERAL",
                        "",
                    ),
                    "COMPARABLE_HISTORICO": pregunta.get(
                        "COMPARABLE_HISTORICO",
                        "",
                    ),
                    "CONDICION_APLICACION": pregunta.get(
                        "CONDICION_APLICACION",
                        "",
                    ),
                    "OBSERVACIONES_ETL": "",
                    "PERIODO_ID": periodo_id,
                    "VERSION_INSTRUMENTO": version_instrumento,
                }
            )

    return registros, qa



def normalizar_filas_esc_preview(
    filas: list[list],
    fila_inicio: int,
) -> tuple[list[dict], list[dict]]:
    """
    Compatibilidad temporal con el preview actual
    de Escolarizadas 2026.
    """
    return normalizar_filas_preview(
        filas=filas,
        fila_inicio=fila_inicio,
        fuente_id="ESC_MAIN",
        anio=2026,
        periodo_id="ESC_2026_NUEVO",
        version_instrumento="FORM_2026",
    )
