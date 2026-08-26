from backend.CONFIG.settings import settings
from backend.FUENTES.calidad.google_sheets import (
    abrir_spreadsheet,
    crear_cliente_google_sheets,
    leer_encabezados,
)


FUENTES = {
    "ESC_MAIN": {
        "spreadsheet_id": settings.calidad_esc_main_sheet_id,
        "hoja": "Respuestas de formulario 1",
    },
    "VIR_MAIN": {
        "spreadsheet_id": settings.calidad_vir_main_sheet_id,
        "hoja": "línea y virtuales",
    },
    "PRE_MAIN": {
        "spreadsheet_id": settings.calidad_pre_main_sheet_id,
        "hoja": "Preparatoria",
    },
}


def _normalizar_texto(valor: str) -> str:
    return " ".join(
        str(valor or "").strip().split()
    )


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


def _separar_header_critico(
    valor: str,
) -> tuple[str | None, str]:
    texto = str(valor or "").strip()

    if "=" not in texto:
        return None, texto

    columna, encabezado = texto.split(
        "=",
        1,
    )

    return (
        columna.strip().upper(),
        encabezado.strip(),
    )


def cargar_firmas_activas() -> list[dict]:
    client = crear_cliente_google_sheets()

    master = abrir_spreadsheet(
        client,
        settings.calidad_master_sheet_id,
    )

    worksheet = master.worksheet(
        "14_FIRMAS_FUENTE"
    )

    registros = worksheet.get_all_records()

    return [
        fila
        for fila in registros
        if str(
            fila.get("ESTADO", "")
        ).strip().upper()
        == "ACTIVA"
    ]


def validar_firma(
    firma: dict,
) -> dict:
    fuente_id = str(
        firma["FUENTE_ID"]
    ).strip()

    if fuente_id not in FUENTES:
        return {
            "fuente_id": fuente_id,
            "ok": False,
            "errores": [
                "Fuente no configurada en backend."
            ],
        }

    configuracion = FUENTES[
        fuente_id
    ]

    encabezados = leer_encabezados(
        configuracion["spreadsheet_id"],
        configuracion["hoja"],
    )

    esperadas = int(
        firma["N_COLUMNAS_ESPERADAS"]
    )

    errores = []

    if len(encabezados) != esperadas:
        errores.append(
            "Número de columnas distinto: "
            f"esperadas={esperadas}, "
            f"actuales={len(encabezados)}"
        )

    validaciones = [
        (
            "A",
            firma.get(
                "HEADER_A",
                "",
            ),
        ),
        (
            "B",
            firma.get(
                "HEADER_B",
                "",
            ),
        ),
    ]

    for campo in [
        "HEADER_CRITICO_1",
        "HEADER_CRITICO_2",
    ]:
        columna, texto = (
            _separar_header_critico(
                firma.get(
                    campo,
                    "",
                )
            )
        )

        if columna and texto:
            validaciones.append(
                (
                    columna,
                    texto,
                )
            )

    for columna, esperado in validaciones:
        indice = _columna_a_indice(
            columna
        )

        if indice >= len(encabezados):
            errores.append(
                f"{columna}: columna inexistente."
            )
            continue

        actual = encabezados[
            indice
        ]

        if (
            _normalizar_texto(actual)
            != _normalizar_texto(
                esperado
            )
        ):
            errores.append(
                f"{columna}: encabezado distinto."
            )

    ultimo_esperado = firma.get(
        "HEADER_ULTIMO",
        "",
    )

    if (
        encabezados
        and ultimo_esperado
        and _normalizar_texto(
            encabezados[-1]
        )
        != _normalizar_texto(
            ultimo_esperado
        )
    ):
        errores.append(
            "Último encabezado distinto."
        )

    return {
        "fuente_id": fuente_id,
        "ok": len(errores) == 0,
        "columnas_esperadas": esperadas,
        "columnas_actuales": len(
            encabezados
        ),
        "errores": errores,
    }


def validar_firmas_activas() -> list[dict]:
    firmas = cargar_firmas_activas()

    return [
        validar_firma(
            firma
        )
        for firma in firmas
    ]
