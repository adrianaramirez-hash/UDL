"""The PDF route is isolated from dashboard and CSV endpoints."""
import re
import unicodedata
import logging

from fastapi import APIRouter, HTTPException, Response
from gspread.exceptions import APIError
from backend.SERVICES.calidad.results_report_source import source_snapshot, select_service
from backend.SERVICES.calidad.results_report import prepare_report
from backend.SERVICES.calidad.results_report_pdf import render_report

router = APIRouter(prefix="/api/encuesta-calidad", tags=["Informe de resultados PDF"])
logger = logging.getLogger(__name__)


@router.get("/informe-resultados.pdf")
def download_report(periodo: str = "2026", modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = "", idioma: str = "", tipo_curso: str = "", participacion: str = ""):
    if periodo != "2026" or modalidad != "Escolarizadas":
        raise HTTPException(status_code=400, detail="El informe de nueve módulos corresponde a Escolarizadas, instrumento 2026.")
    try:
        with source_snapshot(periodo, modalidad) as access:
            snapshot = access.snapshot
            result_key = (periodo, modalidad, servicio, carrera, modalidad_academica, idioma, tipo_curso, participacion)
            result_hit = result_key in snapshot.results
            if result_hit:
                pdf, slug = snapshot.results[result_key]
            else:
                now, now_map = select_service(snapshot.current, servicio)
                old, old_map = select_service(snapshot.historical, servicio)
                model = prepare_report(now, now_map, old, old_map, career=carrera, modality=modalidad_academica,
                                       source=modalidad, service=servicio, language=idioma, course=tipo_curso, participation=participacion)
                pdf = render_report(model)
                slug = unicodedata.normalize("NFD", f"{model['career']}_{model['modality']}")
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", "".join(c for c in slug if not unicodedata.combining(c))).strip("_")
                snapshot.results[result_key] = (pdf, slug)
            reads = snapshot.transport.network_requests - access.initial_reads
            diagnostics = {"X-PDF-Sheets-Reads": str(reads), "X-PDF-Snapshot": snapshot.identifier,
                           "X-PDF-Snapshot-Cache": "HIT" if access.hit else "MISS", "X-PDF-Result-Cache": "HIT" if result_hit else "MISS"}
            logger.info("PDF snapshot=%s source_cache=%s result_cache=%s Sheets_reads=%d", snapshot.identifier, access.hit, result_hit, reads)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except APIError as exc:
        logger.exception("Original Sheets error during PDF: period=%s source=%s career=%s modality=%s", periodo, modalidad, carrera, modalidad_academica)
        raise HTTPException(status_code=503, detail="La fuente de datos no está disponible temporalmente. Intenta generar el informe nuevamente en un minuto.", headers={"Retry-After": "60"}) from exc
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="Informe_de_resultados_Escolarizadas_2026_{slug}.pdf"',
        "Cache-Control": "no-store",
        **diagnostics,
    })
