"""Live, unfiltered source snapshots and filter-keyed PDF results (PDF endpoint only)."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from uuid import uuid4

from backend.FUENTES.calidad.read_snapshot import ReadSnapshot, reading
from backend.FUENTES.calidad.google_sheets import leer_encabezados
from backend.GATEWAY import calidad_api
from backend.GATEWAY.institutional_services_api import _fuente_base as _fuente
from backend.SERVICES.calidad.institutional_services import clave, nombre_desde_encabezado
from backend.SERVICES.calidad.normalizer import cargar_personas_evaluadas


@dataclass
class SourceSnapshot:
    key: tuple
    transport: ReadSnapshot
    current: tuple
    historical: tuple
    expires: float
    identifier: str = field(default_factory=lambda: uuid4().hex[:12])
    # Immutable bytes + filename, keyed by ALL report filters, never by source alone.
    results: dict = field(default_factory=dict)


@dataclass
class SnapshotAccess:
    snapshot: SourceSnapshot
    hit: bool
    initial_reads: int


_lock = threading.RLock()
_snapshots = {}


def clear_snapshots():
    with _lock:
        _snapshots.clear()


def _current_labels(rows, mapping, source, period):
    """Same institutional label resolution used by the current section endpoint."""
    generic = [r for r in mapping if r.get("PREGUNTA_CANONICA_ID") == "SER_PARAMEDICO"
               and clave(r.get("TEXTO_CORTO")).startswith("servicio institucional ")]
    if not generic:
        return rows, mapping
    source_id = calidad_api._CONFIG_MODALIDADES_2026[source]["fuente_id"]
    state = calidad_api.repository.obtener_estado_fuente(source_id)
    headers = leer_encabezados(str(state["SPREADSHEET_ID"]).strip(), str(state["HOJA_RESPUESTAS"]).strip())
    labels = {}
    for definition in generic:
        column = 0
        for letter in str(definition["COLUMNA_ORIGEN"]).strip().upper():
            column = column * 26 + ord(letter) - ord("A") + 1
        if 0 < column <= len(headers):
            labels[definition["PREGUNTA_ID"]] = nombre_desde_encabezado(definition, headers[column - 1])
    return rows, [labels.get(r.get("PREGUNTA_ID"), r) for r in mapping]


def select_service(dataset, service):
    rows, mapping = dataset
    if not service:
        return rows, mapping
    key = clave(calidad_api._normalizar_servicio_filtro(service))
    return [r for r in rows if clave(calidad_api._normalizar_servicio_filtro(r.get("SERVICIO_PROGRAMA"))) == key], mapping


@contextmanager
def source_snapshot(period="2026", source="Escolarizadas"):
    # Serialize source construction and use of its HTTP session. Concurrent PDFs
    # share one successful snapshot, never a partial/failed load or shared filter state.
    with _lock:
        key = (period, source, "2025")
        snapshot = _snapshots.get(key)
        hit = snapshot is not None and time.monotonic() < snapshot.expires
        if not hit:
            _snapshots.pop(key, None)
            transport = ReadSnapshot()
            with reading(transport):
                current = _fuente(period, source, "")
                historical = _fuente("2025", source, "")
                current = _current_labels(*current, source, period)
                # Prime the same catalog loader the unchanged KPI engine uses.
                # Its two year-specific views share one raw MASTER read.
                for rows, _ in (current, historical):
                    sources = {r["FUENTE_ID"] for r in rows if r.get("FUENTE_ID")}
                    years = {int(r["ANIO"]) for r in rows if r.get("ANIO") not in (None, "")}
                    if len(sources) == len(years) == 1:
                        cargar_personas_evaluadas(next(iter(sources)), next(iter(years)))
            now = time.monotonic()
            # Do not extend the freshness window of the existing normalized cache.
            expiry = min([now + calidad_api._CACHE_TTL_SECONDS] + [
                value for year in (period, "2025")
                if (value := calidad_api._cache_expira.get(f"{year}:{source}")) is not None
            ])
            snapshot = SourceSnapshot(key, transport, current, historical, expiry)
            _snapshots[key] = snapshot
            initial_reads = 0
        else:
            initial_reads = snapshot.transport.network_requests
        with reading(snapshot.transport):
            yield SnapshotAccess(snapshot, hit, initial_reads)
