"""Request-local Sheets memoization, enabled only while building a PDF snapshot.

Outside that context the client executes gspread's original transport unchanged.
No worksheet values, filters, normalization or KPI functions are altered here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import time

from gspread.exceptions import APIError
from gspread.http_client import HTTPClient

logger = logging.getLogger(__name__)


@dataclass
class ReadSnapshot:
    responses: dict = field(default_factory=dict)
    spreadsheets: dict = field(default_factory=dict)
    client: object = None
    network_requests: int = 0
    cache_hits: int = 0
    opens: int = 0
    retries: int = 0


_active: ContextVar[ReadSnapshot | None] = ContextVar("calidad_pdf_read_snapshot", default=None)


def active_snapshot():
    return _active.get()


@contextmanager
def reading(snapshot):
    token = _active.set(snapshot)
    try:
        yield snapshot
    finally:
        _active.reset(token)


class SnapshotHTTPClient(HTTPClient):
    def request(self, method, endpoint, params=None, data=None, json=None, files=None, headers=None):
        snapshot = active_snapshot()
        kwargs = dict(params=params, data=data, json=json, files=files, headers=headers)
        if snapshot is None or method.lower() != "get" or data is not None or json is not None or files is not None:
            return super().request(method, endpoint, **kwargs)
        auth = getattr(self, "auth", None)
        identity = (getattr(auth, "service_account_email", None) or id(self.session), tuple(getattr(auth, "scopes", None) or ()))
        key = (identity, endpoint, _parameters(params), _parameters(headers))
        if key in snapshot.responses:
            snapshot.cache_hits += 1
            return snapshot.responses[key]
        # One retry only, only for 429. A second failure propagates immediately.
        for attempt in range(2):
            snapshot.network_requests += 1
            try:
                response = super().request(method, endpoint, **kwargs)
            except APIError as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "code", None)
                if code != 429 or attempt:
                    raise
                snapshot.retries += 1
                logger.warning("PDF Sheets quota reached; one retry after 2 seconds")
                time.sleep(2)
            else:
                snapshot.responses[key] = response
                return response


def _parameters(value):
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
