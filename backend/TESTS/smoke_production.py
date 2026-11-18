"""Comprueba el arranque local con las dependencias de producción y sin secretos."""

import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

import fastapi

from backend.SERVICES.observation.observation_pdf import generar_pdf_docente


def check_gateway(root: Path) -> None:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("AUTH_", "GOOGLE_", "ACCESOS_", "CALIDAD_", "SIA_AI_"))
        and name != "OC_SHEET_URL"
    }
    env["PYTHON_DOTENV_DISABLED"] = "1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    command = [
        sys.executable, "-m", "uvicorn", "backend.GATEWAY.main:app",
        "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--no-access-log",
    ]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with opener.open(f"{url}/health", timeout=2) as response:
                    if response.status != 200 or json.load(response) != {"status": "healthy"}:
                        raise RuntimeError("GET /health no devolvió la respuesta esperada")
                break
            except HTTPError as exc:
                raise RuntimeError(f"GET /health respondió {exc.code}") from None
            except URLError:
                time.sleep(0.25)
        else:
            raise RuntimeError("El gateway no arrancó")
        print("GET /health -> 200")

        for route, expected in {
            "/api/auth/me": 401,
            "/api/encuesta-calidad/preview/resumen": 401,
            "/api/observacion/resumen": 401,
            "/api/auth/login": 503,
        }.items():
            try:
                with opener.open(f"{url}{route}", timeout=3) as response:
                    status = response.status
            except HTTPError as exc:
                status = exc.code
                exc.close()
            if status != expected:
                raise RuntimeError(f"{route} respondió {status}, se esperaba {expected}")
            print(f"{route} -> {status}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if (root / ".env").exists() or (root / "credentials").exists():
        raise SystemExit("Ejecuta esta prueba desde una copia limpia del repositorio, sin .env ni credentials")
    if sys.version_info[:2] != (3, 12) or sqlite3.sqlite_version_info < (3, 35):
        raise SystemExit("Se requiere Python 3.12 con SQLite 3.35 o posterior")
    print(f"FastAPI {fastapi.__version__} / SQLite {sqlite3.sqlite_version}")
    check_gateway(root)
    pdf = generar_pdf_docente({"docente": "Docente de prueba", "observaciones": 0}).getvalue()
    if not pdf.startswith(b"%PDF-") or not pdf.rstrip().endswith(b"%%EOF"):
        raise SystemExit("El renderer no generó un PDF válido")
    print("PDF -> OK")


if __name__ == "__main__":
    main()
