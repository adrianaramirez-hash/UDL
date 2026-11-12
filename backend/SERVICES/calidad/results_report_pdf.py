"""Paginated vector PDF; no scoring, matching, comments or export side effects."""
from io import BytesIO
from math import ceil
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

INK = colors.HexColor("#172b3a")
MUTED = colors.HexColor("#566776")
TEAL = colors.HexColor("#076579")
OLD = colors.HexColor("#96a7ba")
PALE = colors.HexColor("#eef2f5")
RULE = colors.HexColor("#d5dfe5")
WIDTH = A4[0] - 92
BODY = ParagraphStyle("report-body", fontName="Helvetica", fontSize=9, leading=13, textColor=INK)
SMALL = ParagraphStyle("report-small", parent=BODY, fontSize=8, leading=11, textColor=MUTED)
TITLE = ParagraphStyle("report-module", parent=BODY, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=8)
LABEL = ParagraphStyle("report-label", parent=BODY, fontName="Helvetica-Bold", fontSize=10, leading=13)


def fmt(value, unit="pts"):
    return "—" if value is None else f"{value:.2f}{'%' if unit == 'pp' else ''}"


def paragraph(text, style=BODY):
    return Paragraph(escape(str(text)), style)


def text(c, value, x, y, size=9, bold=False, color=INK, right=False):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    (c.drawRightString if right else c.drawString)(x, y, str(value))


def bar(c, value, y, width, year="2026", unit="pts"):
    text(c, year, 0, y, 8, color=MUTED)
    x, usable = 36, width - 91
    c.setFillColor(PALE)
    c.roundRect(x, y - 1, usable, 7, 3, stroke=0, fill=1)
    c.setFillColor(OLD if year == "2025" else TEAL)
    fill = usable * max(0, min(100, value)) / 100
    if fill > 0:
        c.roundRect(x, y - 1, fill, 7, min(3, fill / 2), stroke=0, fill=1)
    text(c, fmt(value, unit), width, y - 1, 8.5, right=True)


def delta(c, value, unit, x, y, neutral=False):
    if value is None:
        return
    color = MUTED if neutral or value == 0 else colors.HexColor("#087564" if value > 0 else "#a42c3c")
    label = "Estable" if value == 0 else f"{'+' if value > 0 else '-'}{abs(value):.2f} {unit}"
    label_width = stringWidth(label, "Helvetica-Bold", 9)
    text(c, label, x, y, 9, True, color, right=True)
    ax = x - label_width - 11
    c.setStrokeColor(color)
    c.setLineWidth(.9)
    if value == 0:
        c.line(ax - 3, y + 3, ax + 3, y + 3)
        c.line(ax, y + 6, ax + 3, y + 3)
        c.line(ax, y, ax + 3, y + 3)
    else:
        direction = 1 if value > 0 else -1
        end = y + 3 + 3 * direction
        c.line(ax, y + 3 - 3 * direction, ax, end)
        c.line(ax - 2.5, end - 2.5 * direction, ax, end)
        c.line(ax + 2.5, end - 2.5 * direction, ax, end)


class Metric(Flowable):
    def __init__(self, metric, current_only=False, compact=False):
        super().__init__()
        self.metric, self.current_only, self.compact = metric, current_only, compact

    def wrap(self, width, height):
        self.width = width
        self.heading = paragraph(self.metric["name"], LABEL)
        _, h = self.heading.wrap(width - 112, 100)
        self.title_height = h
        self.height = h + (43 if self.current_only else 59)
        return width, self.height

    def draw(self):
        c, m, w = self.canv, self.metric, self.width
        self.heading.drawOn(c, 0, self.height - self.title_height)
        delta(c, m.get("delta"), m["unit"], w, self.height - 10, m["id"] == "AMB_PROBLEMA")
        y = self.height - self.title_height - 13
        label = f"Resultado actual {fmt(m['value'], m['unit'])}"
        if m.get("n") is not None:
            label += f"   ·   {m['n']:,} respuestas válidas"
        text(c, label, 0, y, 8.5, color=MUTED)
        y -= 16
        if m["value"] is None:
            text(c, "Sin respuestas en este periodo", 0, y, 9, color=MUTED)
        elif m.get("previous") is not None:
            bar(c, m["previous"], y, w, "2025", m["unit"])
            bar(c, m["compared_current"], y - 16, w, "2026", m["unit"])
        else:
            bar(c, m["value"], y, w, unit=m["unit"])
            if not self.current_only:
                text(c, "Sin comparable histórico", 0, y - 16, 8, color=MUTED)
        c.setStrokeColor(RULE)
        c.setLineWidth(.4)
        c.line(0, 1, w, 1)


class Person(Flowable):
    def __init__(self, person):
        super().__init__()
        self.person = person

    def wrap(self, width, height):
        self.width = width
        self.height = 48 + len(self.person["criteria"]) * 19
        return width, self.height

    def draw(self):
        c, p, w, top = self.canv, self.person, self.width, self.height
        text(c, p["name"], 0, top - 12, 10, True)
        text(c, fmt(p["value"]), w, top - 12, 10, True, right=True)
        text(c, f"{p['surveys']} encuestas analizadas · 2026", 0, top - 25, 8, color=MUTED)
        if not p["criteria"]:
            text(c, "Sin respuestas en este periodo", 0, top - 39, 8, color=MUTED)
            return
        text(c, "Sin comparable histórico por responsable", w, top - 25, 7.5, color=MUTED, right=True)
        y = top - 42
        for m in p["criteria"]:
            label = paragraph(m["name"], SMALL)
            label.wrap(w - 213, 20)
            label.drawOn(c, 0, y - 2)
            if m["value"] is not None:
                c.saveState()
                c.translate(w - 186, 0)
                bar(c, m["value"], y, 186, year="", unit=m["unit"])
                c.restoreState()
            else:
                text(c, "Sin respuestas en este periodo", w, y, 7.5, color=MUTED, right=True)
            text(c, f"n = {m['n']}", w - 191, y, 7, color=MUTED, right=True)
            y -= 19


class Overview(Flowable):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def wrap(self, width, height):
        self.width, self.height = width, len(self.rows) * 28 + 20
        return width, self.height

    def draw(self):
        c, w, y = self.canv, self.width, self.height - 12
        for name, value, unit in self.rows:
            label = paragraph(name, SMALL)
            label.wrap(210, 30)
            label.drawOn(c, 0, y - 3)
            if value is None:
                text(c, "Detalle por criterio", w, y, 8, color=MUTED, right=True)
            else:
                c.saveState()
                c.translate(220, 0)
                bar(c, value, y, w - 220, year="", unit=unit)
                c.restoreState()
            y -= 28
        text(c, "Escala fija 0–100 · Los porcentajes se identifican con %.", 0, 0, 7.5, color=MUTED)


class NumberedCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def showPage(self):
        self.states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            text(self, "Universidad de Londres · Encuesta de Calidad", 46, 27, 7, color=MUTED)
            text(self, f"Página {self._pageNumber} de {total}", A4[0] - 46, 27, 7, color=MUTED, right=True)
            super().showPage()
        super().save()


def qualitative():
    return [KeepTogether([Spacer(1, 10), paragraph("Resumen cualitativo", LABEL),
                          paragraph("Análisis cualitativo con SIA · En construcción", SMALL)])]


def _groups(items, budget, height):
    groups, group, used = [], [], 0
    for item in items:
        amount = height(item)
        if group and used + amount > budget:
            groups.append(group)
            group, used = [], 0
        group.append(item)
        used += amount
    if group:
        groups.append(group)
    return groups or [[]]


def render_report(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=46, rightMargin=46,
                            topMargin=72, bottomMargin=48, title="Informe de resultados",
                            author="Universidad de Londres · SIA Intelligence", pageCompression=1)
    scope = f"{report['source']} · {report['career']} · {report['modality']}"

    def header(c, document):
        text(c, "INFORME DE RESULTADOS", 46, A4[1] - 28, 8, True, TEAL)
        text(c, f"Encuesta de Calidad · {report['period']}", A4[0] - 46, A4[1] - 28, 8, color=MUTED, right=True)
        heading = paragraph(scope, SMALL)
        _, h = heading.wrap(WIDTH, 28)
        heading.drawOn(c, 46, A4[1] - 35 - h)
        c.setStrokeColor(RULE)
        c.line(46, A4[1] - 61, A4[0] - 46, A4[1] - 61)

    story = [paragraph("Informe de resultados", ParagraphStyle("cover", parent=TITLE, fontSize=27, leading=32)),
             paragraph("Encuesta de Calidad · 2026", LABEL), Spacer(1, 18), paragraph("Filtros activos", LABEL),
             paragraph(f"Fuente: {report['source']} · Periodo: {report['period']}"),
             paragraph(f"Carrera: {report['career']}"), paragraph(f"Modalidad académica: {report['modality']}")]
    if report["service"]:
        story.append(paragraph(f"Programa seleccionado: {report['service']}", SMALL))
    story += [Spacer(1, 18), paragraph("Panorama general", TITLE),
              paragraph(f"Resultado general: {fmt(report['index'])}    ·    Encuestas analizadas: {report['surveys']:,}"),
              paragraph(f"Respuestas KPI válidas: {report['responses']:,}", SMALL), Spacer(1, 20), paragraph("Gráfica general de secciones", LABEL)]
    rows = [(m["name"], m["index"], "pts") for m in report["modules"] if m["id"] not in {"satisfaction", "languages", "continuing"}]
    sat = next(m for m in report["modules"] if m["id"] == "satisfaction")
    rows += [(r["name"], r["value"], r["unit"]) for r in sat["criteria"]]
    rows += [("Centro de Idiomas", None, "pts"), ("Educación Continua", None, "pts")]
    story += [Overview(rows), Spacer(1, 16), paragraph("Cada criterio utiliza únicamente sus respuestas válidas. La falta de respuestas o de un comparable no equivale a una calificación de cero.", SMALL)]

    for number, module in enumerate(report["modules"], 1):
        if "people" in module:
            groups = _groups(module["people"], 554, lambda p: 48 + 19 * len(p["criteria"]))
        else:
            parts = max(1, ceil(len(module["criteria"]) / 7))
            size = max(1, ceil(len(module["criteria"]) / parts))
            groups = [module["criteria"][i:i + size] for i in range(0, len(module["criteria"]), size)] or [[]]
        for part, group in enumerate(groups):
            share_page = module["id"] == "languages" or (module["id"] == "staff" and len(module["people"]) <= 1)
            story += [Spacer(1, 24) if share_page else PageBreak(), paragraph(f"{number:02d} · {module['name']}" + (" · continuación" if part else ""), TITLE)]
            if part == 0:
                values = []
                if module["index"] is not None or module["id"] in {"direction", "staff", "institutional", "academic", "facilities", "environment"}:
                    values.append(f"Resultado general {fmt(module['index'])}")
                if module.get("surveys") is not None:
                    values.append(f"{module['surveys']:,} encuestas analizadas")
                if module.get("responses") is not None:
                    values.append(f"{module['responses']:,} respuestas a criterios")
                story += [paragraph(" · ".join(values), SMALL), Spacer(1, 12)]
                if module.get("history"):
                    previous, now = module["history"]
                    if previous is not None and now is not None:
                        m = {"id": "global", "name": "Comparativo global comparable", "value": module["index"],
                             "unit": "pts", "n": None, "previous": previous, "compared_current": now, "delta": round(now - previous, 2)}
                        story += [Metric(m), Spacer(1, 10)]
                    elif "people" not in module:
                        story += [paragraph("Sin comparable histórico", SMALL), Spacer(1, 8)]
                if module.get("current_only"):
                    selected = [f"{key}: {value}" for key, value in module.get("local_filters", {}).items() if value]
                    if selected:
                        story.append(paragraph("Filtros del módulo · " + " · ".join(selected), SMALL))
                    usage = module["usage"]
                    label = "Uso del servicio" if module["id"] == "languages" else "Participación en Educación Continua"
                    story += [paragraph(f"{label}: {fmt(usage['si_pct'], 'pp')} Sí · {usage['total_valido']} respuestas válidas", SMALL), Spacer(1, 10)]
            story += [Person(item) if "people" in module else Metric(item, module.get("current_only", False)) for item in group]
            if "people" in module and not group:
                story.append(paragraph("Sin respuestas en este periodo", SMALL))
            if part == len(groups) - 1:
                if module.get("incidence"):
                    story += [Spacer(1, 15), paragraph("Incidencia · indicador independiente", LABEL),
                              paragraph("No forma parte del resultado general de Ambiente escolar.", SMALL), Spacer(1, 8), Metric(module["incidence"])]
                story += qualitative()
    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
