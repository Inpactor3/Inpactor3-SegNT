"""Graficos ligeros en HTML/SVG, sin dependencias externas."""
from markupsafe import Markup

PALETA = ["#4c1d95", "#0f766e", "#b45309", "#9f1239", "#1d4ed8",
          "#6d28d9", "#047857", "#a16207", "#be123c", "#0369a1",
          "#7e22ce", "#15803d", "#c2410c", "#e11d48", "#0e7490"]


def color_grupo(nombre, grupos=None):
    if grupos:
        base = {g: PALETA[i % len(PALETA)] for i, g in enumerate(grupos)}
        return base.get(nombre, PALETA[0])
    return PALETA[sum(map(ord, str(nombre))) % len(PALETA)]


def bar_chart(items, value_fmt="{:.3f}", max_value=None):
    """items: [{label, value, sub?, color?}] -> barras horizontales."""
    if not items:
        return Markup('<p class="nota">Sin datos.</p>')
    vals = [float(i["value"]) for i in items if i.get("value") is not None]
    if not vals:
        return Markup('<p class="nota">Sin datos.</p>')
    vmax = max_value if max_value is not None else max(vals)
    vmax = vmax or 1.0
    filas = []
    for it in items:
        v = it.get("value")
        color = it.get("color") or PALETA[0]
        if v is None:
            w, txt = 0, "—"
        else:
            w = max(0.0, min(100.0, float(v) / vmax * 100))
            txt = value_fmt.format(float(v))
        sub = f'<span class="bar-sub">{it["sub"]}</span>' if it.get("sub") else ""
        filas.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{it["label"]}">{it["label"]}{sub}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{w:.1f}%;background:{color}"></div>'
            '</div>'
            f'<div class="bar-val">{txt}</div>'
            '</div>'
        )
    return Markup('<div class="barras">' + "".join(filas) + "</div>")


def heatmap(rows, cols, values, fmt=".3f", color=(76, 29, 149), vmin=None, vmax=None,
            cell_title=None):
    """values: dict (fila, col) -> numero o None. Devuelve tabla con intensidad."""
    if not rows or not cols:
        return Markup('<p class="nota">Sin datos.</p>')
    vals = [float(v) for v in values.values() if v is not None]
    if vmin is None:
        vmin = min(vals) if vals else 0.0
    if vmax is None:
        vmax = max(vals) if vals else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-9
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for r in rows:
        celdas = []
        for c in cols:
            v = values.get((r, c))
            if v is None:
                celdas.append('<td class="hm-vacia">—</td>')
                continue
            t = (float(v) - vmin) / (vmax - vmin)
            alpha = 0.08 + 0.72 * max(0.0, min(1.0, t))
            txt = format(float(v), fmt)
            dark = "color:#fff" if alpha > 0.55 else ""
            title = ""
            if cell_title:
                title = f' title="{cell_title(r, c, v)}"'
            celdas.append(
                f'<td class="hm-celda" style="background:rgba({color[0]},{color[1]},{color[2]},{alpha:.2f});{dark}"{title}>{txt}</td>'
            )
        body.append(f"<tr><th class='hm-fila'>{r}</th>{''.join(celdas)}</tr>")
    return Markup(
        '<div class="hm-wrap"><table class="heatmap"><thead><tr><th></th>'
        + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def line_svg(nombre, labels, values, color="#4c1d95", width=860, height=190, fmt=".4f"):
    """Un solo gráfico de líneas con eje Y min/max y puntos con tooltip."""
    pts = [(str(l), float(v)) for l, v in zip(labels, values) if v is not None]
    if not pts:
        return Markup('<p class="nota">Sin datos.</p>')
    pad_l, pad_r, pad_t, pad_b = 46, 14, 18, 30
    vmin = min(v for _, v in pts)
    vmax = max(v for _, v in pts)
    if vmax == vmin:
        vmax = vmin + 1e-9
    n = len(pts)
    xs = [pad_l + (i * (width - pad_l - pad_r) / max(n - 1, 1)) for i in range(n)]
    ys = [height - pad_b - ((v - vmin) / (vmax - vmin)) * (height - pad_t - pad_b)
          for _, v in pts]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    circulos = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"><title>{lab}: {fmt.format(v)}</title></circle>'
        for (lab, v), x, y in zip(pts, xs, ys)
    )
    xticks = ""
    step = max(1, n // 8)
    for i in range(0, n, step):
        xticks += f'<text x="{xs[i]:.1f}" y="{height - 8}" text-anchor="middle">{pts[i][0]}</text>'
    return Markup(
        f'<div class="linea"><div class="linea-titulo">{nombre}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#e2e8f0"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#e2e8f0"/>'
        f'<text x="4" y="{pad_t+4}">{fmt.format(vmax)}</text>'
        f'<text x="4" y="{height-pad_b}">{fmt.format(vmin)}</text>'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        f'<g fill="{color}">{circulos}</g>'
        f'<g fill="#64748b" font-size="10">{xticks}</g>'
        f'</svg></div>'
    )


def confusion_table(tp, fp, fn, tn):
    tp, fp, fn, tn = int(round(tp)), int(round(fp)), int(round(fn)), int(round(tn))
    total = max(tp + fp + fn + tn, 1)
    def pct(v):
        return f"{v/total*100:.1f} %".replace(".", ",")
    return Markup(
        '<table class="matriz"><thead><tr><th></th><th>Predicho LTR</th>'
        '<th>Predicho no-LTR</th></tr></thead><tbody>'
        f'<tr><th>Real LTR</th><td class="tp">{tp:,} <em>{pct(tp)}</em></td>'
        f'<td class="fn">{fn:,} <em>{pct(fn)}</em></td></tr>'
        f'<tr><th>Real no-LTR</th><td class="fp">{fp:,} <em>{pct(fp)}</em></td>'
        f'<td class="tn">{tn:,} <em>{pct(tn)}</em></td></tr>'
        '</tbody></table>'
    )
