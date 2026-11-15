"""Small presentation helpers for the judge-facing notebook; no model changes."""
import html
from pathlib import Path


def table(frame, caption=None):
    """Small, accessible rich table instead of console-aligned DataFrame text."""
    from IPython.display import HTML, display
    title = f'<caption style="text-align:left;font-weight:600;padding:8px 0">{html.escape(caption)}</caption>' if caption else ''
    body = frame.to_html(index=False, escape=True, border=0)
    body = body.replace('<thead>', title + '<thead>', 1)
    css = '<style>.hc-table table{border-collapse:collapse;font-size:14px;line-height:1.5}.hc-table th,.hc-table td{padding:8px 12px;border-bottom:1px solid #cbd5e1;text-align:left;vertical-align:top}.hc-table th{font-weight:650}.hc-table{overflow-x:auto}</style>'
    display(HTML(css + '<div class="hc-table">' + body + '</div>'))


def accuracy_table(report, names):
    import pandas as pd
    table(pd.DataFrame([{
        'Method': name,
        'Correct directions': f"{report['metrics'][name]['accuracy']:.2%}",
        'Brier error (lower is better)': f"{report['metrics'][name]['brier']:.4f}",
    } for name in names]), 'All methods evaluated on the same held-out rows')


def accuracy_chart(report, names):
    """Inline SVG: deterministic, readable, no plotting dependency or hidden axis."""
    from IPython.display import SVG, display
    width, row, left, scale = 720, 40, 205, 440
    height = 55 + row * len(names)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="720" role="img" aria-label="Test accuracy comparison on a zero to one hundred percent scale">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in (0, .25, .5, .75, 1):
        x = left + tick * scale
        svg.append(f'<line x1="{x}" x2="{x}" y1="25" y2="{height-22}" stroke="#e2e8f0"/><text x="{x}" y="17" text-anchor="middle" font-size="11" fill="#475569">{tick:.0%}</text>')
    for i, name in enumerate(names):
        y = 32 + row * i
        score = report['metrics'][name]['accuracy']
        color = '#2563eb' if name == report['primary_model'] else '#64748b'
        svg.append(f'<text x="{left-12}" y="{y+18}" text-anchor="end" font-size="13" fill="#0f172a">{html.escape(name)}</text><rect x="{left}" y="{y}" width="{score*scale}" height="26" rx="3" fill="{color}"/><text x="{left+score*scale+8}" y="{y+18}" font-size="13" fill="#0f172a">{score:.2%}</text>')
    svg.append('</svg>')
    display(SVG(''.join(svg)))


def replay(test, scored, predictions, coin, date, reveal=False):
    """Reconstruct the actual saved forecast, with an optional outcome reveal."""
    import json
    import numpy as np
    import pandas as pd
    from IPython.display import Markdown, display
    saved = json.loads(Path('artifacts/headline-model.json').read_text())
    day = pd.Timestamp(date, tz='UTC')
    rows = test[(test.coin == coin) & (test.date == day)]
    if len(rows) != 1:
        raise ValueError('Choose a supported coin and a held-out UTC date.')
    row = rows.iloc[0]
    numeric = (row[saved['features']].to_numpy(dtype=float) - np.array(saved['numeric_mean'])) / np.array(saved['numeric_scale'])
    categories = np.array([float(coin == name.removeprefix('coin__coin_')) for name in saved['feature_names'][len(numeric):]])
    logit = float(np.r_[numeric, categories] @ np.array(saved['coefficient'])[0] + saved['intercept'][0])
    probability = float(1 / (1 + np.exp(-logit)))
    recorded = predictions[(predictions.coin == coin) & (pd.to_datetime(predictions.date, utc=True) == day)].iloc[0]
    assert np.isclose(probability, recorded['Sentiment + coin'], atol=1e-12)
    context = scored[(scored.date == day) & scored.tickers.apply(lambda values: coin in values)]
    display(Markdown(f"**{coin} · {date}** — forecast: **{'up' if probability >= .5 else 'not up'}**; estimated P(up): **{probability:.1%}** (not calibrated)."))
    if context.empty:
        print('No eligible archive posts; this is not evidence of silence on X.')
    else:
        table(context[['text']].head(3).rename(columns={'text': 'Prior-day input post'}))
    if reveal:
        print(f"Actual: {'up' if row.actual_up else 'not up'}; open-to-close return {row.actual_return:+.2%}.")
    else:
        print('Outcome hidden. Change reveal to True and rerun this cell.')
    print('Verified against the recorded model prediction.')
