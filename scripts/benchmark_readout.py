"""Offline HTML readout of saved benchmark evidence, never a scorer or selector.

python scripts/benchmark_readout.py SCORES_JSON NEW_OUTPUT_DIRECTORY [--context CONTEXT_JSON]
Only the named JSON files are read. Missing arrays remain missing; no model or Final data is opened.
"""

import argparse
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def text(value):
    if value is None:
        return "未保存 / 未評価"
    if isinstance(value, float):
        return f"{value:.7g}" if math.isfinite(value) else "無効"
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + text(h) + '</th>' for h in headers) + (
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(v) + '</td>' for v in r) + '</tr>'
                                        for r in rows) + '</tbody></table></div>')


def metric_rows(row):
    """Keep partition names; never silently pool schemas or recompute scores."""
    result = []
    if "selection_rmse" in row:
        result.append(("Selection", {"rmse": row["selection_rmse"]}))
    if isinstance(row.get("test_metrics"), dict):
        result.append(("Test / Final", {**row["test_metrics"], "coverage": row.get("coverage")}))
    for key in ("strata", "final"):
        for name, value in (row.get(key) or {}).items():
            if isinstance(value, dict):
                result.append((key + ": " + name, value))
    if "rss" in row or "rss_relative_error" in row:
        result.append(("全観測へのfit（独立testではない）", row))
    return result


def scatter(x, y, *, title, xlabel, ylabel, valid=None, diagonal=False):
    if len(x) != len(y) or (valid is not None and len(valid) != len(x)):
        raise ValueError("Stored plot arrays have different lengths")
    valid = [True] * len(x) if valid is None else valid
    points = [(a, b, i) for i, (a, b, ok) in enumerate(zip(x, y, valid, strict=True))
              if ok and finite(a) and finite(b)]
    if not points:
        return ('<figure><figcaption>' + text(title) + '</figcaption><p class="warning">'
                + f'図に描ける有効な保存行がありません。描画 0 / 保存 {len(x)}行。全行を分母に残します。</p></figure>')
    # Scale first: finite opposite extremes must not overflow the axis span.
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    if diagonal:
        xs = ys = xs + ys
    def axis(values):
        scale = max(abs(min(values)), abs(max(values))) or 1.0
        low, high = min(values) / scale, max(values) / scale
        return lambda value: 0.5 if high == low else (value / scale - low) / (high - low)
    sx, sy = axis(xs), axis(ys)
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    svg = f'<svg viewBox="0 0 720 320" role="img" aria-label="{text(title)}"><title>{text(title)}</title>'
    svg += '<path d="M80 35 V255 H675" fill="none" stroke="#8596a5"/>'
    if diagonal:
        svg += '<path d="M80 255 L675 35" fill="none" stroke="#888" stroke-dasharray="6 4"/>'
    elif lo_y <= 0 <= hi_y:
        yy = 255 - 220 * sy(0)
        svg += f'<path d="M80 {yy} H675" fill="none" stroke="#888" stroke-dasharray="6 4"/>'
    for a, b, i in points:
        svg += f'<circle cx="{80 + 595 * sx(a):.4f}" cy="{255 - 220 * sy(b):.4f}" r="3" fill="#146889" opacity=".62"><title>row {i}: {text(a)}, {text(b)}</title></circle>'
    for yy, label in [(255, lo_y), (35, hi_y)]:
        svg += f'<text x="75" y="{yy}" text-anchor="end">{text(label)}</text>'
    svg += f'<text x="80" y="274">{text(lo_x)}</text><text x="675" y="274" text-anchor="end">{text(hi_x)}</text>'
    svg += f'<text x="375" y="307" text-anchor="middle">{text(xlabel)}</text><text x="80" y="20">{text(ylabel)}</text></svg>'
    return '<figure><figcaption>' + text(title) + '</figcaption>' + svg + f'<p class="small">描画 {len(points)} / 保存 {len(x)}行。無効・欠損 {len(x) - len(points)}行は非表示、指標は再計算しません。</p></figure>'


def error_chart(rows, partition, unit):
    entries = [(i, dict(metric_rows(row)).get(partition, {}).get("rmse"))
               for i, row in enumerate(rows, 1)]
    return scatter([i for i, _ in entries], [v for _, v in entries], title=partition + " の保存RMSE",
                   xlabel="下の試行番号（値の大小で並べ替えない）", ylabel="RMSE [" + unit + "]")


def prediction_figures(row, unit):
    saved = row.get("prediction_rows")
    if not saved:
        return '<p class="warning">実測・予測・残差の行配列はこの結果に未保存です。誤差の要約から散布図を捏造せず、既存結果の再採点もしません。今後の対応scorerは採点時に配列を保存します。</p>'
    observed, predicted = saved["observed"], saved["predicted"]
    valid = saved.get("valid")
    scope = text(saved.get("partition", "対象partition未記録"))
    body = '<p>描画対象: ' + scope + '。各点は実際の入力条件での1行です。</p><div class="figures">'
    body += scatter(observed, predicted, valid=valid, diagonal=True, title="実測 vs 予測（破線は一致線）",
                    xlabel="実測 [" + unit + "]", ylabel="予測 [" + unit + "]")
    if saved.get("residual") is not None:
        body += scatter(predicted, saved["residual"], valid=valid, title="予測 vs 残差（実測 − 予測）",
                        xlabel="予測 [" + unit + "]", ylabel="残差 [" + unit + "]")
    else:
        body += '<p>残差配列は未保存です。</p>'
    return body + '</div>'


def render(evidence, *, source_name, source_sha256, context=None):
    rows = evidence.get("results", evidence.get("attempts"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a results or attempts array; unknown schema is not inferred")
    cases = defaultdict(list)
    for row in rows:
        cases[str(row.get("case", row.get("dataset", "case未記録")))].append(row)
    contexts = {**evidence.get("case_context", {}), **(context or {})}
    declared = evidence.get("attempts")
    count = str(len(rows)) + (f" / 宣言 {declared}" if isinstance(declared, int) else "（予定総数は未保存）")
    body = '<header><p class="eyebrow">SAVED BENCHMARK EVIDENCE</p><h1>ベンチマーク結果を読む</h1>'
    body += '<p>対象: <strong>' + text(source_name) + '</strong> ／ ' + text(count) + '試行 ／ ' + str(len(cases)) + 'ケース</p>'
    body += '<p>ケース → 指標の図 → 全試行の式・精度・失敗理由の順に読みます。保存値のみを表示し、探索・再fit・再選択・新しいFinal評価は行いません。</p>'
    body += '<p class="warning">実行完了、要求適格、予測精度、真式の同定は別です。異なるデータ・単位・予算のケースを一つのランキングにしません。</p>'
    body += '<p>' + text(evidence.get("interpretation", evidence.get("purpose", "実験の解釈は元protocolを確認してください。"))) + '</p>'
    body += '<p><a href="evidence.json">元の保存結果 JSON</a> · <a href="#guide">指標の読み方</a></p></header><nav>'
    body += ''.join(f'<a href="#case-{i}">{text(name)} ({len(items)}試行)</a> ' for i, (name, items) in enumerate(cases.items())) + '</nav>'
    for case_index, (name, items) in enumerate(cases.items()):
        info = contexts.get(name, {})
        unit = str(info.get("target_unit") or "未記録")
        body += f'<section id="case-{case_index}"><h2>{text(name)}</h2>'
        body += '<p><strong>何を調べたか:</strong> ' + text(info.get("question", "未記録。元protocolを参照")) + '</p>'
        body += '<p><strong>真値・referenceの区分:</strong> ' + text(info.get("truth_status", "未記録。良いfitを真式とみなさない")) + '</p>'
        if info.get("reference_formula"):
            body += '<p>生成式 / 公表reference（推定式とは別）:</p><pre class="equation">' + text(info["reference_formula"]) + '</pre>'
        body += '<p><strong>変数・単位:</strong> ' + text(info.get("variables", "未記録")) + ' ／ target [' + text(unit) + ']</p>'
        if info.get("scope"):
            body += '<p>' + text(info["scope"]) + '</p>'
        if info.get("source"):
            body += '<p class="small">出典: ' + text(info["source"]) + '</p>'
        labels = list(dict.fromkeys(label for row in items for label, m in metric_rows(row) if "rmse" in m))
        body += '<div class="figures">' + ''.join(error_chart(items, label, unit) for label in labels) + '</div>'
        body += '<h3>全試行と保存式</h3><p>data seedはデータの違い、solver seedは同じデータ上の探索の違いです。試行を独立データとして平均しません。</p>'
        for number, row in enumerate(items, 1):
            engine = row.get("engine", "supplied-form least squares" if row.get("formula") else "未記録")
            body += '<article><h3>' + text(number) + '. ' + text(engine) + ' ｜ data seed ' + text(row.get("data_seed")) + ' ｜ solver seed / start ' + text(row.get("solver_seed", row.get("seed", row.get("start")))) + '</h3>'
            body += '<p><strong>状態:</strong> ' + text(row.get("score_status", row.get("status", row.get("native_status")))) + ' ／ native: ' + text(row.get("native_execution_status", row.get("native_status"))) + '</p>'
            if row.get("reason") or row.get("selection_failure"):
                body += '<p class="warning">理由: ' + text(row.get("reason") or row.get("selection_failure")) + '</p>'
            body += '<p class="small">候補 ' + text(row.get("candidate_count")) + ' / 適格 ' + text(row.get("eligible_count")) + ' ／ nodes ' + text(row.get("n_nodes")) + ' ／ free parameters ' + text(row.get("n_parameters")) + '</p>'
            body += '<p><strong>保存された推定式 / 供給式</strong></p><pre class="equation">' + text(row.get("equation", row.get("formula", "実行可能な式は未保存。失敗・未対応から式を補いません。"))) + '</pre>'
            if row.get("parameters") is not None or row.get("certified") is not None:
                names = list(dict.fromkeys([*(row.get("parameters") or {}), *(row.get("certified") or {})]))
                body += table(["係数", "推定値", "公表reference値"], [[text(k), text((row.get("parameters") or {}).get(k)), text((row.get("certified") or {}).get(k))] for k in names])
            metrics = metric_rows(row)
            body += table(["対象", "RMSE [" + unit + "]", "MAE [" + unit + "]", "R²", "有効率", "理由"], [
                [text(label), text(m.get("rmse")), text(m.get("mae")), text(m.get("r2")), text(m.get("coverage")), text(m.get("reason", "なし（保存値）"))] for label, m in metrics])
            for label, m in metrics:
                extras = {key: m[key] for key in ("rss", "certified_rss", "rss_relative_error", "prediction_relative_l2", "clean_rmse", "scientific_failure") if key in m}
                if extras:
                    body += table(["保存診断 / " + label, "値"], [[text(k), text(v)] for k, v in extras.items()])
            body += prediction_figures(row, unit)
            body += '<details><summary>この試行の保存値（省略なし）</summary><pre>' + text(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details></article>'
        body += '</section>'
    body += '<section id="guide"><h2>指標の読み方・再現性</h2><ul><li>RMSE / MAE は小さいほど観測との差が小さく、targetと同じ単位です。単位未保存の場合は推測しません。</li><li>R² は1が完全一致で、負にもなります。真の構造・因果・外挿保証ではありません。</li><li>有効率は数値計算できた行の割合で、予測区間のcoverageとは別です。欠損や失敗はゼロ誤差ではありません。</li><li>Selectionは選択に使用。Test / Finalは固定後の評価。NISTの全点fitは独立testではありません。</li><li>残差は保存された実測 − 予測。ノイズなし真値との誤差は clean_rmse と区別します。</li></ul>'
    body += '<p>入力SHA-256: <code>' + text(source_sha256) + '</code></p><p>このHTMLは外部CDN・フォント・通信・JavaScript不要で開けます。元の結果は evidence.json に同じバイト列で保存しています。</p></section>'
    return '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ベンチマーク結果</title><style>' + CSS + '</style><main>' + body + '</main></html>'


CSS = """
:root{font:16px/1.65 system-ui,sans-serif;color:#183444;background:#eff4f6}body{margin:0}main{max-width:1240px;margin:auto;padding:26px}header,section{background:white;border:1px solid #cfdae1;border-radius:12px;padding:26px;margin-bottom:22px}header{border-top:6px solid #176e85}h1{font-size:2rem;line-height:1.3}h2{font-size:1.5rem}h3{font-size:1.06rem}.eyebrow{color:#176e85;font-size:.8rem;letter-spacing:.1em}.warning{background:#fff4e4;border-left:4px solid #ba7624;padding:12px}.equation{font-size:1.05rem;background:#edf5f8;border-left:4px solid #176e85;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}nav{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}nav a{background:white;padding:9px 14px;border:1px solid #bacdd5;border-radius:6px}a{color:#065f7e}article{border-top:2px solid #bdcfd9;padding:18px 0}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:9px;border-bottom:1px solid #d1dde3;text-align:left;vertical-align:top}th{background:#edf3f6}.scroll{overflow-x:auto}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}figure{margin:8px 0;padding:12px;background:#f8fafc;border:1px solid #d5dfe6}figcaption{font-weight:600}svg{width:100%;height:auto}svg text{font:12px system-ui,sans-serif;fill:#344d5b}.small{font-size:.8rem;color:#536b79}details{margin-top:14px}details pre{max-height:350px;overflow:auto;font-size:.8rem}p,li,td{overflow-wrap:anywhere}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:10px}header,section{padding:16px}.figures{display:block}}
"""


def write_report(source, destination, *, context=None):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    evidence = json.loads(raw)
    rendered = render(evidence, source_name=source.name, source_sha256=hashlib.sha256(raw).hexdigest(), context=context)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "evidence.json").write_bytes(raw)
    if context is not None:
        (destination / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    path = destination / "report.html"
    path.write_text(rendered, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--context", type=Path, help="Optional display-only case-name to context mapping; never read by a solver")
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else None
    print(write_report(args.scores, args.output_dir, context=context))
