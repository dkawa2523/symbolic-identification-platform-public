'use strict';
// Unit checks for the shipped JavaScript, not browser or visual acceptance.
// Run: node tests/verify_report_runtime.cjs [path/to/report_template.html]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(process.argv[2] || path.join(__dirname,
  '../pkgs/symbolic/src/ml_platform_symbolic/report_template.html'), 'utf8');
const start = template.indexOf('function plot(');
const end = template.indexOf('function curves(', start);
assert(start >= 0 && end > start, 'The shipped plot implementation must be tested');
const context = {esc: String, fmt: String};
vm.createContext(context);
vm.runInContext(template.slice(start, end), context);
const plot = (x, y, extra = {}) => context.plot([{x, y, label: 'probe', ...extra}], 'x', 'y');
const coordinates = html => [...html.matchAll(/<circle cx="([^"]+)" cy="([^"]+)"/g)]
  .map(match => match.slice(1).map(Number));
function checkCoordinates(html, count) {
  const values = coordinates(html);
  assert.equal(values.length, count);
  for (const [x, y] of values) {
    assert(Number.isFinite(x) && Number.isFinite(y), 'All SVG coordinates must be finite');
    assert(x >= 65 && x <= 675 && y >= 45 && y <= 245, 'Points must stay inside the plot');
  }
  return values;
}
let passed = 0;
function test(name, run) { run(); passed++; console.log(`PASS ${name}`); }
test('retains every valid row beyond the JavaScript argument limit', () => {
  const x = Array.from({length: 150000}, (_, i) => i);
  checkCoordinates(plot(x, x.map(i => i % 17)), x.length);
});
test('finite opposite extremes do not overflow the displayed span', () => {
  assert.deepEqual(checkCoordinates(plot([-1e308, 0, 1e308], [-1e308, 0, 1e308]), 3),
    [[65, 245], [370, 145], [675, 45]]);
});
test('subnormal scales do not collapse distinct points', () => {
  assert.deepEqual(checkCoordinates(plot([5e-324, 1e-323], [5e-324, 1e-323]), 2),
    [[65, 245], [675, 45]]);
});
test('constant series uses the center of each axis', () => {
  assert.deepEqual(checkCoordinates(plot([7, 7], [0, 0]), 2), [[370, 145], [370, 145]]);
});
test('invalid rows remain gaps and are never painted as zero', () => {
  const html = plot([0, 1, 2, 3, 4, 5], [0, 1, 2, null, 4, Infinity],
    {valid: [true, false, true, true, true, true], line: true});
  checkCoordinates(html, 3);
  assert.equal((html.match(/<path/g) || []).length, 1, 'No line may bridge invalid rows');
});
test('all invalid input produces no SVG', () => {
  assert(!plot([NaN, 1], [0, Infinity]).includes('<svg'));
});
test('plot formatting does not mutate stored evidence', () => {
  const series = [{x: [1, 2], y: [2, 3], valid: [true, false], label: 'unchanged'}];
  const before = JSON.stringify(series);
  context.plot(series, 'x', 'y');
  assert.equal(JSON.stringify(series), before);
});
// Run the actual comparison renderer and handlers with a small DOM adapter.
// This verifies render scope, not browser layout or accessibility behavior.
const script = template.split('<script>')[1].split('</script>')[0];
assert(script.includes('header();candidateTable();renderComparison();context();'));
function comparisonHarness(localEvidence, templateEvidence, customize) {
  const candidates = Array.from({length: 9}, (_, i) => ({
    id: `candidate-${i}`, method: 'probe', kind: 'explicit', parameters: [], members: [], questions: [],
    complexity: {nodes: 1, free: 0}, source: {}, selection: {rmse: i},
    interpretation: {paths: [{id: 'common', steps: [0, 1], prediction: [i, i + 1],
      valid: [true, true], path_unit: 's', coverage: 1}]},
    residual_scatter: {x: [0, 1], y: [i, i + 1]},
  }));
  candidates[0].local_parameter_identifiability = localEvidence;
  candidates[0].template = templateEvidence;
  candidates[8].interpretation.paths.push({...candidates[8].interpretation.paths[0], id: 'hidden-only'});
  const pairs = candidates.flatMap((left, i) => candidates.slice(i + 1).map(right => ({
    left: left.id, right: right.id, equivalence: 'unproven', common_coverage: 1,
  })));
  const evidence = {candidates, display_ids: candidates.map(c => c.id),
    decision: {selected_id: candidates[7].id}, unit: 'm', working_unit: 'm',
    pairs, analyses: [], residual_axis: {x_label: 'x', y_label: 'residual'},
    analysis_comparisons: pairs.map(pair => ({...pair, axis: 'path_response_difference',
      path_id: 'common', steps: [0, 1], difference: [1, 1], valid: [true, true],
      path_unit: 's', unit: 'm', meaning: 'stored-only'})),
  };
  if(customize)customize(evidence);
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {value: '', textContent: '', checked: false,
      addEventListener() {}, scrollIntoView() {},
      get innerHTML() { return this.html || ''; },
      set innerHTML(value) {
        this.html = value;
        // Native select chooses its first option when its option list is replaced.
        if (id === 'path') {const option=/<option(?: value="([^"]*)")?>([^<]*)<\/option>/.exec(value);this.value=option?.[1]||option?.[2]||'';}
      }});
    return nodes.get(id);
  }
  node('report-data').textContent = JSON.stringify(evidence);
  const dom = {getElementById: node, querySelectorAll: () => []};
  const runtime = vm.createContext({document: dom});
  vm.runInContext("'use strict';\n" + script.replace('header();candidateTable();renderComparison();context();', ''), runtime);
  const run = code => vm.runInContext(code, runtime);
  run('function freeze(value){if(value&&typeof value===\"object\"){Object.values(value).forEach(freeze);Object.freeze(value)}};freeze(report);');
  const original = run('JSON.stringify(report)');
  return {node, run, original, visible: () => JSON.parse(run('JSON.stringify(visibleCards().map(c=>c.id))'))};
}
function assertScope(harness, count) {
  const {node, run, original} = harness, ids = harness.visible();
  assert.equal(ids.length, count);
  assert.equal((node('cards').innerHTML.match(/<article /g) || []).length, count);
  assert.equal((node('response-plot').innerHTML.match(/<circle /g) || []).length, count * 2);
  const expectedPairs = count < 2 ? 0 : count * (count - 1) / 2;
  assert.equal((node('difference-plot').innerHTML.match(/<svg /g) || []).length, expectedPairs);
  assert.equal((node('disagreement').innerHTML.match(/<p>/g) || []).length, expectedPairs);
  const tbody = node('pair-summary').innerHTML.split('<tbody>')[1].split('</tbody>')[0];
  assert.equal((tbody.match(/<tr>/g) || []).length, expectedPairs);
  for (let i = 0; i < 9; i++) if (!ids.includes(`candidate-${i}`)) {
    for (const name of ['cards', 'response-plot', 'difference-plot', 'pair-summary', 'residual-plot']) {
      assert(!node(name).innerHTML.includes(`candidate-${i}`), `${name} must exclude hidden candidates`);
    }
  }
  assert.equal(run('JSON.stringify(report)'), original, 'Display controls must not alter scientific evidence');
}

for (const count of [0, 1, 6, 9]) test(`comparison panels share the visible scope for ${count} selected`, () => {
  const h = comparisonHarness();
  h.run(`selected.clear();report.candidates.slice(0,${count}).forEach(c=>selected.add(c.id));renderComparison();`);
  assertScope(h, Math.min(count, 6));
});

test('pinned decision stays visible while hidden-only paths are excluded', () => {
  const h = comparisonHarness();h.run('renderComparison();');
  assert.equal(h.visible()[0], 'candidate-7');
  assert(!h.node('path').innerHTML.includes('hidden-only'));
  assertScope(h, 6);
});

test('promoting a candidate updates every panel and available path consistently', () => {
  const h = comparisonHarness();h.run('promote("candidate-8");renderComparison();');
  assert.deepEqual(h.visible().slice(0, 2), ['candidate-7', 'candidate-8']);
  assert(h.node('path').innerHTML.includes('hidden-only'));
  h.node('path').value = 'hidden-only';h.run('curves();renderComparison();');
  assert.equal(h.node('path').value, 'hidden-only', 'Keep a path when it is still visible');
  h.run('selected.delete("candidate-8");renderComparison();');
  assert.equal(h.node('path').value, 'common', 'Drop a path when its last visible candidate leaves');
  assert(!h.node('path').innerHTML.includes('hidden-only'));
  assertScope(h, 6);
});

test('repeated clear and decision restore leave no stale plots or duplicate selections', () => {
  const h = comparisonHarness();h.run('renderComparison();');
  for (let i = 0; i < 3; i++) {
    h.node('clear').onclick();assertScope(h, 0);
    assert.equal(h.node('path').innerHTML, '');
    assert.equal(h.node('difference-plot').innerHTML, '');
    h.node('show-decision').onclick();h.node('show-decision').onclick();
    assertScope(h, 1);assert.equal(h.run('selected.size'), 1);
  }
});

test('stored local rank evidence renders bounds, insensitivity, warnings without mutation', () => {
  const h = comparisonHarness({status: 'computed', rank: 1, n_parameters: 2, nullity: 1,
    n_samples: 13, full_column_rank: false, insensitive_parameters: ['<a>'],
    singular_values: [2, 0], rank_threshold: 2e-7,
    parameters: [{name: '<a>', scale: 4, unit: '1', step_minus: 0, step_plus: 0.001,
      difference: 'forward', bounds: [0, 4], active_bound: 'lower'}]});
  h.run('selected.clear();selected.add("candidate-0");renderComparison();');
  const html = h.node('cards').innerHTML;
  for (const text of ['局所ランク 1 / 2', 'Selection全 13', 'ランク不足', '低感度係数',
    '&lt;a&gt;', 'lower', 'forward', '成分の一意性を証明しません']) assert(html.includes(text), text);
  assert(!html.includes('<a>'), 'Parameter labels must be escaped');
  assert.equal(h.run('JSON.stringify(report)'), h.original);
});
test('missing or unavailable local evidence is not called identified', () => {
  for (const evidence of [undefined, {status: 'unavailable', reason: 'audit pending'},
    {status: 'not_applicable', reason: 'No independent free parameters'}]) {
    const h = comparisonHarness(evidence);
    h.run('selected.clear();selected.add("candidate-0");renderComparison();');
    const html = h.node('cards').innerHTML;
    assert(html.includes(evidence?.status === 'not_applicable' ? '対象外' : '不明'));
    assert(!html.includes('この点では列フルランクです'));
    assert.equal(h.run('JSON.stringify(report)'), h.original);
  }
});
test('unresolved numerical rank never renders as insensitive or full rank', () => {
  const h = comparisonHarness({status: 'unavailable', reason: 'finite-difference numerical resolution',
    numerical_resolution: 1e-9, rank_threshold: 1e-15, singular_values: [1e-8, 1e-10],
    parameters: [{name: 'a', step_plus: 1e-5, verification_step_plus: 5e-6}]});
  h.run('selected.clear();selected.add("candidate-0");renderComparison();');
  const html = h.node('cards').innerHTML;
  assert(html.includes('不明 / 診断不可'));
  assert(html.includes('いずれとも断定しません'));
  assert(html.includes('numerical_resolution'));
  assert(!html.includes('この点では列フルランクです'));
  assert(!html.includes('低感度係数:'));
  assert.equal(h.run('JSON.stringify(report)'), h.original);
});
test('full local rank still carries non-uniqueness warning', () => {
  const h = comparisonHarness({status: 'computed', rank: 1, n_parameters: 1, nullity: 0,
    n_samples: 10, full_column_rank: true, insensitive_parameters: [], singular_values: [1], parameters: []});
  h.run('selected.clear();selected.add("candidate-0");renderComparison();');
  assert(h.node('cards').innerHTML.includes('この点では列フルランクです'));
  assert(h.node('cards').innerHTML.includes('成分の一意性を証明しません'));
});


test('slot coefficient tables expand outside table cells without changing evidence', () => {
  const h = comparisonHarness(undefined, {status: 'computed',
    knowledge: {kind: 'hypothesis', statement: 'Test'}, slots: [{name: 'h<probe>',
      arguments: [{name: 'u'}], formula: 'a*u', unit: '1', description: 'shared slot',
      parameters: [{name: 'a', value: 0.1234567890123456, unit: '1', role: 'free',
        bounds: [null, null], origin: 'declared'}]}], interpretation_limit: 'Not unique'});
  h.run('selected.clear();selected.add("candidate-0");renderComparison();');
  const html = h.node('cards').innerHTML;
  const detail = html.indexOf('<details class="slot-parameters">');
  assert(detail > 0, 'Slot coefficients need their own full-width disclosure');
  let tableDepth = 0;
  for(const match of html.slice(0, detail).matchAll(/<(\/?)table\b[^>]*>/g))
    tableDepth += match[1] ? -1 : 1;
  assert.equal(tableDepth, 0, 'A scrollable coefficient table must not be nested in another table cell');
  assert(html.includes('h&lt;probe&gt;: この関数内の係数'));
  assert(html.slice(detail).includes('class="parameter-table"'));
  assert(html.includes('0.1234567890123456'), 'Retain full saved coefficient precision');
  assert.equal(h.run('JSON.stringify(report)'), h.original);
});

test('same-named paths keep both analyses and never overlay different conditions', () => {
  const h = comparisonHarness(undefined, undefined, evidence => {
    for(const c of evidence.candidates){
      c.interpretations = ['primary', 'supplement'].map((source, index) => ({
        analysis_sha256: source, reference: {x: index + 1}, paths: [{
          id: 'response', display_id: source + ':0', at: {x: index + 1}, direction: {x: 1},
          steps: [0, 0.1], prediction: [index + 1, index + 1.1], valid: [true, true],
          path_unit: 's', input_units: {x: 'm'}, unit: 'm', coverage: 1,
        }],
      }));
      c.interpretation=c.interpretations[0];
    }
    evidence.analysis_comparisons=['primary', 'supplement'].map((source,index)=>({
      left:'candidate-0',right:'candidate-1',axis:'path_response_difference',path_id:'response',
      path_display_id:source+':0',steps:[0,0.1],difference:[index+1,index+1.1],valid:[true,true],
      maximum_point:{x:index+1.1},path_unit:'s',unit:'m',
    }));
  });
  h.run('selected.clear();selected.add("candidate-0");selected.add("candidate-1");renderComparison();');
  assert.equal((h.node('path').innerHTML.match(/<option /g)||[]).length,2);
  assert(h.node('cards').innerHTML.includes('primary')&&h.node('cards').innerHTML.includes('supplement'));
  for(const [source, x] of [['primary',1], ['supplement',2]]){
    h.node('path').value=source+':0';h.run('curves();');
    assert.equal((h.node('response-plot').innerHTML.match(/<circle /g)||[]).length,4);
    assert.equal((h.node('difference-plot').innerHTML.match(/<svg /g)||[]).length,1);
    assert(h.node('path-context').innerHTML.includes('&quot;x&quot;:'+x));
    assert(h.node('disagreement').innerHTML.includes('&quot;x&quot;:'+(x+0.1)));
  }
  assert.equal(h.run('JSON.stringify(report)'),h.original);
});

test('ambiguous historical path differences remain visible as raw evidence, not curves', () => {
  const h = comparisonHarness(undefined,undefined,evidence=>{
    for(const item of evidence.analysis_comparisons)item.path_display_id=null;
  });
  h.run('selected.clear();selected.add("candidate-0");selected.add("candidate-1");renderComparison();');
  assert(!h.node('difference-plot').innerHTML.includes('<svg'));
  assert(h.node('axis-analyses').innerHTML.includes('経路の出所を一意に照合できない旧差分（1）'));
  assert.equal(h.run('JSON.stringify(report)'),h.original);
});

test('empty selection is distinct from missing stored path or residual evidence', () => {
  const h = comparisonHarness(undefined, undefined, evidence => {
    evidence.candidates[0].interpretation.paths=[];
    delete evidence.candidates[0].residual_scatter;
  });
  h.run('selected.clear();renderComparison();');
  for(const panel of ['path-context','residual-plot']){
    assert(h.node(panel).innerHTML.includes('比較対象が選択されていません'));
    assert(!h.node(panel).innerHTML.includes('未実施'));
  }
  assert.equal(h.node('response-plot').innerHTML,'');
  assert.equal(h.node('difference-plot').innerHTML,'');
  assert.equal(h.node('disagreement').innerHTML,'');
  h.run('selected.add("candidate-0");renderComparison();');
  assert(h.node('path-context').innerHTML.includes('表示中候補には保存された共通経路の評価がありません'));
  assert(h.node('residual-plot').innerHTML.includes('target予測残差がありません'));
  h.run('selected.add("candidate-1");renderComparison();');
  assert(h.node('response-plot').innerHTML.includes('<svg'));
  assert(h.node('residual-plot').innerHTML.includes('<svg'));
  assert.equal(h.run('JSON.stringify(report)'),h.original);
});


test('metric failures and missing audits stay distinct and saved reasons are escaped', () => {
  const h = comparisonHarness(undefined, undefined, evidence => {
    Object.assign(evidence.candidates[0], {eligible: false,
      selection: {rmse: null, mae: null, r2: null, reason: 'invalid <probe>; all rows retained'},
      checks: [{name: 'units<probe>', required: true, status: 'fail', method: 'structural', reason: 'unit mismatch'}]});
    evidence.candidates[1].selection={rmse: 3, mae: 2, r2: null, reason: 'R2 undefined; RMSE/MAE retained'};
  });
  h.run('selected.clear();selected.add("candidate-0");renderComparison();candidateTable();');
  const html=h.node('cards').innerHTML;
  assert(html.includes('算出不可'));
  assert(html.includes('invalid &lt;probe&gt;; all rows retained'));
  assert(html.includes('このレポートに未提供'));
  assert(html.includes('units&lt;probe&gt;'));
  assert(html.includes('structural')&&html.includes('unit mismatch'));
  assert(!html.includes('<probe>'));
  assert(h.node('candidate-table').innerHTML.includes('要求検査で不適格（理由未記録）'));
  h.run('selected.clear();selected.add("candidate-1");renderComparison();');
  assert(h.node('cards').innerHTML.includes('<td>3</td><td>2</td><td>算出不可</td>'));
  assert.equal(h.run('JSON.stringify(report)'),h.original);
});

test('metric guide distinguishes observed error, model disagreement, validity and interval coverage', () => {
  const h=comparisonHarness(undefined,undefined,evidence=>Object.assign(evidence,{
    columns:[{column:'length<probe>',public_unit:'cm',working_unit:'m',scale:0.01,offset:0}],
    context:{target:{column:'response<probe>',unit:'m'}},mappings:[],
  }));
  h.run('context();renderComparison();');
  const guide=h.node('metric-guide').innerHTML;
  assert(guide.includes('length&lt;probe&gt; [cm]'));
  assert(guide.includes('response&lt;probe&gt;'));
  for(const label of ['RMSE =','MAE =','R² =','正解確率','候補間の予測RMS差','数値有効率','観測予測区間の被覆率','点間の成立'])
    assert(guide.includes(label),label);
  assert(h.node('response-plot').innerHTML.includes('経路の数値有効率'));
  assert(h.node('pair-summary').innerHTML.includes('共通数値有効率'));
  assert.equal(h.run('JSON.stringify(report)'),h.original);
});

test('saved prediction scatter shares selection scope and retains invalid-row gaps', () => {
  const h = comparisonHarness(undefined, undefined, evidence => {
    for(const candidate of evidence.candidates)candidate.prediction_scatter={
      observed:[1,2,3], predicted:[1,null,4], valid:[true,false,true]};
  });
  h.run('selected.clear();selected.add("candidate-0");renderComparison();');
  const rendered=h.node('prediction-plot').innerHTML;
  assert(rendered.includes('<svg'));
  assert.equal((rendered.match(/<circle /g)||[]).length,4); // two rows plus identity endpoints
  assert(rendered.includes('candidate-0'));
  assert(!rendered.includes('candidate-1'));
  assert.equal(h.run('JSON.stringify(report)'),h.original);
  h.run('selected.clear();renderComparison();');
  assert(!h.node('prediction-plot').innerHTML.includes('<svg'));
});



test('Final coverage is shown only when actually saved', () => {
  for (const coverage of [undefined, null, 0, 0.5, 1]) {
    const h = comparisonHarness(undefined, undefined, evidence => {
      evidence.purpose = 'local_prediction';
      evidence.exposure = {current_access_record: null, selection_snapshot: {status: 'none'}};
      evidence.candidates.forEach(c => {c.source = {status: 'ok'};});
      evidence.decision = {selected_id: evidence.candidates[0].id};
      evidence.audit = {label: 'saved audit', rows: {
        [evidence.candidates[0].id]: {target_metrics: {rmse: 1}, coverage}
      }};
    });
    h.run('header();');
    const html = h.node('stages').innerHTML;
    assert.equal(html.includes('有効予測率'), coverage !== undefined && coverage !== null);
    if (coverage !== undefined && coverage !== null) assert(html.includes('有効予測率 ' + coverage));
  }
});
console.log(JSON.stringify({passed, scope: 'JavaScript unit checks; no browser or visual QA'}));
