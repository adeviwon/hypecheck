'use strict';
(() => {
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const probability = value => finite(value) && value >= 0 && value <= 1;
  const text = value => typeof value === 'string' ? value : '';
  const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
  const utcTime = value => typeof value === 'string' && /(?:Z|\+00:00)$/.test(value) && Number.isFinite(Date.parse(value));
  const percent = value => probability(value) ? `${(value * 100).toFixed(1)}%` : 'Not reported';
  const count = value => Number.isInteger(value) && value >= 0 ? value.toLocaleString('en-US') : '—';
  const signed = value => finite(value) ? `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)}%` : 'Not reported';
  const stamp = value => utcTime(value) ? new Date(value).toISOString().replace('T', ' ').replace('.000Z', ' UTC') : 'Not reported';
  function parseReport(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('The report must be a JSON object.');
    const meta = raw.meta && typeof raw.meta === 'object' ? raw.meta : {};
    const test = meta.splits && meta.splits.test;
    const bounded = Array.isArray(test) && validDate(test[0]) && validDate(test[1]);
    const seen = new Set();
    const records = (Array.isArray(raw.predictions) ? raw.predictions : []).filter(row => {
      if (!row || typeof row !== 'object' || !text(row.coin) || !validDate(row.date)) return false;
      if (row.date < (validDate(meta.retrospective_pretrained_cutoff) ? meta.retrospective_pretrained_cutoff : '2024-03-01')) return false;
      if (bounded && (row.date < test[0] || row.date > test[1])) return false;
      const key = `${row.coin}\n${row.date}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).sort((a, b) => a.date.localeCompare(b.date) || a.coin.localeCompare(b.coin));
    return { meta, records, metrics: Array.isArray(raw.metrics) ? raw.metrics.filter(x => x && typeof x === 'object') : [], coverage: Array.isArray(raw.coverage) ? raw.coverage.filter(x => x && typeof x === 'object') : [], uncertainty: raw.uncertainty, learningCurve: raw.learning_curve, subgroups: raw.subgroups, extended: raw.extended && typeof raw.extended === 'object' && !Array.isArray(raw.extended) ? raw.extended : null };
  }
  function eligibleExamples(row) {
    if (!utcTime(row.cutoff)) return [];
    return (Array.isArray(row.examples) ? row.examples : []).filter(post => {
      if (!post || !text(post.text) || !utcTime(post.created_at)) return false;
      if (post.first_posted_at != null && !utcTime(post.first_posted_at)) return false;
      if (post.available_at != null && !utcTime(post.available_at)) return false;
      const available = Math.max(Date.parse(post.created_at), post.first_posted_at ? Date.parse(post.first_posted_at) : -Infinity, post.available_at ? Date.parse(post.available_at) : -Infinity);
      return available >= Date.parse(row.cutoff) - 86400000 && available < Date.parse(row.cutoff);
    });
  }
  // Runnable parser/boundary check: node docs/app.js --self-test
  if (typeof document === 'undefined') {
    if (typeof module !== 'undefined') module.exports = { parseReport, eligibleExamples };
    if (typeof process !== 'undefined' && process.argv.includes('--self-test')) {
      const assert = require('node:assert/strict');
      const row = { coin: 'DOGE', date: '2025-01-02', cutoff: '2025-01-02T00:00:00Z', examples: [{ text: '<script>not markup</script>', created_at: '2025-01-01T23:59:59Z' }, { text: 'at cutoff', created_at: '2025-01-02T00:00:00Z' }, { text: 'later', created_at: '2025-01-03T00:00:00Z' }] };
      assert.equal(parseReport({ predictions: [row, row, null, { coin: 'DOGE', date: '2025-02-30' }] }).records.length, 1);
      assert.equal(eligibleExamples(row).length, 1);
      assert.equal(eligibleExamples({ ...row, examples: [{ text: 'late publication', created_at: '2025-01-01T00:00:00Z', first_posted_at: '2025-01-02T01:00:00Z' }] }).length, 0);
      assert.equal(eligibleExamples({ ...row, examples: [{ text: 'late observation', created_at: '2025-01-01T22:00:00Z', available_at: '2025-01-02T01:00:00Z' }] }).length, 0);
      assert.equal(eligibleExamples({ ...row, examples: [{ text: 'too old', created_at: '2024-12-31T22:00:00Z' }] }).length, 0);
      assert.equal(parseReport({ predictions: [{ coin: 'DOGE', date: '2024-02-29' }] }).records.length, 0);
      assert.equal(eligibleExamples({ ...row, cutoff: null }).length, 0);
      assert.equal(parseReport({ meta: { splits: { test: ['2025-02-01', '2025-03-01'] } }, predictions: [row] }).records.length, 0);
      assert.equal(percent(null), 'Not reported');
      assert.equal(parseReport({}).extended, null);
      assert.equal(parseReport({ extended: 'nope' }).extended, null);
      assert.equal(parseReport({ extended: { metrics: [{ name: 'x', accuracy: 0.5, n: 3, n_dates: 2 }] } }).extended.metrics.length, 1);
      assert.equal(parseReport({}).records.length, 0);
      assert.throws(() => parseReport(null));
      console.log('HypeCheck self-test passed: dates, deduplication, test split, cutoff and missing values.');
    }
    return;
  }
  const $ = id => document.getElementById(id);
  const node = (tag, content, className) => {
    const element = document.createElement(tag);
    if (content !== undefined) element.textContent = content;
    if (className) element.className = className;
    return element;
  };
  let report;
  function addProvenance(label, value) {
    $('provenance').append(node('dt', label), node('dd', text(value) || 'Not reported'));
  }
  function renderEvidence() {
    const { meta, metrics, coverage } = report;
    $('results-section').hidden = false;
    $('evaluation-scope').textContent = text(meta.evaluation_scope) || 'Evaluation scope not supplied. Metrics below are taken from the local report; do not assume a particular test design.';
    const body = $('metrics');
    body.replaceChildren();
    metrics.forEach(metric => {
      const tr = node('tr');
      if (metric.name === 'Market + X') tr.className = 'highlight';
      const values = [text(metric.name) || 'Unnamed model', percent(metric.accuracy), percent(metric.balanced_accuracy), probability(metric.brier) ? metric.brier.toFixed(4) : '—', count(metric.n), count(metric.n_dates)];
      values.forEach((value, index) => {
        const cell = node(index ? 'td' : 'th', value);
        if (!index) cell.scope = 'row';
        tr.append(cell);
      });
      body.append(tr);
    });
    if (!metrics.length) { const tr = node('tr'); const td = node('td', 'No evaluation metrics supplied.'); td.colSpan = 6; tr.append(td); body.append(tr); }
    const analysis = node('div', undefined, 'research-summary');
    const uncertainty = report.uncertainty;
    if (uncertainty && finite(uncertainty.mean_social_minus_market_brier) && Array.isArray(uncertainty.interval95) && uncertainty.interval95.every(finite)) {
      const [low, high] = uncertainty.interval95;
      analysis.append(node('h3', low <= 0 && high >= 0 ? 'No demonstrated improvement from X sentiment.' : 'Inspect the uncertainty, not just the point estimate.'));
      analysis.append(node('p', `Social minus market Brier error: ${uncertainty.mean_social_minus_market_brier.toFixed(4)}. Exploratory 95% date-block interval: ${low.toFixed(4)} to ${high.toFixed(4)}. Negative values favour X; an interval crossing zero does not demonstrate an improvement.`));
      if (text(uncertainty.method)) analysis.append(node('p', uncertainty.method, 'small muted'));
    }
    const observed = report.subgroups && report.subgroups.with_observed_posts;
    if (observed && Number.isInteger(observed.n)) analysis.append(node('p', `${count(observed.n)} test coin-days had observed eligible posts. Most dates may have no observed social input; this archive is not complete X coverage.`, 'small muted'));
    if (Array.isArray(report.learningCurve)) {
      analysis.append(node('h3', 'Did more training history help?'));
      report.learningCurve.filter(x => x && probability(x.validation_brier)).forEach(x => analysis.append(node('p', `${count(x.train_dates)} training dates → validation Brier ${x.validation_brier.toFixed(4)}.`, 'small mono')));
      analysis.append(node('p', 'These are validation-only comparisons of historical training windows, not proof that adding arbitrary posts improves future forecasts.', 'small muted'));
    }
    $('results-section').append(analysis);
    const extendedHost = $('extended');
    extendedHost.replaceChildren();
    const extended = report.extended;
    const extendedRows = extended && Array.isArray(extended.metrics) ? extended.metrics.filter(x => x && typeof x === 'object') : [];
    if (extendedRows.length) {
      extendedHost.append(node('h3', 'Declared secondary comparison'));
      if (text(extended.note)) extendedHost.append(node('p', text(extended.note), 'muted'));
      const wrap = node('div', undefined, 'table-wrap');
      wrap.tabIndex = 0;
      wrap.setAttribute('role', 'region');
      wrap.setAttribute('aria-label', 'Secondary comparison using extended market features');
      const table = node('table');
      table.append(node('caption', 'Extended market features · same held-out rows · lower Brier score is better'));
      const head = node('thead'); const header = node('tr');
      ['Model', 'Accuracy', 'Balanced accuracy', 'Brier score', 'Observations', 'Dates'].forEach(label => { const th = node('th', label); th.scope = 'col'; header.append(th); });
      head.append(header); table.append(head);
      const tbody = node('tbody');
      extendedRows.forEach(metric => {
        const tr = node('tr');
        const values = [text(metric.name) || 'Unnamed model', percent(metric.accuracy), percent(metric.balanced_accuracy), probability(metric.brier) ? metric.brier.toFixed(4) : '—', count(metric.n), count(metric.n_dates)];
        values.forEach((value, index) => { const cell = node(index ? 'td' : 'th', value); if (!index) cell.scope = 'row'; tr.append(cell); });
        tbody.append(tr);
      });
      table.append(tbody); wrap.append(table); extendedHost.append(wrap);
      const primaryBaseline = extendedRows.find(x => text(x.name) === 'Market only (primary)');
      const extendedBaseline = extendedRows.find(x => text(x.name) === 'Market + BTC + intraday');
      if (primaryBaseline && extendedBaseline && probability(primaryBaseline.accuracy) && probability(extendedBaseline.accuracy)) {
        extendedHost.append(node('p', extendedBaseline.accuracy > primaryBaseline.accuracy
          ? 'Hourly bars and BTC context improved the market baseline on these rows.'
          : 'Hourly bars and BTC context did not improve the market baseline on these rows.', 'small'));
      }
      const extendedUncertainty = extended.uncertainty;
      if (extendedUncertainty && finite(extendedUncertainty.mean_social_minus_market_brier) && Array.isArray(extendedUncertainty.interval95) && extendedUncertainty.interval95.every(finite)) {
        const [low, high] = extendedUncertainty.interval95;
        extendedHost.append(node('p', `Sentiment minus richer-market Brier error: ${extendedUncertainty.mean_social_minus_market_brier.toFixed(4)}. Exploratory 95% date-block interval: ${low.toFixed(4)} to ${high.toFixed(4)}.`, 'small'));
      }
      extendedHost.append(node('p', 'These rows were added after the primary run was frozen and are reported separately; they were not used to select the primary models.', 'small muted'));
    }
    const coverageHost = $('coverage');
    coverageHost.replaceChildren();
    if (coverage.length) {
      const table = node('table');
      table.append(node('caption', 'Coverage reported by the data pipeline'));
      const head = node('thead'); const header = node('tr');
      ['Asset', 'Total', 'Original', 'Eligible', 'Post days', 'Price days'].forEach(label => { const th = node('th', label); th.scope = 'col'; header.append(th); });
      head.append(header); table.append(head);
      const tbody = node('tbody');
      coverage.forEach(item => {
        const tr = node('tr');
        [text(item.coin) || 'Unknown', count(item.total_posts), count(item.original_posts), count(item.eligible_posts), count(item.days_with_posts), count(item.price_days)].forEach((value, index) => { const cell = node(index ? 'td' : 'th', value); if (!index) cell.scope = 'row'; tr.append(cell); });
        tbody.append(tr);
      });
      table.append(tbody); coverageHost.append(table);
      coverage.forEach(item => coverageHost.append(node('p', `${text(item.coin) || 'Unknown'} · ${validDate(item.first) ? item.first : 'start not reported'} → ${validDate(item.last) ? item.last : 'end not reported'}`, 'small muted')));
    } else coverageHost.append(node('p', 'Coverage statistics were not supplied.', 'small muted'));
    (Array.isArray(meta.limitations) ? meta.limitations : []).filter(value => typeof value === 'string').forEach(value => $('limitations').append(node('li', value)));
    if (text(meta.disclaimer)) $('limitations').append(node('li', meta.disclaimer));
    addProvenance('Dataset revision', meta.dataset_revision);
    addProvenance('Sentiment model', meta.sentiment_model);
    addProvenance('Sentiment revision', meta.sentiment_revision);
    addProvenance('Report generated', meta.generated_at);
    addProvenance('Retrospective checkpoint cutoff', meta.retrospective_pretrained_cutoff);
    addProvenance('Source availability policy', meta.availability_policy);
    addProvenance('Eligible posts without collection timestamps', Number.isInteger(meta.eligible_posts_missing_collection_timestamp) ? String(meta.eligible_posts_missing_collection_timestamp) : 'Not reported');
    addProvenance('Target window', 'Daily UTC 00:00 → next 00:00; historical replay, not live forecasting.');
    addProvenance('Extended market inputs', meta.hourly_market_source);
    addProvenance('BTC context', meta.btc_context_source);
    ['train', 'validation', 'test'].forEach(split => {
      const range = meta.splits && meta.splits[split];
      addProvenance(`${split} dates`, Array.isArray(range) ? range.map(text).join(' → ') : 'Not reported');
    });
  }
  function selectedRows() { return report.records.filter(row => row.coin === $('coin').value); }
  function selectedRow() { return selectedRows().find(row => row.date === $('date').value); }
  function updateDates() {
    const previous = $('date').value;
    $('date').replaceChildren(...selectedRows().map(row => { const option = node('option', row.date); option.value = row.date; return option; }));
    if (selectedRows().some(row => row.date === previous)) $('date').value = previous;
    else { const firstObserved = selectedRows().find(row => row.post_count > 0); if (firstObserved) $('date').value = firstObserved.date; }
    $('reveal').checked = false;
    renderReplay();
  }
  function renderReplay() {
    const row = selectedRow();
    if (!row) return;
    $('cutoff').textContent = `Information cutoff: ${stamp(row.cutoff)} · Target ends: ${stamp(row.target_end)}`;
    $('market-prob').textContent = percent(row.market_prob);
    $('social-prob').textContent = percent(row.social_prob);
    const delta = probability(row.market_prob) && probability(row.social_prob) ? (row.social_prob - row.market_prob) * 100 : null;
    $('delta').textContent = delta === null ? 'Probability comparison unavailable for this record.' : `${delta > 0 ? '+' : ''}${delta.toFixed(1)} percentage points with X · not evidence of improved accuracy on its own`;
    $('post-count').textContent = Number.isInteger(row.post_count) && row.post_count >= 0 ? `${count(row.post_count)} eligible posts` : 'Post count unavailable';
    const sentiment = finite(row.sentiment) ? `Reported sentiment score: ${row.sentiment.toFixed(3)}.` : 'Sentiment not available. Missing posts are not neutral.';
    $('sentiment').textContent = `${sentiment} Bullish: ${percent(row.bullish)} · bearish: ${percent(row.bearish)}. These describe text, not return probabilities.`;
    const posts = $('posts'); posts.replaceChildren();
    const examples = eligibleExamples(row);
    examples.forEach(post => {
      const article = node('article', undefined, 'post');
      article.append(node('p', post.text));
      const time = node('time', stamp(post.created_at)); time.dateTime = post.created_at; article.append(time);
      if (utcTime(post.available_at)) article.append(node('p', `Source availability: ${stamp(post.available_at)}`, 'small muted'));
      try {
        const url = new URL(post.url);
        if (url.protocol === 'https:' && /^(www\.)?(x\.com|twitter\.com)$/.test(url.hostname)) {
          const a = node('a', 'Original post ↗'); a.href = url.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.setAttribute('aria-label', 'Read original post on X (opens in a new tab)'); article.append(a);
        }
      } catch (_) { /* Missing or unsafe source links stay unlinked. */ }
      posts.append(article);
    });
    if (!examples.length) posts.append(node('p', row.post_count === 0 ? 'No eligible posts for this cutoff. There is no social evidence to display.' : 'No verifiable pre-cutoff original-post examples supplied for this record.', 'no-posts'));
    const outcome = $('outcome'); outcome.replaceChildren();
    if ($('reveal').checked) {
      outcome.append(node('strong', signed(row.actual_return), 'actual-return'));
      outcome.append(node('span', `Realized next-24h return · direction: ${row.actual_up === 1 ? 'up' : row.actual_up === 0 ? 'not up' : 'not reported'}. Target ends ${stamp(row.target_end)}.`));
    } else outcome.textContent = 'Outcome hidden. Make your own assessment first.';
    renderChart();
  }
  function renderChart() {
    const rows = selectedRows();
    const ns = 'http://www.w3.org/2000/svg';
    const svgNode = (tag, attributes, content) => {
      const element = document.createElementNS(ns, tag);
      Object.entries(attributes || {}).forEach(([key, value]) => element.setAttribute(key, String(value)));
      if (content !== undefined) element.textContent = content;
      return element;
    };
    const svg = svgNode('svg', { viewBox: '0 0 1040 265', role: 'img', 'aria-labelledby': 'plot-title plot-desc' });
    svg.append(svgNode('title', { id: 'plot-title' }, `${$('coin').value}: reported probabilities across available test dates`));
    svg.append(svgNode('desc', { id: 'plot-desc' }, 'Both model lines show probability of a positive next-24-hour return, from zero to one hundred percent. Missing probabilities break the line. Actual directions are shown only when reveal is enabled.'));
    const x = index => rows.length === 1 ? 540 : 52 + index * 960 / (rows.length - 1);
    const y = value => 215 - value * 190;
    [0, .25, .5, .75, 1].forEach(value => {
      svg.append(svgNode('line', { x1: 52, x2: 1012, y1: y(value), y2: y(value), class: 'grid-line' }));
      svg.append(svgNode('text', { x: 0, y: y(value) + 4 }, `${value * 100}%`));
    });
    ['market_prob', 'social_prob'].forEach((key, series) => {
      let path = ''; let connected = false;
      rows.forEach((row, index) => {
        if (!probability(row[key])) { connected = false; return; }
        path += `${connected ? 'L' : 'M'}${x(index).toFixed(2)},${y(row[key]).toFixed(2)} `;
        connected = true;
        if (rows.length === 1) svg.append(svgNode('circle', { cx: x(index), cy: y(row[key]), r: 4, class: series ? 'social-dot' : 'market-dot' }));
      });
      if (path) svg.append(svgNode('path', { d: path, class: series ? 'social-path' : 'market-path' }));
    });
    const selected = rows.findIndex(row => row.date === $('date').value);
    if (selected >= 0) svg.append(svgNode('line', { x1: x(selected), x2: x(selected), y1: 20, y2: 220, class: 'selection-line' }));
    if ($('reveal').checked) rows.forEach((row, index) => {
      if (row.actual_up === 0 || row.actual_up === 1) svg.append(svgNode('circle', { cx: x(index), cy: y(row.actual_up), r: 3, class: 'actual-dot' }));
    });
    if (rows.length) {
      svg.append(svgNode('text', { x: 52, y: 250 }, rows[0].date));
      if (rows.length > 1) svg.append(svgNode('text', { x: 1012, y: 250, 'text-anchor': 'end' }, rows[rows.length - 1].date));
    }
    $('actual-key').hidden = !$('reveal').checked;
    $('chart').replaceChildren(svg);
    $('chart-caption').textContent = `${rows.length} available ${$('coin').value} test records · dates equally spaced; missing dates are not interpolated. Vertical marker: selected date. ${$('reveal').checked ? 'Circles mark realized direction at 0% or 100%; these are outcomes, not forecast probabilities.' : 'Realized outcomes remain hidden.'} Exact selected-date estimates appear above.`;
  }
  async function load() {
    try {
      const response = await fetch('report.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Report request returned HTTP ${response.status}.`);
      report = parseReport(await response.json());
      renderEvidence();
      if (!report.records.length) {
        $('load-status').textContent = 'The report loaded, but contains no valid historical test predictions. Coverage and any reported metrics are shown below; no forecasts have been invented.';
        return;
      }
      const coins = [...new Set(report.records.map(row => row.coin))].sort();
      $('coin').replaceChildren(...coins.map(coin => { const option = node('option', coin); option.value = coin; return option; }));
      if (coins.includes('DOGE')) $('coin').value = 'DOGE';
      $('replay').hidden = false;
      $('load-status').hidden = true;
      updateDates();
    } catch (error) {
      $('load-status').classList.add('error');
      $('load-status').textContent = 'The local research report could not be loaded. No predictions or results are available. Serve this folder over HTTP with report.json beside index.html, then refresh. ' + (error instanceof SyntaxError ? 'The report is not valid JSON.' : text(error.message));
    }
  }
  $('coin').addEventListener('change', updateDates);
  $('date').addEventListener('change', () => { $('reveal').checked = false; renderReplay(); });
  $('reveal').addEventListener('change', renderReplay);
  load();
})();
