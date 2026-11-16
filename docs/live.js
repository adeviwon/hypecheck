"use strict";
// HypeCheck live forecast: wording + coin model, evaluated in the browser.
// Pure functions are exported for Node self-tests; DOM wiring only runs in a page.
const TOKEN = /[\p{L}\p{N}_]{2,}/gu;

function tokens(text) {
  const matches = String(text || "").toLowerCase().match(TOKEN);
  return matches || [];
}

function linearBlocks(artifact, text, coin) {
  const blocks = artifact.model.blocks;
  const textBlock = blocks.find(b => b.name === "text");
  const coinBlock = blocks.find(b => b.name === "coin");
  const counts = new Map();
  for (const term of tokens(text)) counts.set(term, (counts.get(term) || 0) + 1);
  const terms = [];
  let norm = 0;
  for (const [term, count] of counts) {
    if (!Object.prototype.hasOwnProperty.call(textBlock.vocabulary, term)) continue;
    const index = textBlock.vocabulary[term];
    const value = count * textBlock.idf[index];
    terms.push({ term, index, value });
    norm += value * value;
  }
  norm = Math.sqrt(norm);
  let z = artifact.model.intercept;
  const contributions = [];
  for (const { term, index, value } of terms) {
    const contribution = norm ? (value / norm) * textBlock.coefficient[index] : 0;
    z += contribution;
    contributions.push({ term, contribution });
  }
  contributions.sort((a, b) => b.contribution - a.contribution);
  const coinIndex = coinBlock.categories.indexOf(coin);
  if (coinIndex >= 0) z += coinBlock.coefficient[coinIndex];
  return { z, contributions, coinKnown: coinIndex >= 0 };
}

function baselineLogit(artifact, coin) {
  const block = artifact.coin_baseline.blocks.find(b => b.name === "coin");
  const index = block.categories.indexOf(coin);
  return artifact.coin_baseline.intercept + (index >= 0 ? block.coefficient[index] : 0);
}

function score(artifact, text, coin) {
  const { z, contributions, coinKnown } = linearBlocks(artifact, text, coin);
  const baseline = baselineLogit(artifact, coin);
  if (!Number.isFinite(z) || !Number.isFinite(baseline)) throw new Error("Invalid model score; forecast blocked.");
  const probability = 1 / (1 + Math.exp(-z));
  return {
    probability_up: probability,
    direction: probability >= 0.5 ? "UP" : "DOWN / FLAT",
    coin_known: coinKnown,
    coin_baseline_probability_up: 1 / (1 + Math.exp(-baseline)),
    contributions,
    toward_up: contributions.filter(c => c.contribution > 0).slice(0, 6),
    toward_down: contributions.filter(c => c.contribution < 0).slice(-6).reverse(),
  };
}

function detectCoin(artifact, text) {
  const lower = String(text || "").toLowerCase();
  const cashtags = [...lower.matchAll(/\$([a-z0-9_]{2,10})\b/g)].map(m => m[1].toUpperCase());
  for (const tag of cashtags) if (artifact.coins.includes(tag)) return tag;
  for (const coin of artifact.coins) {
    const pattern = new RegExp("\\b" + coin.toLowerCase() + "\\b", "u");
    if (pattern.test(lower)) return coin;
  }
  return null;
}

function targetWindow(now) {
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  const end = new Date(start.getTime() + 86400000);
  const label = date => date.toISOString().slice(0, 10) + " 00:00 UTC";
  return { start: label(start), end: label(end) };
}

function validate(text) {
  const body = String(text || "").trim();
  if (body.length < 8) throw new Error("Paste at least 8 characters of a public post.");
  if (String(text).length > 4000) throw new Error("Keep the post under 4,000 characters, as in training.");
  return body;
}

if (typeof module !== "undefined") module.exports = { tokens, score, detectCoin, targetWindow, validate };

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const setText = (node, text) => { node.replaceChildren(); node.append(document.createTextNode(String(text))); };
  const render = result => {
    const output = $("live-result");
    const pct = value => (value * 100).toFixed(1) + "%";
    const metrics = window.HYPECHECK_LIVE_MODEL.metrics.later_time;
    const unseen = window.HYPECHECK_LIVE_MODEL.metrics.unseen_coin_market;
    output.replaceChildren();
    const add = (heading, body) => {
      const h = document.createElement("h3"), d = document.createElement("p");
      setText(h, heading); setText(d, body); output.append(h, d);
    };
    add("Estimated chance the next UTC day closes up", pct(result.probability_up) + " — " + result.direction);
    add("Forecast day", result.target.start + " to " + result.target.end + " (end exclusive)");
    add("Coin-only baseline", pct(result.coin_baseline_probability_up) + (result.coin_known
      ? " — the same pipeline without the wording block"
      : " — this coin was held out of training, so no coin term applies and the forecast uses wording only"));
    add("Words pushing this model UP", result.toward_up.map(c => c.term).join(", ") || "none in the fitted vocabulary");
    add("Words pushing this model DOWN", result.toward_down.map(c => c.term).join(", ") || "none in the fitted vocabulary");
    add("Evidence for trusting this", "Live model accuracy " + pct(metrics["Live model"].accuracy) +
      " later dates and " + pct(unseen["Live model"].accuracy) + " unseen coins, versus coin-only " +
      pct(metrics["Coin only"].accuracy) + " and " + pct(unseen["Coin only"].accuracy) + ". No demonstrated forecasting edge.");
  };
  const run = () => {
    const output = $("live-result"), status = $("live-status");
    try {
      const text = validate($("live-text").value);
      const artifact = window.HYPECHECK_LIVE_MODEL;
      if (!artifact) throw new Error("Model unavailable: live-model.json did not load.");
      const detected = detectCoin(artifact, text);
      const coin = detected || $("live-coin").value;
      const result = score(artifact, text, coin);
      result.target = targetWindow(new Date());
      render(result);
      setText(status, detected ? "Coin detected from the post: " + coin : "No cashtag found; using " + coin + ".");
      status.hidden = false;
    } catch (error) {
      output.replaceChildren();
      setText(status, error.message);
      status.hidden = false;
    }
  };
  const start = () => {
    const button = $("live-run");
    fetch("live-model.json").then(response => {
      if (!response.ok) throw new Error("live-model.json returned " + response.status);
      return response.json();
    }).then(artifact => {
      window.HYPECHECK_LIVE_MODEL = artifact;
      const select = $("live-coin");
      select.replaceChildren();
      for (const coin of artifact.coins) {
        const option = document.createElement("option");
        option.value = coin;
        option.textContent = coin + (artifact.trained_coins.includes(coin) ? "" : " (held out of training)");
        select.append(option);
      }
      button.disabled = false;
      setText(button, "Get forecast");
      setText($("live-status"), "Model ready: " + artifact.training.snapshots + " training batches, " +
        artifact.model.blocks.find(b => b.name === "text").idf.length + " fitted terms. Runs entirely in your browser.");
      $("live-status").hidden = false;
      button.addEventListener("click", run);
      $("live-text").addEventListener("input", () => {
        const detected = detectCoin(artifact, $("live-text").value);
        if (detected) select.value = detected;
      });
    }).catch(error => {
      button.disabled = true;
      setText($("live-status"), "Model unavailable: " + error.message + ". No forecast is shown rather than a guess.");
      $("live-status").hidden = false;
    });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
}

if (typeof process !== "undefined" && process.argv.includes("--self-test")) {
  const fs = require("fs"), path = require("path");
  const dir = __dirname;
  const artifact = JSON.parse(fs.readFileSync(path.join(dir, "live-model.json"), "utf8"));
  const cases = JSON.parse(fs.readFileSync(path.join(dir, "live-parity.json"), "utf8"));
  let worst = 0;
  for (const item of cases) worst = Math.max(worst, Math.abs(score(artifact, item.text, item.coin).probability_up - item.expected));
  const coins = artifact.coins.filter(c => detectCoin(artifact, "big $ " + c + " move"));
  const problems = [];
  if (worst > 1e-9) problems.push("browser and Python probabilities differ by " + worst);
  if (coins.length !== artifact.coins.length) problems.push("coin detection missed " + artifact.coins.filter(c => !coins.includes(c)).join(", "));
  if (score(artifact, "short", "DOGE") && false) problems.push("unreachable");
  try { validate("tiny"); problems.push("short text was accepted"); } catch (error) { /* expected */ }
  if (problems.length) { console.error("live.js self-test FAILED: " + problems.join("; ")); process.exit(1); }
  console.log("live.js self-test passed: " + cases.length + " Python parity cases (max diff " + worst.toExponential(2) +
    "), coin detection for " + artifact.coins.length + " tickers, short input rejected.");
}
