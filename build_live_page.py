"""Build docs/live.html from the exported live model so page prose cannot drift from measured results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
model = json.loads((ROOT / "docs" / "live-model.json").read_text())
L, U = model["metrics"]["later_time"], model["metrics"]["unseen_coin_market"]
pct = lambda value: f"{value * 100:.2f}%"
html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>HypeCheck - paste a tweet, get the forecast</title>
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="live.css">
  <script src="live.js" defer></script>
</head>
<body>
<main>
  <p><a href="./">HypeCheck</a> / live forecast</p>
  <h1>Paste tweet here</h1>
  <p class="lede">One box. Paste a public X post about a memecoin and read the trained next-day direction forecast. The model runs in your browser: nothing is uploaded, and no server is involved.</p>
  <label for="live-text">Paste tweet here</label>
  <textarea id="live-text" maxlength="4000" placeholder="Paste tweet here" autocomplete="off" spellcheck="false"></textarea>
  <div class="controls">
    <label for="live-coin">Memecoin</label>
    <select id="live-coin"></select>
    <button id="live-run" disabled>Loading model...</button>
  </div>
  <p id="live-status" hidden></p>
  <section id="live-result" aria-live="polite"></section>

  <aside class="honest">
    <strong>No demonstrated forecasting edge.</strong> This live model uses the wording of the post plus the coin; on held-out data it scores
    {pct(L["Live model"]["accuracy"])} on later dates and {pct(U["Live model"]["accuracy"])} on coins excluded from training,
    against a coin-only baseline of {pct(L["Coin only"]["accuracy"])} and {pct(U["Coin only"]["accuracy"])}.
    A bullish-sounding post is not a reliable price forecast, and no forecast here is financial advice.
  </aside>

  <aside class="method">
    <strong>What this page runs, exactly.</strong> The submitted model adds a frozen FinTwitBERT sentiment block, which needs the Colab
    runtime and cannot run in a static page. This page runs the identical wording + coin pipeline without that block, trained on the same
    {model["training"]["snapshots"]:,} batches ({model["training"]["coin_days"]} coin-days, {model["training"]["post_count_range"][0]}-{model["training"]["post_count_range"][1]} posts per batch)
    with the same coin-day weighting and target: {model["target"]}. Word contributions are multiplied out from the fitted vocabulary, so they explain
    this model rather than the effect a post had on a price.
    <table>
      <tr><th>Method</th><th>Later dates</th><th>Unseen coins</th></tr>
      <tr><td>Live model (wording + coin)</td><td>{pct(L["Live model"]["accuracy"])}</td><td>{pct(U["Live model"]["accuracy"])}</td></tr>
      <tr><td>Coin only</td><td>{pct(L["Coin only"]["accuracy"])}</td><td>{pct(U["Coin only"]["accuracy"])}</td></tr>
      <tr><td>Always up</td><td>{pct(L["Always up"]["accuracy"])}</td><td>{pct(U["Always up"]["accuracy"])}</td></tr>
      <tr><td>Training majority</td><td>{pct(L["Training majority"]["accuracy"])}</td><td>{pct(U["Training majority"]["accuracy"])}</td></tr>
      <tr><td>Previous completed day</td><td>{pct(L["Previous completed day"]["accuracy"])}</td><td>{pct(U["Previous completed day"]["accuracy"])}</td></tr>
    </table>
    {model["test_reuse"]} The Colab notebook carries the full model, the frozen sentiment block, the error analysis and the complete results.
  </aside>

  <footer>
    <p><a href="https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb">Open the full model in Colab</a>
    &middot; <a href="https://github.com/adeviwon/hypecheck">Repository</a>
    &middot; <a href="https://github.com/adeviwon/hypecheck/blob/main/docs/writeup-current.md">Write-up</a></p>
    <p>{model["disclaimer"]}</p>
  </footer>
</main>
</body>
</html>
"""
(ROOT / "docs" / "live.html").write_text(html)
print("wrote docs/live.html", len(html), "chars")
