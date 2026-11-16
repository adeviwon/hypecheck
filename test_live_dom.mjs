import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("./docs/live.html", import.meta.url), "utf8");
const source = readFileSync(new URL("./docs/live.js", import.meta.url), "utf8");
const model = JSON.parse(readFileSync(new URL("./docs/live-model.json", import.meta.url), "utf8"));
const problems = [];
const check = (condition, message) => { if (!condition) problems.push(message); };

for (const id of ["live-text", "live-coin", "live-run", "live-status", "live-result"]) {
  check(html.includes(`id="${id}"`), `live.html is missing #${id}`);
}
check(html.includes('src="live.js"'), "live.html does not load live.js");
check(html.includes('placeholder="Paste tweet here"'), "live.html lost the single input box label");
check(html.includes('http-equiv="Content-Security-Policy"'), "live.html is missing its CSP");
check(!/https?:\/\/(?!colab\.research|github\.com|adeviwon\.github)/.test(html.replace(/href="[^"]*"/g, "")), "live.html references an unexpected remote host");

// Page prose must equal the measured artifact, not a hand-typed copy.
const pct = value => (value * 100).toFixed(2) + "%";
for (const [label, value] of [["later", model.metrics.later_time["Live model"].accuracy], ["unseen", model.metrics.unseen_coin_market["Live model"].accuracy],
  ["later coin only", model.metrics.later_time["Coin only"].accuracy], ["unseen coin only", model.metrics.unseen_coin_market["Coin only"].accuracy]]) {
  check(html.includes(pct(value)), `live.html does not state the measured ${label} accuracy ${pct(value)}`);
}
check(html.includes(model.disclaimer), "live.html is missing the prototype disclaimer");
check(html.includes("No demonstrated forecasting edge"), "live.html does not state the negative result");

// Minimal DOM so the page script can run outside a browser.
function makeEl(tag) {
  const el = {
    tagName: tag, children: [], value: "", textContent: "", hidden: false, disabled: false, listeners: {},
    append(...nodes) {
      this.children.push(...nodes);
      if (tag === "select" && !this.value) this.value = String(nodes[0]?.value ?? "");
    },
    replaceChildren(...nodes) { this.children = nodes; },
    addEventListener(type, handler) { this.listeners[type] = handler; },
    setText(text) { this.textContent = String(text); },
  };
  return el;
}
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]);
const elements = new Map(ids.map(id => [id, makeEl("div")]));
const document = {
  readyState: "complete",
  getElementById: id => elements.get(id) ?? null,
  createElement: makeEl,
  createTextNode: text => ({ text: String(text) }),
  addEventListener() {},
};
const context = { document, console, process: { argv: ["node"] }, fetch: async url => ({ ok: url === "live-model.json", status: url === "live-model.json" ? 200 : 404, json: async () => model }) };
context.window = context;
vm.createContext(context);
vm.runInContext(source, context);
const flush = () => new Promise(resolve => setTimeout(resolve, 20));
await flush();

const render = node => node.children.map(child => child.text !== undefined ? child.text : (child.children?.length ? render(child) : child.textContent)).join(" ");
const status = elements.get("live-status");
const output = elements.get("live-result");
const button = elements.get("live-run");
check(button.disabled === false, "the forecast button never became enabled after the model loaded");
check(elements.get("live-coin").children.length === model.coins.length, "the coin selector was not populated from the artifact");
check(render(status).includes("Model ready"), "the page never reported the model as ready");

const post = "The last final infinity stone to my bags $PEPE - strong narrative, this will be a core holding of mine moving forward";
elements.get("live-text").value = post;
button.listeners.click();
const text = render(output);
check(render(status).includes("PEPE"), "a cashtag was not detected from the post");
check(/UP|DOWN \/ FLAT/.test(text), "no direction was rendered");
check(/\d{4}-\d{2}-\d{2} 00:00 UTC to \d{4}-\d{2}-\d{2} 00:00 UTC/.test(text), "the exact forecast window was not rendered");
check(text.includes("Coin-only baseline"), "the coin-only baseline was not shown beside the forecast");
check(text.includes("No demonstrated forecasting edge"), "the forecast was rendered without its evidence warning");

// A held-out coin must say so rather than pretend it has a coin term.
elements.get("live-text").value = "$TRUMP looks ready to run today, loading up here";
button.listeners.click();
const heldOut = render(output);
check(heldOut.includes("held out of training"), "a coin excluded from training was not flagged");
check(heldOut.includes("wording only"), "a held-out coin did not fall back to wording only");

// Corrupt model data must clear any earlier forecast, never render false certainty.
const intercept = model.model.intercept;
model.model.intercept = NaN;
button.listeners.click();
check(render(output) === "", "a corrupt model left a forecast visible");
check(render(status).includes("forecast blocked"), "a corrupt model was not rejected");
model.model.intercept = intercept;
elements.get("live-text").value = "__proto__ constructor DOGE public wording";
button.listeners.click();
check(!render(output).includes("NaN"), "inherited object keys corrupted the forecast");

// Bad input must block the forecast instead of producing a number.
elements.get("live-text").value = "hi";
button.listeners.click();
check(render(output) === "", "a too-short post still produced a forecast");
check(render(status).includes("8 characters"), "short input did not explain the minimum length");

console.log(`live page: ${model.coins.length} coins, ${model.training.snapshots} training batches, ${problems.length} problems`);
if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
console.log("live page test passed: model loads, cashtag detection, forecast window, baseline, evidence warning, held-out coin and rejected input.");
