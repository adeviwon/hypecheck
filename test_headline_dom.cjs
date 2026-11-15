/* Execute the shipped replay script against its real JSON artifacts. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const report=JSON.parse(fs.readFileSync('docs/headline-report.json'));
const records=JSON.parse(fs.readFileSync('docs/headline-predictions.json'));
const html=fs.readFileSync('docs/headlines.html','utf8');
class E{constructor(){this.children=[];this.textContent='';this.value='';this.disabled=false;}append(...items){for(const item of items){this.children.push(item);if(item.option&&!this.value)this.value=item.value;}}replaceChildren(){this.children=[];}}
async function harness(fail=false){
 const ids=['cards','metrics','period','coin','day','forecast','reveal','outcome','kinds','error'];
 const els=Object.fromEntries(ids.map(id=>[id,new E()]));
 const context={document:{createElement:()=>new E(),getElementById:id=>{assert(els[id],id);return els[id];}},
  Option:function(label,value){this.option=true;this.textContent=label;this.value=value;},
  fetch:async url=>({ok:!fail,json:async()=>url==='headline-report.json'?report:records}),console};
 vm.runInNewContext(fs.readFileSync('docs/headlines.js','utf8'),context);
 await new Promise(resolve=>setImmediate(resolve));
 return els;
}
(async()=>{
 const e=await harness();assert.equal(e.error.textContent,'');
 assert.equal(e.metrics.children.length,7);assert.equal(e.cards.children.length,4);assert.equal(e.kinds.children.length,4);
 assert.equal(e.coin.children.length,report.protocol.coins.length);assert.equal(e.day.children.length,report.protocol.splits.test.dates);
 assert.equal(e.outcome.textContent,'Outcome hidden.');e.reveal.onclick();assert.match(e.outcome.textContent,/Actual:/);
 e.coin.value='DOGE';e.day.value='2026-10-21';e.day.onchange();assert.equal(e.outcome.textContent,'Outcome hidden.');
 e.reveal.onclick();const r=records.find(x=>x.coin==='DOGE'&&x.date.startsWith('2026-10-21'));
 assert.equal(e.outcome.textContent,'Actual: '+(r.actual_up?'up':'not up')+' · open-to-close return '+(100*r.actual_return).toFixed(2)+'%');
 const bad=await harness(true);assert.match(bad.error.textContent,/Could not load/);assert.equal(bad.reveal.disabled,true);
 console.log('Headline replay DOM passed: measured metrics, expanded coin/date support, reveal/reset, failed-fetch handling.');
})().catch(e=>{console.error(e);process.exit(1);});
