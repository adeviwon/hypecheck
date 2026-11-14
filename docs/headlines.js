'use strict';
function text(tag,value){const e=document.createElement(tag);e.textContent=value;return e;}
function row(target,values){const tr=document.createElement('tr');values.forEach(v=>tr.append(text('td',v)));document.getElementById(target).append(tr);}
const pct=x=>(100*x).toFixed(2)+'%';let records=[];
function showForecast(){
 const c=document.getElementById('coin').value,d=document.getElementById('day').value;
 const item=records.find(r=>r.coin===c&&r.date.slice(0,10)===d);if(!item)throw Error('No replay row for selected key');
 const out=document.getElementById('forecast');out.replaceChildren();
 const dl=document.createElement('dl');
 const values=[['Model probability of up (uncalibrated)',pct(item['Sentiment + coin'])],['Model direction',item['Sentiment + coin']>=.5?'Up':'Not up'],['Previous-day direction',item.previous_day?'Up':'Not up'],['Eligible archive posts',String(item.post_count)]];
 for(const [key,value]of values){dl.append(text('dt',key),text('dd',value));}out.append(dl);
 document.getElementById('outcome').textContent='Outcome hidden.';
 document.getElementById('reveal').onclick=()=>{document.getElementById('outcome').textContent='Actual: '+(item.actual_up?'up':'not up')+' · open-to-close return '+pct(item.actual_return);};
}
async function load(){
 const responses=await Promise.all([fetch('headline-report.json'),fetch('headline-predictions.json')]);
 if(responses.some(r=>!r.ok))throw Error('Could not load measured results. Open the notebook instead.');
 const [r,p]=await Promise.all(responses.map(x=>x.json()));records=p;
 if(p.length!==r.protocol.splits.test.rows)throw Error('Prediction count does not match measured report.');
 const s=r.protocol.splits.test;document.getElementById('period').textContent=s.start+'–'+s.end+' · '+s.rows+' coin-days · '+s.dates+' UTC dates.';
 const order=['Always up','Previous-day direction','Training majority','Sentiment + coin','Coin only','Market only','Market + sentiment'];
 for(const name of order){const m=r.metrics[name];row('metrics',[name,pct(m.accuracy),m.brier.toFixed(4)]);}
 for(const name of ['Always up','Previous-day direction','Sentiment + coin','Training majority']){
  const card=document.createElement('div');card.className='card';card.append(text('div',name));const value=text('div',pct(r.metrics[name].accuracy));value.className='big';card.append(value);document.getElementById('cards').append(card);
 }
 for(const k of r.kinds){const v=k.paired_brier_vs_market;row('kinds',[k.kind,k.test_coin_days_with_kind,v.mean.toFixed(6),'['+v.interval95.map(x=>x.toFixed(6)).join(', ')+']',k.holm_adjusted_p.toFixed(3)]);}
 const coins=[...new Set(p.map(x=>x.coin))].sort(),days=[...new Set(p.map(x=>x.date.slice(0,10)))].sort();
 for(const c of coins)document.getElementById('coin').append(new Option(c,c));for(const d of days)document.getElementById('day').append(new Option(d,d));
 document.getElementById('coin').onchange=showForecast;document.getElementById('day').onchange=showForecast;showForecast();
}
load().catch(e=>{document.getElementById('error').textContent=e.message;document.getElementById('reveal').disabled=true;});
