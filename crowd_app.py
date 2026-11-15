"""Real Colab multi-post input; three panels distinguish sentiment, forecast and evidence."""
import html
from sample_pipeline import batch_check,COINS

def panels(result,report):
 esc=lambda v:html.escape(str(v))
 s=result["sample_sentiment"];q=result["data_quality"]
 crowd="<h3>Crowd Pulse — supplied sample only</h3>"+"<br>".join(esc(k)+": "+format(s[k],".1%") for k in ["bullish","bearish","neutral"])
 crowd+="<p>Estimated language sentiment, not a market-up probability. Disagreement: "+format(s["disagreement"],".3f")+"</p>"
 forecast="<h3>Market Forecast</h3><p>Estimated P(UP): <strong>"+format(result["probability_up"],".1%")+"</strong> · "+esc(result["direction"])+"</p>"
 forecast+="<p>Target UTC: "+esc(result["target_start_utc"])+" → "+esc(result["target_end_utc"])+" (end exclusive).</p><p>Coin-only baseline P(UP): "+format(result["baseline_probability_up"],".1%")+". Probabilities are not independently calibrated.</p>"
 for label,key in [("Words pushing this fitted model UP","toward_up"),("Words pushing this fitted model DOWN","toward_down")]:
  forecast+="<p><strong>"+label+":</strong> "+", ".join(esc(x["word"]) for x in result[key])+"</p>"
 forecast+="<p>Word contributions explain the model, not the effect caused by a post.</p>"
 evidence="<h3>No demonstrated forecasting edge</h3><p>Supplied posts: "+esc(q["unique_post_count"])+"; duplicates removed: "+esc(q["duplicates_removed"])+"; excluded outside window: "+esc(q["excluded_outside_window"])+".</p><p>Timing: "+esc(q["timing_status"])+". Source authenticity and complete X coverage are not verified.</p>"
 for name,label in [("later_time","Later dates"),("unseen_coin_market","Unseen-coin markets")]:
  m=report["tests"][name];evidence+="<p>"+label+": word-balanced "+format(m["Model 2"]["accuracy"],".2%")+" vs coin-only "+format(m["Coin only"]["accuracy"],".2%")+".</p>"
 if q["batch_size_outside_training_range"]:evidence+="<p><strong>Batch size is outside the observed training range. This forecast is extrapolating.</strong></p>"
 evidence+="<p>Reused retrospective data; not independent confirmation. Not financial advice. No live X feed is connected.</p>"
 return crowd,forecast,evidence

def build_app(artifact):
 import ipywidgets as w
 from IPython.display import display
 posts=w.Textarea(placeholder="Blank line starts a NEW post. Use CSV with a text column to preserve blank lines inside a post.",layout=w.Layout(width="100%",height="180px"))
 coin=w.Dropdown(options=list(COINS),description="Memecoin")
 form=w.Dropdown(options=[("Public posts","posts"),("CSV text","csv")],description="Input")
 cutoff=w.Text(placeholder="Blank = current submission time; historical mode requires timezone-aware ISO time",description="UTC cutoff",layout=w.Layout(width="100%"))
 button=w.Button(description="Analyse batch",button_style="primary")
 error=w.HTML();crowd=w.HTML();forecast=w.HTML();evidence=w.HTML();state={}
 def run(_):
  button.disabled=True;error.value="";state.clear()
  try:
   result=batch_check(posts.value,coin.value,cutoff.value.strip() or None,artifact,input_format=form.value)
   state["result"]=result;crowd.value,forecast.value,evidence.value=panels(result,artifact["report"])
  except Exception as exc:
   crowd.value=forecast.value=evidence.value="";error.value="<p>Forecast blocked: "+html.escape(str(exc))+"</p>"
  finally:button.disabled=False
 button.on_click(run)
 ui=w.VBox([w.HTML("<h2>HypeCheck — test the crowd</h2><p>Only public content you are allowed to use. No claim that the supplied sample represents all of X.</p>"),coin,form,posts,cutoff,button,error,crowd,forecast,evidence])
 return ui,state,{"posts":posts,"coin":coin,"format":form,"cutoff":cutoff,"button":button,"error":error,"crowd":crowd,"forecast":forecast,"evidence":evidence}
