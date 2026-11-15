"""Judge-facing tables and explanations. Presentation only: never alter model inputs."""
import pandas as pd
MODEL_NAMES={"Model 1":"Model 1 — word-based","Model 2":"Model 2 — balanced word-based","Character trial":"Character-pattern alternative","Coin only":"Coin-only baseline","Always up":"Always predict a rise","Training majority":"Always predict the training majority","Training prevalence":"Training-frequency probability","Previous completed day":"Copy the last completed day"}
TEST_NAMES={"later_time":"Later dates: established coins","unseen_coin_market":"New coins: excluded from training"}

def outcome(row):
 if "actual_return" in row and pd.notna(row["actual_return"]):
  r=float(row["actual_return"]);label="Rose" if r>0 else "Fell" if r<0 else "Unchanged"
  return label+" ("+format(r,"+.2%")+")"
 return "Rose" if int(row["actual_up"])==1 else "Did not rise"

def examples_table(frame,n=5):
 records=[]
 for _,r in frame.head(n).iterrows():
  text=" ".join(str(r["text"]).split());text=text[:137]+"…" if len(text)>140 else text
  records.append({"Snapshot time (UTC)":pd.Timestamp(r["cutoff"]).strftime("%d %b %Y, %H:%M"),"Forecast day (UTC)":pd.Timestamp(r["date"]).strftime("%d %b %Y"),"Coin":r["coin"],"Posts":int(r["post_count"]),"Bullish language":format(r["bullish"],".1%"),"Bearish language":format(r["bearish"],".1%"),"Actual next-day result":outcome(r),"Example wording":text})
 return pd.DataFrame(records)

def examples_findings(frame):
 r=frame.iloc[0];date=pd.Timestamp(r["date"]).strftime("%d %b %Y")
 text="For the first "+str(r["coin"])+" snapshot, the language model gave an average bullish score of "+format(r["bullish"],".1%")+" across "+str(int(r["post_count"]))+" post(s). The coin "+outcome(r).lower()+" on "+date+". This bullish score describes the wording: it is NOT the probability that the price will rise."
 contrary=frame[(frame.bullish>.7)&(frame.actual_up==0)]
 if len(contrary):
  x=contrary.iloc[0];text+=" Another "+str(x["coin"])+" snapshot scored "+format(x["bullish"],".1%")+" bullish, but the coin "+outcome(x).lower()+" the next day. Bullish language and a rising market are different outcomes."
 text+=" Rows for the same coin and forecast day can contain overlapping posts; they are not independent trading examples."
 return text

def scores_table(scores):
 rows=[]
 for name,s in scores.items():
  cm=s.get("confusion_coin_day_weighted")
  precision="No UP calls" if cm is not None and cm[0][1]+cm[1][1]==0 else format(s["up_precision"],".2%")
  rows.append({"Model / rule":MODEL_NAMES.get(name,name),"Correct directions":format(s["accuracy"],".2%"),"UP calls that were right":precision,"Rising days caught":format(s["up_recall"],".2%"),"Probability error ↓":format(s["brier"],".4f"),"Distinct coin-days":s["coin_days"]})
 return pd.DataFrame(rows)

def confusion_table(metric):
 cm=metric["confusion_coin_day_weighted"]
 return pd.DataFrame({"What actually happened":["Did not rise","Rose"],"Model predicted no rise":[format(cm[0][0],".1f"),format(cm[1][0],".1f")],"Model predicted a rise":[format(cm[0][1],".1f"),format(cm[1][1],".1f")]})

def errors_table(frame):
 return pd.DataFrame([{ "Forecast day":pd.Timestamp(r["date"]).strftime("%d %b %Y"),"Coin":r["coin"],"Mistake":"Missed a rise" if r["actual_up"]==1 else "Predicted a rise that did not happen","Estimated chance of UP":format(r["estimated_up"],".1%"),"Actual result":outcome(r),"Words influencing this call":r["actual_word_reasons"],"Input wording":" ".join(r["text"].split())[:160]} for _,r in frame.iterrows()])

def results_table(report):
 return pd.DataFrame([{ "Model / rule":MODEL_NAMES.get(name,name),"Later dates — correct":format(row["accuracy"],".2%"),"New coins — correct":format(report["tests"]["unseen_coin_market"][name]["accuracy"],".2%")} for name,row in report["tests"]["later_time"].items()])

def model_findings(scores,name="Model 1"):
 m=scores[name];base=scores["Coin only"]
 return MODEL_NAMES[name]+" predicted "+format(m["accuracy"],".2%")+" of directions correctly, compared with "+format(base["accuracy"],".2%")+" for the coin-only baseline. Of its UP calls, "+format(m["up_precision"],".2%")+" were correct; it caught "+format(m["up_recall"],".2%")+" of actual rising days. These are coin-day-weighted rates, not counts of independent trades. "+("It did not beat the no-text baseline." if m["accuracy"]<=base["accuracy"] else "Its higher point estimate alone does not establish a reliable edge.")

def balance_findings(report):
 parts=[]
 for key,label in TEST_NAMES.items():
  one=report["tests"][key]["Model 1"];two=report["tests"][key]["Model 2"]
  delta=100*(two["accuracy"]-one["accuracy"]);recall=100*(two["up_recall"]-one["up_recall"])
  parts.append(label+": balanced training changed accuracy from "+format(one["accuracy"],".2%")+" to "+format(two["accuracy"],".2%")+" ("+format(delta,"+.2f")+" percentage points), and rising-day recall by "+format(recall,"+.2f")+" points.")
 return " ".join(parts)+" Catching more rising days can come with more false UP calls. Balanced weights did not create a dependable forecasting advantage."

def overall_findings(report):
 text=[]
 for key,label in TEST_NAMES.items():
  x=report["tests"][key];text.append(label+": the balanced model was correct "+format(x["Model 2"]["accuracy"],".2%")+" of the time versus "+format(x["Coin only"]["accuracy"],".2%")+" without post wording or sentiment.")
 return " ".join(text)+" Neither result justifies following the crowd or trading on the model alone. The tested posts did not add useful direction information in this experiment; that does not prove all social information is useless. We have not measured profits or established that posts cause price movements."
