"""Validated user-supplied post batches; never claim a complete or authentic X feed."""
import csv,io,re
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
COINS=("DOGE","SHIB","PEPE","BONK","FLOKI","WIF","TRUMP","NEIRO","PENGU","PNUT","TURBO","BOME","MEME")

def utc(value):
 try:t=value if isinstance(value,datetime) else datetime.fromisoformat(str(value).replace("Z","+00:00"))
 except (TypeError,ValueError) as exc:raise ValueError("Invalid ISO timestamp") from exc
 if t.tzinfo is None or t.utcoffset() is None:raise ValueError("An explicit timezone is required")
 return t.astimezone(timezone.utc)

def iso(t):return t.isoformat().replace("+00:00","Z")

@dataclass(frozen=True)
class Snapshot:
 coin:str
 cutoff_utc:str
 target_start_utc:str
 target_end_utc:str
 posts:tuple
 audit:dict

def build_snapshot(records,coin,cutoff=None):
 if coin not in COINS:raise ValueError("Choose a supported memecoin")
 if not isinstance(records,(list,tuple)) or not 1<=len(records)<=200:raise ValueError("Supply 1–200 public posts")
 cutoff=utc(cutoff) if cutoff is not None else datetime.now(timezone.utc)
 earliest=cutoff-timedelta(hours=24);unique={}
 for record in records:
  if not isinstance(record,dict) or not isinstance(record.get("text"),str):raise ValueError("Every post requires text")
  text=record["text"].strip()
  if not 8<=len(text)<=4000:raise ValueError("Each post must contain 8–4000 characters")
  times=[]
  for key in ["published_at_utc","available_at_utc"]:
   value=record.get(key)
   if value is None or (isinstance(value,str) and not value.strip()):continue
   times.append(utc(value))
  available=max(times) if times else None
  if available is not None and available>cutoff:raise ValueError("A post is in the future relative to the cutoff")
  key=" ".join(text.split());item={"text":text,"available_at_utc":iso(available) if available else None}
  if key not in unique:unique[key]=item
  elif available is not None:
   prior=unique[key]["available_at_utc"]
   if prior is None or available<utc(prior):unique[key]=item
 posts=[];outside=0
 for item in unique.values():
  if item["available_at_utc"] is not None and utc(item["available_at_utc"])<earliest:outside+=1;continue
  posts.append(item)
 if not posts:raise ValueError("No usable posts remain inside the supplied window")
 unknown=sum(p["available_at_utc"] is None for p in posts)
 status="unverified_user_supplied" if unknown==len(posts) else "mixed_unverified_timing" if unknown else "provided_timestamps_checked"
 start=cutoff.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
 audit={"input_post_count":len(records),"unique_post_count":len(posts),"duplicates_removed":len(records)-len(unique),
  "excluded_outside_window":outside,"unknown_timestamps":unknown,"timing_status":status,
  "source_authenticity_verified":False,"platform_coverage":"unknown",
  "scope":"Supplied sample only; source authenticity and completeness are not independently verified"}
 return Snapshot(coin,iso(cutoff),iso(start),iso(start+timedelta(days=1)),tuple(posts),audit)

def parse_pasted(text,coin,cutoff=None):
 if not isinstance(text,str) or len(text)>1000000:raise ValueError("Public-post input is too large or is not text")
 records=[{"text":p} for p in re.split(r"\n\s*\n",text.strip()) if p.strip()]
 return build_snapshot(records,coin,cutoff)

def parse_csv(text,coin,cutoff=None):
 if not isinstance(text,str) or len(text)>1000000:raise ValueError("CSV input is too large or is not text")
 reader=csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
 if not reader.fieldnames or "text" not in reader.fieldnames:raise ValueError("CSV requires a text column")
 records=[]
 for row in reader:
  records.append({k:row.get(k) for k in ["text","published_at_utc","available_at_utc"]})
  if len(records)>200:raise ValueError("Supply at most 200 public posts")
 return build_snapshot(records,coin,cutoff)
