"""Compact Colab setup: download immutable source/data and verify every file."""
import hashlib,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
SOURCE_REVISION="e396d50d1ac40eabd648f1cedc4d9316e80e7759"
BASE_URL=f"https://raw.githubusercontent.com/adeviwon/hypecheck/{SOURCE_REVISION}/"
FILE_DIGESTS={'pipeline.py': '58a80d95767c43b794bad48ed5818bf8b18d0ce49e5fed35e5201e509b7574c6', 'headline_study.py': 'a2b36d884a58be19f43c1871e65a962fceb1e84820dabd703eb8ed9f4bd24307', 'expanded_inputs.py': 'bbb6cd7c8179a12cedb36d5c5e70a8d3738b638f29ab7b71b81c29de9c314d81', 'test_headline_study.py': '61a1b14ca13e6b2ad1744b5aee6f382a68ee314211bc6a2e29187fa2d49a3cf3', 'test_expanded_inputs.py': '7a8415bfe90acd7390c8e2932ddf46c600d2a6025b9236a002edd7a32d8ce49f', 'data/expanded-memecoin-daily.csv': '4ea2b737ee19daff97414a444ec3ea5313b797b78705e1b56dbf055bd5d47a56', 'data/expanded-sentiment.json': 'cee6a0649aa3484750e972593e27dd8d966639d5a8a6b1a1829ededcdce80ac2'}

def fetch_verified(name,digest):
    with urllib.request.urlopen(BASE_URL+name,timeout=30) as response:
        payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=digest:
        raise ValueError('SHA-256 mismatch: '+name)
    return payload

def prepare(recompute_sentiment=False,root='.'):
    root=Path(root)
    missing=[(name,digest) for name,digest in FILE_DIGESTS.items()
             if not (root/name).exists() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    with ThreadPoolExecutor(max_workers=4) as pool:
        payloads=list(pool.map(lambda item:fetch_verified(*item),missing))
    for (name,_),payload in zip(missing,payloads):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    for name in ['data','docs','artifacts']:(root/name).mkdir(exist_ok=True)
    if recompute_sentiment:
        for name in ['data/expanded-sentiment.json','.cache/expanded-headlines/data/sentiment.json']:
            (root/name).unlink(missing_ok=True)
    print('Verified source and market data. Sentiment:', 'recompute' if recompute_sentiment else 'verified cache')
