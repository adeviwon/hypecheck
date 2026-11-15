"""Compact loader tests. Fixtures are not study data."""
import hashlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import colab_setup as setup
class SetupTests(unittest.TestCase):
 def test_hash_mismatch_rejected(self):
  with patch('urllib.request.urlopen',return_value=io.BytesIO(b'incorrect')):
   with self.assertRaisesRegex(ValueError,'SHA-256'):setup.fetch_verified('fixture',hashlib.sha256(b'correct').hexdigest())
 def test_cached_verified_file_needs_no_network(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'fixture').write_bytes(b'correct')
   with patch.dict(setup.FILE_DIGESTS,{'fixture':hashlib.sha256(b'correct').hexdigest()},clear=True),patch('urllib.request.urlopen') as request:
    setup.prepare(root=d);request.assert_not_called()
 def test_failed_download_writes_no_code(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(setup.FILE_DIGESTS,{'fixture.py':hashlib.sha256(b'correct').hexdigest()},clear=True),patch('urllib.request.urlopen',return_value=io.BytesIO(b'incorrect')):
   with self.assertRaises(ValueError):setup.prepare(root=d)
   self.assertFalse(Path(d,'fixture.py').exists())
 def test_recompute_removes_sentiment_cache(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(setup.FILE_DIGESTS,{},clear=True):
   p=Path(d,'data/expanded-sentiment.json');p.parent.mkdir();p.write_text('{}')
   setup.prepare(root=d,recompute_sentiment=True);self.assertFalse(p.exists())
if __name__=='__main__':unittest.main()
