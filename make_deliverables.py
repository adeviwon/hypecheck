"""Generate deliverables for the restored headline-to-direction study."""
import subprocess, sys
from pathlib import Path
if __name__ == '__main__':
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'make_headline_deliverables.py')],check=True)
