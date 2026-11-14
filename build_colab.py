"""Generate the notebook for the restored headline-to-direction study."""
import subprocess, sys
from pathlib import Path
if __name__ == '__main__':
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'build_headline_colab.py')],check=True)
