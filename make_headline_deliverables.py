from pathlib import Path
import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).resolve().parent/"build_headline_colab.py")],check=True)
