
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
script = ROOT / "src" / "show_accuracy.py"
if not script.exists():
    print("Not found:", script)
    sys.exit(1)
subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
