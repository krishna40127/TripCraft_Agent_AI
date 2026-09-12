import sys
from pathlib import Path

# Let tests `import backend`, `import guardrails`, etc. without installing
# the project as a package -- this is a flat single-app layout, not a
# distributable library.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
