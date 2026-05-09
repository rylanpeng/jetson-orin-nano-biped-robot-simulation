#!/bin/bash
# Patches genesis 0.4.0 installed in .venv to handle NVIDIA Jetson pre-release
# torch version strings (e.g. '2.5.0a0+...') which break genesis's int() parsing.
# Run this after any `uv sync` that reinstalls genesis.

set -e

python3 - <<'EOF'
import re

files = [
    ".venv/lib/python3.10/site-packages/genesis/utils/misc.py",
    ".venv/lib/python3.10/site-packages/genesis/__init__.py",
]

OLD_PATTERNS = [
    # misc.py / __init__.py TORCH_MPS pattern
    (
        r"tuple\(map\(int, torch\.__version__\.replace\(\"\+\", \"\.\"\)\.split\(\"\.\"\)\[:3\]\)\)",
        r'tuple(int(x) for x in re.split(r"[^0-9]+", torch.__version__) if x)[:3]',
    ),
    # __init__.py _IS_OLD_TORCH pattern
    (
        r"tuple\(map\(int, torch\.__version__\.split\(\"\.\"\)\[:2\]\)\)",
        r'tuple(int(x) for x in re.split(r"[^0-9]+", torch.__version__) if x)[:2]',
    ),
]

for path in files:
    with open(path, "r") as f:
        content = f.read()

    changed = False

    # Add 'import re' if missing
    if "import re\n" not in content:
        content = content.replace("import sys\n", "import re\nimport sys\n", 1)
        print(f"  [+] Added 'import re' to {path}")
        changed = True

    # Apply version string patches
    for old_pat, new_str in OLD_PATTERNS:
        new_content = re.sub(old_pat, new_str, content)
        if new_content != content:
            content = new_content
            print(f"  [+] Patched version parsing in {path}")
            changed = True

    if changed:
        with open(path, "w") as f:
            f.write(content)
    else:
        print(f"  [=] Already patched or no match: {path}")

print("Done.")
EOF
