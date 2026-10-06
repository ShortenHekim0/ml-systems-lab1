import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "results" / "versions.txt"


def package_names():
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split("==")[0])
    return names


lines = [f"Python {platform.python_version()}", f"Platform: {platform.platform()}", ""]
for name in package_names():
    try:
        v = version(name)
    except PackageNotFoundError:
        v = "NOT INSTALLED"
    lines.append(f"{name}=={v}")

text = "\n".join(lines)
print(text)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(text + "\n", encoding="utf-8")