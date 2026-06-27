import tomllib
import json
import re

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

spec = data["project"]["requires-python"]

m = re.search(r">=\s*(\d+)\.(\d+)", spec)
min_major, min_minor = int(m.group(1)), int(m.group(2))

m2 = re.search(r"<\s*(\d+)\.(\d+)", spec)
max_major, max_minor = int(m2.group(1)), int(m2.group(2)) if m2 else (3, 15)

versions = [
    f"{min_major}.{i}"
    for i in range(min_minor, max_minor)
]

print(json.dumps(versions))