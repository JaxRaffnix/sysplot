import json
import re
import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

spec: str = data["project"]["requires-python"]

# 1. Parse minimum version with a safety guard
m = re.search(r">=\s*(\d+)\.(\d+)", spec)
if not m:
    raise ValueError(f"Could not parse minimum Python version from: '{spec}'")

min_major, min_minor = int(m.group(1)), int(m.group(2))

# 2. Parse maximum version with an explicit tuple fallback
m2 = re.search(r"<\s*(\d+)\.(\d+)", spec)
if m2:
    max_major, max_minor = int(m2.group(1)), int(m2.group(2))
else:
    max_major, max_minor = 3, 15

# 3. Build version list
versions = [f"{min_major}.{i}" for i in range(min_minor, max_minor)]

print(json.dumps(versions))