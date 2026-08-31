import os
from datetime import date
from pathlib import Path
import tomllib


project = tomllib.loads(Path("pyproject.toml").read_text())
package_version = project["project"]["version"]

citation_lines = Path("CITATION.cff").read_text().splitlines()
citation_version = next(
    line.split(":", 1)[1].strip().strip("'\"")
    for line in citation_lines
    if line.startswith("version:")
)
citation_date = next(
    line.split(":", 1)[1].strip().strip("'\"")
    for line in citation_lines
    if line.startswith("date-released:")
)
date.fromisoformat(citation_date)

if citation_version != package_version:
    raise SystemExit(
        "Release version mismatch: "
        f"package={package_version!r}, citation={citation_version!r}"
    )

release_tag = os.environ.get("RELEASE_TAG")
expected_tag = f"v{package_version}"
if release_tag is not None and release_tag != expected_tag:
    raise SystemExit(
        f"Release tag mismatch: tag={release_tag!r}, expected={expected_tag!r}"
    )

print(
    f"Release metadata matches version {package_version} "
    f"with release date {citation_date}."
)
