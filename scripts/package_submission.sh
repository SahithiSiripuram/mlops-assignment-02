#!/usr/bin/env bash
# Build the submission zip: source code, configuration, and the trained model
# artifacts -- without the dataset, the virtualenv or the MLflow store.
#
#   bash scripts/package_submission.sh
#
# Produces AIMLCZG523_Assignment02_SahithiSiripuram.zip in the repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="AIMLCZG523_Assignment02_SahithiSiripuram"
ZIP="${ROOT}/${NAME}.zip"

if [[ ! -f models/model.pt ]]; then
  echo "error: models/model.pt is missing -- run 'make train' first." >&2
  exit 1
fi

rm -f "$ZIP"

zip -r "$ZIP" \
  src tests deployment scripts docs notebooks \
  .github .dvc .dvcignore dvc.yaml dvc.lock params.yaml \
  Dockerfile .dockerignore Makefile pyproject.toml \
  requirements.txt requirements-dev.txt \
  README.md .gitignore \
  models/model.pt models/metadata.json models/model.pkl \
  -x '*/__pycache__/*' '*.pyc' '*/.pytest_cache/*' '*/.ruff_cache/*' \
     '.dvc/cache/*' '.dvc/tmp/*' 'docs/screenshots/.gitkeep' \
  >/dev/null

echo "Created $(basename "$ZIP") ($(du -h "$ZIP" | cut -f1))"
echo
echo "Contents summary:"
unzip -l "$ZIP" | tail -1
echo
echo "Remember to add the screen recording (< 5 minutes) alongside this zip."
