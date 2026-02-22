#!/usr/bin/env bash
set -euo pipefail

# builds a self-contained zenodo bundle zip including submodule content
# run from the root of the code repo

BUNDLE_ROOT="rr_bundle"
ZIP_NAME="${BUNDLE_ROOT}.zip"

echo "[1/8] verifying we are in a git repo..."
git rev-parse --show-toplevel >/dev/null

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[2/8] updating submodules..."
git submodule update --init --recursive

echo "[3/8] capturing commit provenance..."
CODE_COMMIT="$(git rev-parse HEAD)"
SUB_PATH="resource/defending-against-authorship-attribution-corpus"
if [[ ! -d "$SUB_PATH" ]]; then
  echo "error: expected submodule path missing: $SUB_PATH" >&2
  exit 1
fi
DATA_COMMIT="$(git -C "$SUB_PATH" rev-parse HEAD)"

rm -rf "$BUNDLE_ROOT" "$ZIP_NAME"
mkdir -p "$BUNDLE_ROOT"

cat > "${BUNDLE_ROOT}/FROZEN_COMMITS.txt" <<EOF
code_repo_commit: ${CODE_COMMIT}
data_repo_commit: ${DATA_COMMIT}
submodule_path: ${SUB_PATH}
build_time_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "[4/8] copying working tree into bundle (includes checked-out submodule files)..."
# copy everything, but exclude git metadata and common local clutter
rsync -a ./ "${BUNDLE_ROOT}/" \
  --exclude '.git' \
  --exclude '.gitmodules' \
  --exclude '.gitignore' \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.pre-commit-config.yaml' \
  --exclude 'resource/webplotdigitizer' \
  --exclude 'resource/Drexel-AMT-Corpus/' \
  --exclude '.idea'

echo "[5/8] removing any nested git metadata copied from submodules..."
find "${BUNDLE_ROOT}" -name ".git" -type d -prune -exec rm -rf {} \; || true
find "${BUNDLE_ROOT}" -name ".git" -type f -delete || true

echo "[6/8] sanity checks..."
if [[ ! -f "${BUNDLE_ROOT}/README.md" ]]; then
  echo "warning: README.md not found in bundle root"
fi
if [[ ! -f "${BUNDLE_ROOT}/${SUB_PATH}/metadata.csv" ]]; then
  echo "warning: metadata.csv not found at ${BUNDLE_ROOT}/${SUB_PATH}/metadata.csv"
fi

echo "[7/8] creating zip..."
# -q to reduce spam; remove -q if you want verbose
zip -rq "$ZIP_NAME" "$BUNDLE_ROOT"

echo "[8/8] done."
echo "created: ${ZIP_NAME}"
echo "bundle dir: ${BUNDLE_ROOT}/"
