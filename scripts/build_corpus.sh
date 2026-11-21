#!/usr/bin/env bash
# Build a pinned, reproducible DocDrift corpus:
#   <out>/            CPython Lib/ (stdlib; tests etc. are skipped by docdrift.corpus)
#   <out>/site-packages/<pkg>   pinned third-party packages (one project each)
# Usage:  bash scripts/build_corpus.sh [OUT_DIR]   then   DOCDRIFT_CORPUS=OUT_DIR bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-corpus_src}
CPYTHON_VERSION=3.13.1
PACKAGES=(
  # packages behind the report's wild-scan findings
  jsonschema==4.23.0 pathspec==0.12.1 pymdown-extensions==10.12 networkx==3.4.2
  scikit-learn==1.6.1 matplotlib==3.10.0
  # well-documented, widely used libraries (mixed docstring styles)
  numpy==2.2.1 pandas==2.2.3 scipy==1.15.0 requests==2.32.3 click==8.1.8
  rich==13.9.4 attrs==24.3.0 packaging==24.2 jinja2==3.1.5 markdown==3.7
  pygments==2.19.1 werkzeug==3.1.3 flask==3.1.0 httpx==0.28.1
  django==5.1.4 sqlalchemy==2.0.36 statsmodels==0.14.4 sphinx==8.1.3
  docutils==0.21.2 setuptools==75.6.0 pytest==8.3.4 tornado==6.4.2
  aiohttp==3.11.11 xarray==2024.11.0 bokeh==3.6.2 nltk==3.9.1
  seaborn==0.13.2 pillow==11.0.0 pydantic==2.10.4
)

rm -rf "$OUT" && mkdir -p "$OUT/site-packages"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "[corpus] CPython $CPYTHON_VERSION Lib/"
curl -fsSL "https://www.python.org/ftp/python/$CPYTHON_VERSION/Python-$CPYTHON_VERSION.tgz" -o "$tmp/cpython.tgz"
tar -xzf "$tmp/cpython.tgz" -C "$tmp"
cp -R "$tmp/Python-$CPYTHON_VERSION/Lib/." "$OUT/"
rm -rf "$OUT/site-packages/README.txt"

echo "[corpus] ${#PACKAGES[@]} pinned packages"
python3 -m pip install --quiet --no-deps --no-compile --only-binary=:all: \
  --target "$tmp/pkgs" "${PACKAGES[@]}"
# keep only importable package dirs / modules, not *.dist-info or bin/
for p in "$tmp/pkgs"/*; do
  case "$(basename "$p")" in *.dist-info|*.data|bin|__pycache__) continue ;; esac
  cp -R "$p" "$OUT/site-packages/"
done

{
  echo "cpython==$CPYTHON_VERSION"
  printf '%s\n' "${PACKAGES[@]}"
} > "$OUT/CORPUS_MANIFEST.txt"
echo "[corpus] done -> $OUT ($(find "$OUT" -name '*.py' | wc -l | tr -d ' ') .py files)"
