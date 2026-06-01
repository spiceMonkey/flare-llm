#!/usr/bin/env bash
# Render every *.mmd in this directory to a PNG one level up (assets/*.png).
# Requires npx; the Mermaid CLI is fetched on first run.
#
# Usage:  ./assets/mermaid/render.sh           # render all
#         ./assets/mermaid/render.sh foo.mmd   # render one file
set -euo pipefail

cd "$(dirname "$0")"
OUT_DIR=".."
MMDC="npx --no -y -p @mermaid-js/mermaid-cli@latest mmdc"

files=("${@:-}")
if [[ -z "${files[*]:-}" ]]; then
  files=(*.mmd)
fi

for src in "${files[@]}"; do
  name="${src%.mmd}"
  out="${OUT_DIR}/${name}.png"
  echo "→ ${src}  →  ${out}"
  $MMDC -i "$src" -o "$out" -b white --scale 2
done

echo "Done."
