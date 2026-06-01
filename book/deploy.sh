#!/usr/bin/env bash
#
# Local deploy for the Decode Modeling mdBook.
#
# The book source (documentation/modeling/) is kept private and is no longer in
# the public repo, so GitHub Actions cannot build it. Instead we build the book
# locally — where the source files live — and publish only the rendered HTML to
# the gh-pages branch. GitHub Pages serves that branch.
#
#   Pages setting (one-time): Settings -> Pages -> Source = "Deploy from a
#   branch" -> Branch: gh-pages / (root).
#
# Usage:  ./book/deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="book/build"
BRANCH="gh-pages"

# --- preflight ---------------------------------------------------------------
command -v mdbook >/dev/null || { echo "error: mdbook not found"; exit 1; }
command -v mdbook-katex >/dev/null || {
  echo "error: mdbook-katex not found — math would render as raw LaTeX."
  echo "       install it first:  cargo install mdbook-katex"
  exit 1
}
[ -d documentation/modeling ] || {
  echo "error: documentation/modeling/ missing — that is the (local, private) book source."
  exit 1
}

# --- build -------------------------------------------------------------------
echo ">> building book"
rm -rf "$BUILD_DIR"
mdbook build book

# --- stage assets + rewrite relative paths (mirrors the retired CI step) -----
# Modeling docs reference repo-root assets via ../../assets/ so they render on
# github.com too. For the deployed site, copy assets next to the HTML and
# rewrite the relative paths to match the flattened output layout.
echo ">> staging assets"
cp -r assets "$BUILD_DIR/assets"
find "$BUILD_DIR" -maxdepth 1 -name '*.html' -exec perl -i -pe 's{\.\./\.\./assets/}{assets/}g' {} +
find "$BUILD_DIR" -mindepth 2 -name '*.html' -exec perl -i -pe 's{\.\./\.\./assets/}{../assets/}g' {} +

# Branch-served Pages runs Jekyll by default, which drops files it treats as
# special. mdBook output must be served verbatim.
touch "$BUILD_DIR/.nojekyll"

# --- publish to gh-pages via a throwaway worktree ----------------------------
echo ">> publishing to $BRANCH"
WORKTREE="$(mktemp -d)"
cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || true; }
trap cleanup EXIT

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git fetch origin "$BRANCH" --depth=1
  git worktree add -B "$BRANCH" "$WORKTREE" "origin/$BRANCH" >/dev/null
else
  # First deploy: no remote branch yet. Create a fresh orphan branch in the
  # worktree (portable across git versions; `worktree add --orphan` arg order
  # is inconsistent between releases).
  git worktree add --detach "$WORKTREE" HEAD >/dev/null
  git -C "$WORKTREE" checkout --orphan "$BRANCH" >/dev/null 2>&1
  git -C "$WORKTREE" rm -rf --quiet . >/dev/null 2>&1 || true
fi

# Replace the branch contents wholesale with the freshly built site.
rsync -a --delete --exclude='.git' --exclude='.DS_Store' "$BUILD_DIR"/ "$WORKTREE"/
find "$WORKTREE" -name '.DS_Store' -delete   # also purge any already on the branch

git -C "$WORKTREE" add -A
if git -C "$WORKTREE" diff --cached --quiet; then
  echo ">> no changes to publish"
else
  git -C "$WORKTREE" commit -q -m "Deploy book $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git -C "$WORKTREE" push origin "$BRANCH"
  echo ">> pushed to $BRANCH"
fi

echo ">> done"
