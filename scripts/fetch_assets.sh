#!/bin/bash
# Fetch the UR5e model from mujoco_menagerie at pinned commit

set -euo pipefail

MENAGERIE_COMMIT=8161bba264d7fa7c99ca301e91e7fb44737676ad
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DEST="$ROOT/third_party/mujoco_menagerie"

if [ -d "$DEST/universal_robots_ur5e" ]; then
    echo "Already present: $DEST/universal_robots_ur5e"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$TMP" sparse-checkout set universal_robots_ur5e
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$MENAGERIE_COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD

mkdir -p "$DEST"
cp -R "$TMP/universal_robots_ur5e" "$DEST/"
echo "$MENAGERIE_COMMIT" > "$DEST/COMMIT"
echo "Fetched UR5e @ $MENAGERIE_COMMIT → $DEST/universal_robots_ur5e"
