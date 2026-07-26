#!/usr/bin/env bash
# Reproducibly fetch the UR5e model from MuJoCo Menagerie.
#
# The model is vendored (committed) into this repo, so you do NOT need to run
# this to use the project. It exists to document exactly where the model came
# from and to let anyone regenerate the same files.
#
# We keep the functional model (ur5e.xml, scene.xml, meshes) and its LICENSE,
# and drop upstream's own README/CHANGELOG/preview image, which the project
# does not use. Our own files in the model folder (e.g. reach_scene.xml) are
# left untouched: this copies upstream files over the top rather than wiping
# the folder first.
#
# Usage:  ./fetch_model.sh
set -euo pipefail

REPO="https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT="71f066ad0be9cd271f7ed58c030243ef157af9f4"   # pinned for reproducibility
MODEL="universal_robots_ur5e"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Blobless + sparse checkout: download history metadata only, then just the one
# model folder, not the whole multi-hundred-MB Menagerie repo.
git clone --filter=blob:none --no-checkout -q "$REPO" "$TMP"
git -C "$TMP" sparse-checkout init --cone
git -C "$TMP" sparse-checkout set "$MODEL"
git -C "$TMP" checkout -q "$COMMIT"

mkdir -p "${HERE:?}/$MODEL"
cp -r "$TMP/$MODEL/." "$HERE/$MODEL/"
rm -f "$HERE/$MODEL/README.md" "$HERE/$MODEL/CHANGELOG.md" "$HERE/$MODEL/ur5e.png"
echo "Fetched $MODEL @ ${COMMIT:0:7} into $HERE/$MODEL"
