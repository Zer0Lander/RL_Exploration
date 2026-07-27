#!/usr/bin/env bash
# Fetch the Unitree Go2 model from MuJoCo Menagerie, pinned, using curl + the
# GitHub API (no git required).
#
# The model is vendored (committed) into this repo, so you do not need to run
# this to use the project. It documents where the model came from and lets
# anyone regenerate the same files. We keep the model + LICENSE and drop
# upstream's README/CHANGELOG, which the project does not use.
#
# Usage:  ./fetch_model.sh
set -euo pipefail

REPO="google-deepmind/mujoco_menagerie"
COMMIT="71f066ad0be9cd271f7ed58c030243ef157af9f4"   # pinned for reproducibility
MODEL="unitree_go2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="https://api.github.com/repos/$REPO/contents"

# Download every file in one repo directory into a local destination.
fetch_files() {   # $1 = repo subpath, $2 = local dest
    mkdir -p "$2"
    curl -sf "$API/$1?ref=$COMMIT" \
        | jq -r '.[] | select(.type=="file") | .download_url' \
        | while read -r url; do
              curl -sfL "$url" -o "$2/$(basename "$url")"
          done
}

rm -rf "${HERE:?}/$MODEL"
fetch_files "$MODEL" "$HERE/$MODEL"
fetch_files "$MODEL/assets" "$HERE/$MODEL/assets"
# Keep the plain model (go2.xml, scene.xml, meshes) + LICENSE. Drop upstream
# docs, preview images, and the MJX variants, which this project does not use.
rm -f "$HERE/$MODEL/README.md" "$HERE/$MODEL/CHANGELOG.md" \
      "$HERE/$MODEL/go2.png" "$HERE/$MODEL/go2_mjx.png" \
      "$HERE/$MODEL/go2_mjx.xml" "$HERE/$MODEL/scene_mjx.xml"
echo "Fetched $MODEL @ ${COMMIT:0:7} into $HERE/$MODEL"
