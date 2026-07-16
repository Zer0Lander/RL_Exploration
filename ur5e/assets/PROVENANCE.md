# Vendored asset provenance

## `universal_robots_ur5e/`

| | |
| --- | --- |
| **Source** | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |
| **Path in source** | `universal_robots_ur5e/` |
| **Pinned commit** | `71f066ad0be9cd271f7ed58c030243ef157af9f4` (2026-07-04) |
| **License** | BSD-3-Clause (see `universal_robots_ur5e/LICENSE`) |
| **Files kept** | Model only: `ur5e.xml`, `scene.xml`, `assets/` meshes, `LICENSE`. Upstream's README, CHANGELOG, and preview image are dropped (unused). |
| **Modifications** | None to the model files themselves. |

The model is vendored (committed into this repo) so the project is
self-contained, with no submodule or download step needed to run the code. To
regenerate the same files from upstream, run [`fetch_model.sh`](fetch_model.sh).

Attribution: UR5e model by the ROS-Industrial Consortium, adapted for MuJoCo by
the Menagerie authors. The BSD-3-Clause license permits redistribution with
attribution; the original `LICENSE` is retained alongside the model files.
