# UR5e: custom manipulation environment

A custom Gymnasium environment built from scratch around the Universal Robots
UR5e 6-DoF arm (MuJoCo physics), trained with the same from-scratch SAC agent as
the `fetch_reach` project.

The point of this project is the environment-design side of RL. In
`fetch_reach` the environment came prebuilt; here the observation space, action
space, reward, target sampling, and success criteria are all defined by hand.

## Status

In progress. So far the repo holds the vendored UR5e model under `assets/`,
taken unmodified from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
at a pinned commit (see [`assets/PROVENANCE.md`](assets/PROVENANCE.md);
regenerate with [`assets/fetch_model.sh`](assets/fetch_model.sh)).

Next: a scene file (arm, floor, target marker), then a `UR5eReach` environment,
then training and videos.

## The robot

6 revolute joints (base, shoulder, elbow, and three wrist joints), each driven
by a position-controlled actuator. A `home` keyframe sets the starting pose, and
the wrist flange carries a site we treat as the end-effector.
