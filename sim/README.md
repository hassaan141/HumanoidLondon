# Sim (SO-100 Isaac Lab baseline)

Code and assets copied over from the earlier `sim2real` repo's SO-100 marker pick-place task
(`02_Sim2Real/marker_pick_place/`). This is the reusable simulation foundation for the phone-video
baseline — the robot asset, scene/camera rig, domain randomization, and LeRobot dataset recorder
all transfer directly. Left behind: the GR00T N1.6 / Cosmos Transfer / Brev training pipeline (too
heavy for current compute) and the RL/Newton/VR exploration pieces, which weren't needed here.

## Layout

```text
sim/
  assets/
    so100.usd          # converted SO-100 robot USD (joint-tuned, collision-baked)
    banana/             # placeholder graspable object — swap for the actual task object
  marker_pick_place.py   # Isaac Lab entry point: viewer / --teleop / --record
  marker_pick_place/
    assets/
      so100.py           # SO100_CFG ArticulationCfg: init pose, joint names, stiffness/damping
      scene_objects.py    # table/wall/floor/cup primitives as AssetBaseCfg/RigidObjectCfg
    tasks/
      marker_env_cfg.py   # MarkerSceneCfg, ActionsCfg, ObservationsCfg, camera sync + DR events
      marker_rl_env_cfg.py
    mdp/
      obs.py, resets.py, terms.py   # observation/reset/event helper functions
    teleop/
      dataset_recorder.py  # TeleopDatasetRecorder -> writes LeRobot-format episodes
      lerobot_interface.py # SO100LeaderTeleop: real leader-arm -> sim joint radians mapping
  tools/
    bake_so100_drive_gains.py     # bakes joint drive gains + convex-decomp collisions into USD
    export_recording_cameras.py   # dumps recorded camera frames from a teleop_dataset.pt
  docs/
    camera.md, runCommand.md      # original notes on camera setup and launch commands
```

## What's already solved here

- SO-100 joint names, limits, signs, and tuned PD gains (`marker_pick_place/assets/so100.py`,
  `tools/bake_so100_drive_gains.py`) — don't re-derive these from scratch.
- Gripper-mounted + external camera rig that tracks the end-effector frame
  (`sync_gripper_camera_to_frame`, `sync_env_camera_to_world` in `marker_env_cfg.py`).
- Domain randomization hooks for object pose, material color/roughness, and lighting.
- A LeRobot-format episode recorder (`dataset_recorder.py`) — the same format SmolVLA fine-tuning
  consumes, so sim rollouts saved here need no reformatting.

## What needs adapting for the phone-video baseline

- `lerobot_interface.py` assumes a physical SO-101 leader arm feeding joint targets. The phone-video
  pipeline has no leader arm — replace the `leader.read_sample()` call in `marker_pick_place.py`'s
  `_run_teleop` with a controller that follows the retargeted object/gripper trajectory extracted
  from the phone recordings.
- `scene_objects.py` / `MarkerSceneCfg` are built around a banana + cup task. Swap in the actual
  pick-and-place object and target for the chosen baseline task, keeping the table/wall/camera
  setup fixed.
- RL configs (`marker_rl_env_cfg.py`) and the `banana` asset are placeholders carried over for
  reference — replace or delete once the real task object is modeled.

## Running

```bash
PYTHONPATH=$PWD/sim \
<path-to-IsaacLab>/isaaclab.sh -p sim/marker_pick_place.py

# with teleop + dataset recording
PYTHONPATH=$PWD/sim \
<path-to-IsaacLab>/isaaclab.sh -p sim/marker_pick_place.py --teleop --record
```

Recording hotkeys: `R` start recording, `Y` stop and save, `T` reset environment (+ domain
randomization if `--domain-randomization` is passed).

Requires a local Isaac Lab installation (not included here — too large/environment-specific to
vendor into this repo).
