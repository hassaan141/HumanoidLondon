# HumanoidLondon — Baseline Plan

## Context

Initial baseline proposal (GR00T-based) was discarded as too computationally heavy for current hardware. Available compute:

- MacBook Air
- RTX 2070 Super (PC)
- RTX Quadro 4000 (laptop)

Revised plan targets SmolVLA instead of GR00T, with a phone-video-to-simulation data pipeline.

## Original Baseline Proposal

1. Pick one simple pick-and-place task supported by the existing simulator.
2. Record 5–10 phone videos performing the task, using a fixed camera.
3. Extract movements, annotate grasp/release, and retarget them to the simulated arm.
4. Verify the robot can complete the task using those trajectories.
5. Record successful simulated rollouts and fine-tune one VLA (originally GR00T).
6. Measure autonomous success on held-out object positions.

Trajectory execution is the first checkpoint; the trained VLA is the baseline. From there, test whether additional phone data, better retargeting, or residual RL improves performance.

## Revised First-Step Plan

1. **Recreate the tabletop workspace** in the existing simulator — match dimensions, object geometry, and camera pose. Consider MoGe-2 for geometry estimation from images, but verify scale against real measurements.
   - Gaussian splatting (e.g. MuGS with MuJoCo) is a possible later experiment for 1:1 environment cloning, but is **not** part of the baseline — splats reconstruct appearance, not collisions, friction, or movable-object physics. A hybrid (splats for background, physics meshes for interaction) would be additional integration work.
2. **Collect data from camera** — fixed phone camera, one task, measured tabletop reference points.
3. **Retarget recordings into simulation**:
   - Track object position, hand position, and manually marked grasp/release times.
   - Map movement into robot coordinates, accounting for reach and gripper geometry.
   - For pick-and-place, track the **object's** movement primarily — human finger motion doesn't map directly to a parallel gripper. Extract the object's start/goal position and transport path; engineer the robot's approach, grasp offset, lift, and release separately.
   - A single phone view can't reliably recover full metric 3D motion — use calibrated tabletop coordinates and a predefined lift height for the baseline, and disclose this assumption.
   - Execute the retargeted trajectory with a controller in sim; physics determines whether manipulation actually succeeds.
   - Save simulator camera images, robot state, and executed robot actions + instruction as training data.
4. **Fine-tune SmolVLA** (not GR00T, due to compute constraints).
   - On the RTX 2070 Super (8GB VRAM): start with batch size 1, a frozen vision encoder, and gradient accumulation. Test memory usage first — the documented batch-8 configuration exceeds 8GB VRAM.
5. **Evaluate autonomous performance** in sim on unseen/held-out object positions.

### Reviewer-facing experiment

Does a phone-reconstructed scene improve policy performance compared with a simple measured scene? This gives environment reconstruction a measurable purpose rather than being sophistication for its own sake.

**Recommendation:** SmolVLA + existing simulator + phone-derived demonstrations first. Add reconstruction sophistication (e.g. Gaussian splatting) only once that baseline loop works end to end.

## Data Collection Volume

- Start with **5 recordings** to validate the phone-video → sim-demonstration conversion pipeline.
- Then aim for **30–50 successful demonstrations** of one task.
  - Vary object positions, repeat each variation.
  - Reserve separate recordings for held-out testing (not used in training).
- More recordings will not fix incorrect action labels — volume doesn't substitute for correct retargeting.

Previous sim-only data collection/deployment attempt failed; the cause is unknown among camera mismatch, physics mismatch, action convention errors, or insufficient variation. Rollout evidence from this pipeline is needed to diagnose which.

## From Hand Video to VLA Training Data

A hand video cannot directly become VLA training data — it must be converted through simulation:

| Step | What we extract or generate |
|---|---|
| Record | Fixed phone view, visible hand/object, measured tabletop reference points |
| Track | Object position, hand position, manually marked grasp/release times |
| Retarget | Map movement into robot coordinates, accounting for its reach and gripper |
| Execute in sim | A controller follows the target trajectory while physics determines whether manipulation succeeds |
| Save robot data | Sim camera images + robot state + executed robot actions + instruction |
| Fine-tune | Train SmolVLA on those synchronized robot demonstrations |

**First milestone:** one phone recording produces a successful simulated grasp and placement. Inspect alignment and contact before collecting the full dataset — the recording determines the demonstration, but the controller is what makes it executable by the robot.
