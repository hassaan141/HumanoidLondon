  The issue looks like camera transform/render initialization, especially for
  the gripper camera.

  The official tutorial’s key advice is: attach the camera to the robot body, p
  referably through a stable intermediate prim, and avoid moving the live camera
  directly while viewing through it. They explicitly warn that moving a camera
  while displaying through it changes the camera properties, and recommend using
  a parent prim as the mount:
  https://docs.isaacsim.omniverse.nvidia.com/4.5.0/gui/tutorial_gui_camera_senso
rs.html

  What looks wrong in your setup:

  1. Your gripper camera is not actually parented under the gripper.

  In /home/hassaan/robotics/isaac_tuts/02_Sim2Real/marker_pick_place/
  marker_pick_place/tasks/marker_env_cfg.py:171, you create:

  gripper_cam.prim_path = "{ENV_REGEX_NS}/GripperCamera"

  Then you manually sync it to the gripper every interval in /home/hassaan/
  robotics/isaac_tuts/02_Sim2Real/marker_pick_place/marker_pick_place/tasks/
  marker_env_cfg.py:122.

  The official Isaac Lab pattern is more like:

  prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam"

  That is, put the camera under the robot link and let USD hierarchy handle the
  attachment. Your current approach creates an independent world-level camera
  and repeatedly writes world poses into it. That is much more likely to confuse
  viewport/render-product initialization.

  2. You are manually poking camera xforms after Isaac Lab already spawned them.

  refresh_config_cameras() in /home/hassaan/robotics/isaac_tuts/02_Sim2Real/
  marker_pick_place/marker_pick_place/tasks/marker_env_cfg.py:95 writes
  xformOp:translate / xformOp:orient directly. But Isaac Lab’s CameraCfg already
  applies offset during camera spawn. The fact that this workaround exists and
  the viewport fixes itself after you reparent/move the camera strongly suggests
  Hydra/USD is not seeing the camera transform cleanly at startup.

  3. The gripper camera is being updated continuously.

  This event:

  sync_gripper_camera = EventTerm(
      func=sync_gripper_camera_to_frame,
      mode="interval",
      interval_range_s=(0.0, 0.0),
  )

  at /home/hassaan/robotics/isaac_tuts/02_Sim2Real/marker_pick_place/
  marker_pick_place/tasks/marker_env_cfg.py:262 means the camera pose is being
  repeatedly driven by code. If the viewport is also looking through that same
  USD camera, this can produce the exact kind of “white until touched/rebound”
  behavior you’re seeing.

  Most likely fix

  Make the gripper camera a child of the gripper/link directly and remove the
  manual sync path:

  gripper_cam.prim_path = "{ENV_REGEX_NS}/Robot/gripper/GripperCamera"
  gripper_cam.offset.pos = GRIPPER_CAMERA_POS
  gripper_cam.offset.rot =
  euler_angles_to_quat(np.array(GRIPPER_CAMERA_ROT_DEG), degrees=True)

  Then remove/disable:

  - sync_gripper_camera_to_frame
  - sync_gripper_camera interval event
  - reset_sync_gripper_camera
  - the gripper part of _force_camera_refresh

  For env_cam, I’d also try removing the manual refresh_config_cameras() events
  and let CameraCfg author the transform once.

  So yes: this is probably an Isaac Sim render/viewport initialization issue
  triggered by how the camera transforms are being authored. The clean fix is to
  follow the official pattern: stable parent Xform/link, camera as a child,
  local offset, no repeated world-pose writes.
