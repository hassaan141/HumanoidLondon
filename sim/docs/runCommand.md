Run the marker pick-place environment:

```bash
PYTHONPATH=$PWD/02_Sim2Real/marker_pick_place \
/home/hassaan/robotics/IsaacLab/isaaclab.sh -p 02_Sim2Real/marker_pick_place/marker_pick_place.py
```

Run teleop:

```bash
PYTHONPATH=$PWD/02_Sim2Real/marker_pick_place \
/home/hassaan/robotics/IsaacLab/isaaclab.sh -p 02_Sim2Real/marker_pick_place/marker_pick_place.py --teleop
```

Run teleop with control debug logs:

```bash
PYTHONPATH=$PWD/02_Sim2Real/marker_pick_place \
/home/hassaan/robotics/IsaacLab/isaaclab.sh -p 02_Sim2Real/marker_pick_place/marker_pick_place.py --teleop --debug-control --debug-control-interval 60
```

Run teleop with cameras/dataset recording:

```bash
PYTHONPATH=$PWD/02_Sim2Real/marker_pick_place \
/home/hassaan/robotics/IsaacLab/isaaclab.sh -p 02_Sim2Real/marker_pick_place/marker_pick_place.py --teleop --record

  It now prints:

  R - start recording
  Y - stop and save recording
  T - reset env and randomize domain
  Close Isaac Sim - exit teleop

  Run with:

  python3 02_Sim2Real/marker_pick_place/
  marker_pick_place.py --teleop --record

  Syntax check passed.
```
