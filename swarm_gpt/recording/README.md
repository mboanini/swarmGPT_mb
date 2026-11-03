# Experiment recording: data reference

Recording is automatic once enabled with a run name: `python swarm_gpt/launch.py --run <name>`
writes to `experiments/<name>/` (an existing run is resumed). Without `--run` nothing is installed
or recorded and the application behaves exactly as before. Every `meta.json` describes the shape, unit,
frame and time base of each array in its directory; this file summarizes them and explains how to
align the different sources offline (plots, 3D animations, video overlays).

## Conventions

| Item | Convention |
|---|---|
| Frame | World frame of `config/crazyflies.yaml` (`initial_position`) = Crazyflie `reference_frame: world` = Vicon world, x/y horizontal, z up |
| Units | m, m/s, m/s², rad/s, s, ns; choreography text and primitive arguments are in **cm** (only in `choreography/*/parse.json`) |
| Quaternions | `[x, y, z, w]` (crazyflow, ROS) |
| Drone order | Array index `i` = Choreographer index (Crazyflie IDs sorted ascending); `drones.json` and every `meta.json` map `i -> Crazyflie ID`; hardware rows carry the ID itself |
| Array layout | `(time, drone, ...)` |

## Time bases

| Name | Clock | Used by |
|---|---|---|
| `t_sim` / `t_cmd` | simulation time [s] from trajectory start (step counter / frequency) | AxSwarm, Crazyflow, visualizer, splines, waypoints |
| `t_wall_ns` | `time.time_ns()` (UTC, system clock) | every record, events, LLM calls |
| `t_mono_ns` | `time.perf_counter_ns()` | events, hardware commands; the hardware spline time is `t_mono` based |
| `t_ros_ns` | ROS clock of the crazyflie_py node = system clock (no `use_sim_time`) | hardware commands, flight-phase events, clock anchors |
| header stamps in the rosbag | ROS clock of the publishing node | `/cfX/pose` (stamped on **reception** by crazyflie_server), `/cfX/cmd_full_state`, `/poses` |
| bag receive time | system clock of the recorder | every bag message |

`deployment_*/meta.json → clock_anchors` holds back-to-back samples of (wall, monotonic, ROS) at
deploy start/end and at each flight phase (sampling latency ≈ 2 µs). Convert monotonic to ROS time
with the nearest anchor: `t_ros ≈ anchor.t_ros_ns + (t_mono_ns − anchor.t_mono_ns)`.

Simulation and hardware share **trajectory time** τ:
- simulation: τ = `t_sim`
- hardware: τ = `hardware_commands.npz:spline_t`, the exact time at which
  `run_spline_trajectories` evaluated the spline for that command (its own
  `perf_counter() − tstart`); `meta.json:spline_tstart_mono_ns` is that `tstart` on the `t_mono_ns`
  clock. `t_mono_ns` of a row is taken a few µs to ~10 ms (scheduling) after its `spline_t`.

## Signals per command (`commands/NNNN_<text>/`)

| Signal | File | Kind | Rate |
|---|---|---|---|
| Waypoints `time (n,T)`, `pos (n,T,3)` | `choreography/attempt_*/waypoints.npz` (+ `_before_clip`) | desired | irregular |
| AxSwarm solver input state `(S,n,6)` | `simulation_*/axswarm_solves.npz` | predicted (open loop) | 8 Hz |
| Full predicted horizon `pred_pos`, `pred_vel (S,n,51,3)`; sample k at `t_sim[s] + k/8` | same | predicted | 8 Hz |
| Input references `u_pos/u_vel/u_acc (S,n,50,3)`, `zeta`, `distance_matrix`, `success`, `iters`, `solve_time_s` | same | solver outputs | 8 Hz |
| Reference sent to Crazyflow (= `u_pos/u_vel[:, :, 1]`, held 1/8 s) | `axswarm_reference_8hz.npz`, `crazyflow_states_80hz.npz:cmd_state` | commanded | 8 Hz / 80 Hz |
| Crazyflow measured `pos, vel, quat, ang_vel` | `crazyflow_states_80hz.npz` (`t_sim` = state time, `t_cmd` = command time) | measured (sim) | 80 Hz |
| One-step prediction vs measured, `h`, min distances (former "Phase 1") | `prediction_and_safety.npz` | derived | 8 Hz |
| Smoothed splines (exact knots/coefficients, 80 Hz samples) | `smoothed_splines.npz` | commanded (visualizer + hardware) | continuous |
| Visualizer simulation (spline reference + measured `pos, vel, quat, ang_vel`) | `simulation_*/visualizer_*/visualizer_states_80hz.npz` | measured (sim) | 80 Hz sim time |
| Hardware commands (`cmd_pos, cmd_vel`, exact `spline_t`, wall/mono/ROS time, phase) + latest `/cfX/pose` and battery at each command | `deployment_*/hardware_commands.npz` | commanded + measured | controller rate (10 Hz) |
| All ROS topics (`/cfX/pose`, `/cfX/status`, `/cfX/cmd_full_state`, `/poses`, `/tf`, `/rosout`, …) | `deployment_*/rosbag/` (sqlite3; read with `rosbag2_py`) | measured | native rates |
| Flight phases (arm, takeoff, goTo, trajectory, landing), errors | `events.jsonl` | events | — |

## Not available from the current setup

| Missing | Reason | Offline substitute |
|---|---|---|
| Simulated accelerations | crazyflow `states_deriv` stays zero after `sim.step()` | finite differences of the 80 Hz `vel`; AxSwarm `u_acc` |
| Hardware velocity / acceleration | no velocity topic in the current crazyflie_server logging config | differentiate `/poses` (Vicon) or `/cfX/pose` |
| Onboard pose measurement time | crazyflie_server stamps `/cfX/pose` on reception (firmware time dropped) | Vicon `/poses`; treat `/cfX/pose` stamps as delayed by the radio latency |
| Setpoints during takeoff and goTo | executed onboard by the high-level commander, not streamed | goal and duration are in the phase events and `meta.json` (`initial_position`) |
| Pose at more than 10 Hz | `/cfX/pose` is configured at 10 Hz | `/poses` at the Vicon rate, **if** motion_capture_tracking runs in the same ROS domain |

The visualizer runs slower than real time on slow machines (3.2× in the WSL test): use its `t_sim`,
not `t_wall_ns`, for playback. It stops at the last spline knot (1/8 s before the AxSwarm simulation
ends).

## Requirements for overlays on laboratory video

Nothing about the camera is recorded by SwarmGPT. To project recorded 3D trajectories (hardware or
simulation) into video frames (`cv2.projectPoints`) you need:

1. **Intrinsics** for each camera, at the exact resolution, zoom and focus used: focal lengths,
   principal point, distortion coefficients (OpenCV checkerboard/ChArUco calibration). Disable
   autofocus, digital zoom and image stabilization.
2. **Extrinsics**: the camera pose in the Vicon world frame. Either ≥ 6 (better 10+) points with
   known world coordinates visible in the image (e.g. Vicon markers or a marker-equipped board,
   positions read from a short calibration bag) and `cv2.solvePnP`, or a Vicon-tracked object
   attached to the camera. Re-calibrate if the camera moves.
3. **Time synchronization** between video frames and ROS time:
   - best: a ROS camera node publishing images in the same ROS domain; `ros2 bag record -a` then
     stores frames with ROS stamps automatically (no SwarmGPT change);
   - otherwise: a constant-frame-rate video plus a visible event with a known ROS time, e.g. the
     takeoff (`flight_phase_start`, `takeoff` in `events.jsonl`), a clapperboard or a screen
     showing the system clock, refined by cross-correlating projected Vicon positions with
     tracked drone positions in the image (sub-frame offset); check frame timestamps with
     `ffprobe` (variable frame rate phone videos need per-frame timestamps).
4. **High-rate ground truth**: Vicon `/poses` in the bag (check that the rigid-body names in
   `config/motion_capture.yaml` match the flown Crazyflies: it lists `cf1..cf5`, the robots are
   `cf1, cf3, cf6, cf8`).
5. **Body geometry** for drawing the drone: offset between the tracked marker origin and the drone
   center, drone size.
6. **Frame check**: confirm that the choreography world frame (`initial_position` in
   `crazyflies.yaml`) is the Vicon world frame; simulation overlays use the same transform.
