"""
swarmgpt_global_apis.py — implementazione reale per Crazyflie 2.1
Scope: eseguita dal control center; conosce lo stato di tutto lo sciame.

Unità interne: CENTIMETRI (cm), come da robot_api_prompt.py.
pycrazyswarm usa METRI → tutte le chiamate al drone dividono per 100.
Coordinate: X,Y in [-200,200] cm | Z in [20,200] cm

Inizializzazione obbligatoria prima di usare qualsiasi API:
f    from genswarm_overrides.swarmgpt_global_apis import init
    init(allcfs, time_helper, initial_positions_m, settings, task_config={...})

  - allcfs               : pycrazyswarm CrazyflieServer  (swarm.allcfs)
  - time_helper          : pycrazyswarm TimeHelper       (swarm.timeHelper)
  - initial_positions_m  : {drone_id: np.array([x,y,z])} in METRI
  - settings             : dizionario da settings.yaml (axswarm.pos_min/pos_max)
  - task_config (opz.)   : dizionario con parametri specifici del task:
      {
        "prey_id"         : int,              # drone che fa da preda (encircling)
        "lead_id"         : int,              # drone leader (pursuing)
        "formation_points": [[x,y,z], ...],  # target in cm per ogni drone (shaping)
        "explore_step_cm" : float,            # passo griglia esplorazione (default 60)
      }
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ─── stato del modulo ────────────────────────────────────────────────────────

_allcfs: Any = None
_time_helper: Any = None
_initial_positions_cm: dict[int, np.ndarray] = {}
_env_bounds: dict = {}

# parametri task
_prey_id: int | None = None
_lead_id: int | None = None
_formation_points_cm: list[np.ndarray] = []

# griglia esplorazione: set di tuple (x, y, z) → centro di ogni cella
_all_area_positions: list[np.ndarray] = []   # tutte le celle (ordinate)
_visited_area_ids: set[int] = set()          # indici celle già visitate

# velocità di crociera per calcolare la durata dei movimenti
_CRUISE_SPEED_CM_S = 40.0
_MIN_MOVE_DURATION = 1.5
_EXPLORE_STEP_CM = 60.0
# raggio entro cui una cella si considera "visitata" (cm)
_VISIT_RADIUS_CM = 40.0


def init(
    allcfs,
    time_helper,
    initial_positions_m: dict,
    settings: dict,
    task_config: dict | None = None,
) -> None:
    """
    Inizializza il modulo con l'oggetto pycrazyswarm già creato.

    task_config keys (tutti opzionali):
      prey_id         – ID del drone che fa da preda nell'encircling
      lead_id         – ID del drone leader nel pursuing
      formation_points – lista di [x,y,z] in cm per il task shaping
      explore_step_cm  – passo della griglia 3D per il task exploration
    """
    global _allcfs, _time_helper, _initial_positions_cm, _env_bounds
    global _prey_id, _lead_id, _formation_points_cm
    global _all_area_positions, _visited_area_ids, _EXPLORE_STEP_CM

    _allcfs = allcfs
    _time_helper = time_helper
    _initial_positions_cm = {
        int(k): np.asarray(v, dtype=float) * 100.0
        for k, v in initial_positions_m.items()
    }
    lim = settings["axswarm"]
    _env_bounds = {
        "x_min": float(lim["pos_min"][0]) * 100,
        "x_max": float(lim["pos_max"][0]) * 100,
        "y_min": float(lim["pos_min"][1]) * 100,
        "y_max": float(lim["pos_max"][1]) * 100,
        "z_min": float(lim["pos_min"][2]) * 100,
        "z_max": float(lim["pos_max"][2]) * 100,
    }

    cfg = task_config or {}
    _prey_id = cfg.get("prey_id", None)
    _lead_id = cfg.get("lead_id", None)
    _formation_points_cm = [
        np.asarray(p, dtype=float) for p in cfg.get("formation_points", [])
    ]
    _EXPLORE_STEP_CM = float(cfg.get("explore_step_cm", 60.0))

    # Costruisci la griglia 3D per l'exploration
    _all_area_positions = _build_explore_grid()
    _visited_area_ids = set()


# ─── helper interni ──────────────────────────────────────────────────────────

def _cf(drone_id: int):
    return _allcfs.crazyfliesById[drone_id]


def _pos_cm(drone_id: int) -> np.ndarray:
    """Posizione attuale di un drone in cm (legge da mocap via tf)."""
    return _cf(drone_id).position() * 100.0


def _all_ids() -> list[int]:
    return sorted(_allcfs.crazyfliesById.keys())


def _duration(dist_cm: float) -> float:
    return max(_MIN_MOVE_DURATION, dist_cm / _CRUISE_SPEED_CM_S)


def _goto_cm(drone_id: int, target_cm: np.ndarray) -> float:
    dist = float(np.linalg.norm(target_cm - _pos_cm(drone_id)))
    dur = _duration(dist)
    _cf(drone_id).goTo(target_cm / 100.0, yaw=0.0, duration=dur)
    return dur


def _clamp_cm(pos: np.ndarray) -> np.ndarray:
    lo = np.array([_env_bounds["x_min"], _env_bounds["y_min"], _env_bounds["z_min"]])
    hi = np.array([_env_bounds["x_max"], _env_bounds["y_max"], _env_bounds["z_max"]])
    return np.clip(pos, lo, hi)


def _move_all_and_wait(targets_cm: dict[int, np.ndarray]) -> None:
    max_dur = 0.0
    for did, tgt in targets_cm.items():
        dur = _goto_cm(did, tgt)
        max_dur = max(max_dur, dur)
    _time_helper.sleep(max_dur)


def _build_explore_grid() -> list[np.ndarray]:
    """Costruisce una griglia 3D di centri cella per il task exploration."""
    if not _env_bounds:
        return []
    step = _EXPLORE_STEP_CM
    xs = np.arange(_env_bounds["x_min"] + step / 2, _env_bounds["x_max"], step)
    ys = np.arange(_env_bounds["y_min"] + step / 2, _env_bounds["y_max"], step)
    zs = np.arange(_env_bounds["z_min"] + step / 2, _env_bounds["z_max"], step)
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append(np.array([x, y, z]))
    return grid


def _update_visited(drone_id: int) -> None:
    """Segna come visitate le celle vicine alla posizione attuale del drone."""
    if not _all_area_positions:
        return
    pos = _pos_cm(drone_id)
    for i, cell in enumerate(_all_area_positions):
        if i not in _visited_area_ids and np.linalg.norm(cell - pos) <= _VISIT_RADIUS_CM:
            _visited_area_ids.add(i)


# ─── GenSwarm base APIs — scope: global ──────────────────────────────────────

def get_all_drones_id() -> list[int]:
    """
    Description: Get the IDs of all drones in the swarm.
    Returns:
    - list[int]: List of all drone IDs (1-indexed).
    """
    if _allcfs is None:
        return []
    return _all_ids()


def get_all_drones_initial_position() -> dict[int, np.ndarray]:
    """
    Description: Get the initial 3D positions of all drones.
    Returns:
    - dict[int, numpy.ndarray]: drone_id -> [x, y, z] in cm.
    """
    return {k: v.copy() for k, v in _initial_positions_cm.items()}


def get_environment_range() -> dict:
    """
    Description: Get the 3D bounds of the flying volume.
    Returns:
    - dict: {x_min, x_max, y_min, y_max, z_min, z_max} in cm.
    """
    if _env_bounds:
        return dict(_env_bounds)
    return {"x_min": -200, "x_max": 200, "y_min": -200, "y_max": 200, "z_min": 20, "z_max": 200}


# ─── GenSwarm task APIs — scope: global ──────────────────────────────────────

def get_target_formation_points() -> list[np.ndarray]:
    """
    Description: Get the 3D target points for the current formation (task: shaping).
    One target per drone, in cm. Set via task_config['formation_points'] in init().
    Returns:
    - list[numpy.ndarray]: one [x, y, z] per drone in cm.
    """
    return [p.copy() for p in _formation_points_cm]


def get_initial_unexplored_areas() -> list[np.ndarray]:
    """
    Description: Get all initially unexplored areas (task: exploration).
    Restituisce i centri di tutte le celle della griglia 3D non ancora visitate.
    Returns:
    - list[numpy.ndarray]: each [x, y, z] in cm.
    """
    return [
        _all_area_positions[i].copy()
        for i in range(len(_all_area_positions))
        if i not in _visited_area_ids
    ]


def get_prey_initial_position() -> np.ndarray:
    """
    Description: Get the initial 3D position of the prey drone (task: encircling).
    Legge la posizione iniziale dal config. Il drone preda è identificato da
    task_config['prey_id'] passato a init().
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    """
    if _prey_id is not None and _prey_id in _initial_positions_cm:
        return _initial_positions_cm[_prey_id].copy()
    return np.zeros(3)


def get_quadrant_target_position() -> dict[int, np.ndarray]:
    """
    Description: Get the 3D target for each drone in its assigned quadrant (task: clustering).
    Divide il piano XY in una griglia sqrt(N) × sqrt(N) e assegna un quadrante per drone.
    Returns:
    - dict[int, numpy.ndarray]: drone_id -> centro del quadrante [x, y, z] in cm.
    """
    if _allcfs is None:
        return {}
    ids = _all_ids()
    n = len(ids)
    if n == 0:
        return {}
    bounds = get_environment_range()
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    x_step = (bounds["x_max"] - bounds["x_min"]) / cols
    y_step = (bounds["y_max"] - bounds["y_min"]) / rows
    z_mid = (bounds["z_min"] + bounds["z_max"]) / 2.0
    result = {}
    for idx, did in enumerate(ids):
        col = idx % cols
        row = idx // cols
        cx = bounds["x_min"] + x_step * (col + 0.5)
        cy = bounds["y_min"] + y_step * (row + 0.5)
        result[did] = np.array([cx, cy, z_mid])
    return result


# ─── motion primitives che aggiornano le celle visitate (exploration) ─────────

def mark_visited(drone_id: int) -> None:
    """
    Aggiorna le celle esplorate in base alla posizione attuale del drone.
    Chiamare dopo ogni spostamento durante un task di exploration.
    """
    if _allcfs is not None:
        _update_visited(drone_id)


# ─── Motion primitives — scope: global ───────────────────────────────────────

def move(x: float, y: float, z: float, drone_id: int) -> None:
    """
    Description: Move a drone to an ABSOLUTE 3D position and wait until reached.
    Input:
    - x, y, z (float): Target in cm. X,Y in [-200,200], Z in [20,200].
    - drone_id (int): The drone to move.
    """
    if _allcfs is None:
        return
    target = _clamp_cm(np.array([float(x), float(y), float(z)]))
    dur = _goto_cm(drone_id, target)
    _time_helper.sleep(dur)
    _update_visited(drone_id)


def move_z(drone_ids: list[int], distance: float) -> None:
    """
    Description: Move drones up or down by a relative distance and wait.
    Input:
    - drone_ids (list[int]): IDs dei droni. Lista vuota = tutti.
    - distance (float): Spostamento relativo in cm (positivo=su, negativo=giù).
    """
    if _allcfs is None:
        return
    ids = drone_ids if drone_ids else _all_ids()
    targets = {}
    for did in ids:
        p = _pos_cm(did).copy()
        p[2] = float(np.clip(
            p[2] + distance,
            _env_bounds.get("z_min", 20),
            _env_bounds.get("z_max", 200),
        ))
        targets[did] = p
    _move_all_and_wait(targets)
    for did in ids:
        _update_visited(did)


def rotate(angle: float, axis: str) -> None:
    """
    Description: Rotate the entire swarm formation around the swarm centroid.
    Input:
    - angle (float): Degrees (positivo = antiorario).
    - axis (str): "x", "y", or "z".
    """
    if _allcfs is None:
        return
    ids = _all_ids()
    positions = {did: _pos_cm(did) for did in ids}
    centroid = np.mean(list(positions.values()), axis=0)
    theta = np.radians(float(angle))
    c, s = np.cos(theta), np.sin(theta)
    rot = {
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
    }.get(axis.lower())
    if rot is None:
        return
    targets = {
        did: _clamp_cm(centroid + rot @ (positions[did] - centroid))
        for did in ids
    }
    _move_all_and_wait(targets)


def form_circle(drone_ids: list[int], z_coord: float) -> None:
    """
    Description: Arrange drones in a horizontal circle. Radius auto-computed for >= 80 cm spacing.
    Input:
    - drone_ids (list[int]): IDs dei droni. Lista vuota = tutti.
    - z_coord (float): Quota in cm, in [20, 200].
    """
    if _allcfs is None:
        return
    ids = drone_ids if drone_ids else _all_ids()
    n = len(ids)
    radius = max(80.0 * n / (2 * np.pi), 80.0)
    positions = [_pos_cm(did) for did in ids]
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    z = float(np.clip(z_coord, _env_bounds.get("z_min", 20), _env_bounds.get("z_max", 200)))
    targets = {}
    for i, did in enumerate(ids):
        a = 2 * np.pi * i / n
        targets[did] = _clamp_cm(np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z]))
    _move_all_and_wait(targets)


def center(drone_ids: list[int]) -> None:
    """
    Description: Regroup drones into a tight cluster at the swarm centroid.
    Input:
    - drone_ids (list[int]): IDs dei droni da raggruppare.
    """
    if _allcfs is None:
        return
    ids = drone_ids if drone_ids else _all_ids()
    positions = [_pos_cm(did) for did in ids]
    centroid = _clamp_cm(np.mean(positions, axis=0))
    _move_all_and_wait({did: centroid.copy() for did in ids})


def swap(drone_id_1: int, drone_id_2: int) -> None:
    """
    Description: Swap positions of two drones.
    Input:
    - drone_id_1, drone_id_2 (int): I due droni.
    """
    if _allcfs is None:
        return
    pos1 = _pos_cm(drone_id_1).copy()
    pos2 = _pos_cm(drone_id_2).copy()
    z_safe = min(_env_bounds.get("z_max", 200), max(pos1[2], pos2[2]) + 30)
    intermediate_1 = pos1.copy()
    intermediate_1[2] = z_safe
    dur1 = _goto_cm(drone_id_1, _clamp_cm(intermediate_1))
    _time_helper.sleep(dur1)
    _move_all_and_wait({drone_id_1: pos2, drone_id_2: pos1})


def form_star(height: float, min_spacing: float, delta_radius: float) -> None:
    """
    Description: Arrange drones in a star pattern (inner + outer ring).
    Input:
    - height (float): Quota Z in cm.
    - min_spacing (float): Distanza minima tra droni anello interno in cm (>= 40).
    - delta_radius (float): Differenza di raggio tra anello interno e esterno in cm (>= 40).
    """
    if _allcfs is None:
        return
    ids = _all_ids()
    n = len(ids)
    n_inner = n // 2
    n_outer = n - n_inner
    r_inner = max(float(min_spacing) * n_inner / (2 * np.pi), float(min_spacing))
    r_outer = r_inner + float(delta_radius)
    positions = [_pos_cm(did) for did in ids]
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    z = float(np.clip(height, _env_bounds.get("z_min", 20), _env_bounds.get("z_max", 200)))
    targets = {}
    for i, did in enumerate(ids):
        if i < n_inner:
            r, a = r_inner, 2 * np.pi * i / max(n_inner, 1)
        else:
            j = i - n_inner
            r, a = r_outer, 2 * np.pi * j / max(n_outer, 1) + np.pi / max(n_outer, 1)
        targets[did] = _clamp_cm(np.array([cx + r * np.cos(a), cy + r * np.sin(a), z]))
    _move_all_and_wait(targets)


def form_cone(delta_height: float, spacing: float, is_inverted: bool) -> None:
    """
    Description: Arrange drones in a 3D cone (anelli a livelli di altezza crescente).
    Input:
    - delta_height (float): Distanza verticale in cm tra livelli.
    - spacing (float): Distanza orizzontale in cm tra droni per livello.
    - is_inverted (bool): True=aperto verso l'alto, False=aperto verso il basso.
    """
    if _allcfs is None:
        return
    ids = _all_ids()
    n = len(ids)
    positions = [_pos_cm(did) for did in ids]
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    base_z = float(np.mean([p[2] for p in positions]))
    targets = {}
    assigned, layer = 0, 0
    while assigned < n:
        n_in_layer = 1 if layer == 0 else min(layer * 4, n - assigned)
        radius = layer * float(spacing)
        z = float(np.clip(
            base_z + layer * float(delta_height) * (1 if is_inverted else -1),
            _env_bounds.get("z_min", 20), _env_bounds.get("z_max", 200),
        ))
        for j in range(n_in_layer):
            if assigned >= n:
                break
            did = ids[assigned]
            if n_in_layer == 1:
                pos = np.array([cx, cy, z])
            else:
                a = 2 * np.pi * j / n_in_layer
                pos = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z])
            targets[did] = _clamp_cm(pos)
            assigned += 1
        layer += 1
    _move_all_and_wait(targets)


def polygon(n_sides: int, height: float) -> None:
    """
    Description: Arrange drones in a regular polygon. Radius auto-computed for >= 60 cm spacing.
    Input:
    - n_sides (int): es. 3=triangolo, 4=quadrato, 6=esagono.
    - height (float): Quota Z in cm.
    """
    if _allcfs is None:
        return
    ids = _all_ids()
    dids = ids[:n_sides]
    n = len(dids)
    radius = max(60.0 * n / (2 * np.pi), 60.0)
    positions = [_pos_cm(did) for did in ids]
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    z = float(np.clip(height, _env_bounds.get("z_min", 20), _env_bounds.get("z_max", 200)))
    targets = {}
    for i, did in enumerate(dids):
        a = 2 * np.pi * i / n
        targets[did] = _clamp_cm(np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z]))
    _move_all_and_wait(targets)
