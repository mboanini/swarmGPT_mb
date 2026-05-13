"""
swarmgpt_local_apis.py — implementazione reale per Crazyflie 2.1
Scope: eseguita su ogni singolo drone; conosce solo lo stato del drone stesso
       e può osservare i vicini tramite mocap.

Unità interne: CENTIMETRI (cm), come da robot_api_prompt.py.
pycrazyswarm usa METRI → tutte le chiamate al drone dividono per 100.
Coordinate: X,Y in [-200,200] cm | Z in [20,200] cm

Inizializzazione obbligatoria prima di usare qualsiasi API:
    from genswarm_overrides.swarmgpt_local_apis import init
    init(cf, self_id, allcfs, initial_positions_m, settings, task_config={...})

  - cf                   : oggetto Crazyflie di pycrazyswarm per QUESTO drone
  - self_id              : ID intero di questo drone (1-indexed)
  - allcfs               : CrazyflieServer (per leggere posizioni degli altri)
  - initial_positions_m  : {drone_id: np.array([x,y,z])} in METRI
  - settings             : dizionario da settings.yaml (axswarm.pos_min/pos_max)
  - task_config (opz.)   : dizionario con parametri specifici del task:
      {
        "prey_id"         : int,    # drone che fa da preda (encircling)
        "lead_id"         : int,    # drone leader (pursuing)
        "assigned_task"   : Any,    # risultato dell'allocatore globale (shaping, clustering)
        "explore_step_cm" : float,  # passo griglia esplorazione (default 60)
      }
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ─── stato del modulo ────────────────────────────────────────────────────────

_cf: Any = None
_self_id: int = -1
_allcfs: Any = None
_initial_positions_cm: dict[int, np.ndarray] = {}
_env_bounds: dict = {}

_prey_id: int | None = None
_lead_id: int | None = None
_assigned_task: Any = None

# Stato per set_self_velocity (approx. velocità corrente)
_last_cmd_velocity: np.ndarray = np.zeros(3)

# Raggio di percezione per get_surrounding_environment_info (cm)
_PERCEPTION_RADIUS_CM = 150.0
_SELF_RADIUS_CM = 20.0

# Griglia esplorazione — riferimento condiviso con il modulo globale
_explore_step_cm: float = 60.0
_visit_radius_cm: float = 40.0


def init(
    cf,
    self_id: int,
    allcfs,
    initial_positions_m: dict,
    settings: dict,
    task_config: dict | None = None,
) -> None:
    """
    Inizializza il modulo con l'oggetto Crazyflie già creato.

    task_config keys (tutti opzionali):
      prey_id        - ID del drone che fa da preda nell'encircling
      lead_id        - ID del drone leader nel pursuing
      assigned_task  - task assegnato dall'allocatore (dizionario, stringa, ecc.)
      explore_step_cm - passo della griglia 3D per l'exploration
    """
    global _cf, _self_id, _allcfs, _initial_positions_cm, _env_bounds
    global _prey_id, _lead_id, _assigned_task
    global _last_cmd_velocity, _explore_step_cm

    _cf = cf
    _self_id = int(self_id)
    _allcfs = allcfs
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
    _assigned_task = cfg.get("assigned_task", None)
    _explore_step_cm = float(cfg.get("explore_step_cm", 60.0))
    _last_cmd_velocity = np.zeros(3)


# ─── helper interni ──────────────────────────────────────────────────────────

def _clamp_cm(pos: np.ndarray) -> np.ndarray:
    lo = np.array([_env_bounds["x_min"], _env_bounds["y_min"], _env_bounds["z_min"]])
    hi = np.array([_env_bounds["x_max"], _env_bounds["y_max"], _env_bounds["z_max"]])
    return np.clip(pos, lo, hi)


def _other_cf(drone_id: int):
    return _allcfs.crazyfliesById[drone_id]


# ─── GenSwarm local APIs ──────────────────────────────────────────────────────

def get_self_id() -> int:
    """
    Description: Get the unique ID of this drone. IDs start from 1.
    Returns:
    - int: The unique ID of this drone.
    """
    return _self_id


def get_self_position() -> np.ndarray:
    """
    Description: Get the current 3D position of this drone in real-time.
    Returns:
    - numpy.ndarray: [x, y, z] in centimetres.
    """
    if _cf is None:
        return np.zeros(3)
    return _cf.position() * 100.0


def get_self_velocity() -> np.ndarray:
    """
    Description: Get the current 3D velocity of this drone.
    Restituisce l'ultimo valore di velocità inviato tramite set_self_velocity.
    Returns:
    - numpy.ndarray: [vx, vy, vz] in cm/s.
    """
    return _last_cmd_velocity.copy()


def set_self_velocity(velocity) -> None:
    """
    Description: Set the 3D velocity of this drone immediately (streaming command).
    Input:
    - velocity (numpy.ndarray): [vx, vy, vz] in cm/s.
    Note: deve essere chiamata ripetutamente (~10 Hz) nel loop di controllo.
          Internamente chiama cmdVelocityWorld che è un comando streaming.
    """
    global _last_cmd_velocity
    if _cf is None:
        return
    vel_cm = np.asarray(velocity, dtype=float).flatten()[:3]
    _last_cmd_velocity = vel_cm.copy()
    _cf.cmdVelocityWorld(vel_cm / 100.0, yawRate=0.0)


def stop_self() -> None:
    """
    Description: Stop this drone immediately (hover in place).
    Transisce dalla modalità streaming alla modalità high-level e hovera
    sulla posizione corrente.
    """
    global _last_cmd_velocity
    if _cf is None:
        return
    _last_cmd_velocity = np.zeros(3)
    _cf.notifySetpointsStop(remainValidMillisecs=100)
    _cf.goTo(_cf.position(), yaw=0.0, duration=0.5)


def get_self_radius() -> float:
    """
    Description: Get the safety radius of this drone.
    Returns:
    - float: Radius in centimetres (20 cm).
    """
    return _SELF_RADIUS_CM


def get_environment_range() -> dict:
    """
    Description: Get the 3D bounds of the flying volume.
    Returns:
    - dict: {x_min, x_max, y_min, y_max, z_min, z_max} in cm.
    """
    if _env_bounds:
        return dict(_env_bounds)
    return {"x_min": -200, "x_max": 200, "y_min": -200, "y_max": 200, "z_min": 20, "z_max": 200}


def get_surrounding_environment_info() -> list[dict]:
    """
    Description: Get real-time info of surrounding drones within perception range.
    Legge le posizioni degli altri droni via mocap (TF) e filtra per raggio.
    Returns:
    - list[dict]: each with keys: type, id, position ([x,y,z] cm),
                  velocity ([vx,vy,vz] cm/s — approssimazione zero), radius.
    """
    if _cf is None or _allcfs is None:
        return []
    my_pos = get_self_position()
    result = []
    for did, other_cf in _allcfs.crazyfliesById.items():
        if int(did) == _self_id:
            continue
        try:
            other_pos_cm = other_cf.position() * 100.0
        except Exception:
            continue
        if float(np.linalg.norm(other_pos_cm - my_pos)) <= _PERCEPTION_RADIUS_CM:
            result.append({
                "type":     "drone",
                "id":       int(did),
                "position": other_pos_cm,
                "velocity": np.zeros(3),
                "radius":   _SELF_RADIUS_CM,
            })
    return result


# ─── Task APIs — scope: local ─────────────────────────────────────────────────

def get_target_position() -> np.ndarray:
    """
    Description: Get the 3D target position assigned to this drone (task: shaping, bridging, crossing).
    Legge il target dall'assigned_task (campo 'target') oppure dalla lista
    formation_points indicizzata per self_id.
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    """
    if isinstance(_assigned_task, dict):
        if "target" in _assigned_task:
            return np.asarray(_assigned_task["target"], dtype=float)
        if "position" in _assigned_task:
            return np.asarray(_assigned_task["position"], dtype=float)
    if isinstance(_assigned_task, (list, np.ndarray)):
        return np.asarray(_assigned_task, dtype=float)
    return np.zeros(3)


def get_prey_position() -> np.ndarray:
    """
    Description: Get the real-time 3D position of the prey drone (task: encircling).
    Legge la posizione live dal mocap del drone identificato come preda
    (task_config['prey_id'] passato a init()).
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    """
    if _prey_id is None or _allcfs is None:
        return np.zeros(3)
    try:
        return _other_cf(_prey_id).position() * 100.0
    except Exception:
        return np.zeros(3)


def get_lead_position() -> np.ndarray:
    """
    Description: Get the real-time 3D position of the lead drone (task: pursuing).
    Legge la posizione live dal mocap del drone leader
    (task_config['lead_id'] passato a init()).
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    """
    if _lead_id is None or _allcfs is None:
        return np.zeros(3)
    try:
        return _other_cf(_lead_id).position() * 100.0
    except Exception:
        return np.zeros(3)


def get_surrounding_unexplored_area() -> list[dict]:
    """
    Description: Get unexplored 3D areas within perception range (task: exploration).
    Usa la griglia globale di celle non visitate condivisa con swarmgpt_global_apis.
    Returns:
    - list[dict]: each with keys: id (int), position ([x,y,z] cm).
    """
    try:
        import genswarm_overrides.swarmgpt_global_apis as _gapi
        all_areas = _gapi._all_area_positions
        visited = _gapi._visited_area_ids
    except Exception:
        return []

    if not all_areas or _cf is None:
        return []

    my_pos = get_self_position()
    result = []
    for i, cell in enumerate(all_areas):
        if i not in visited and float(np.linalg.norm(cell - my_pos)) <= _PERCEPTION_RADIUS_CM:
            result.append({"id": i, "position": cell.copy()})
    return result


def get_quadrant_target_position() -> dict[int, np.ndarray]:
    """
    Description: Get the 3D target for this drone in its assigned quadrant (task: clustering).
    Delega alla funzione globale e restituisce solo il quadrante di questo drone.
    Returns:
    - dict[int, numpy.ndarray]: {self_id: [x, y, z] in cm} oppure dict completo.
    """
    try:
        import genswarm_overrides.swarmgpt_global_apis as _gapi
        full = _gapi.get_quadrant_target_position()
    except Exception:
        return {}
    if _self_id in full:
        return {_self_id: full[_self_id]}
    return full


def get_all_drones_initial_position() -> dict[int, np.ndarray]:
    """
    Description: Get the initial 3D positions of all drones.
    Returns:
    - dict[int, numpy.ndarray]: drone_id -> [x, y, z] in cm.
    """
    return {k: v.copy() for k, v in _initial_positions_cm.items()}


def get_assigned_task() -> Any:
    """
    Description: Get the task assigned to this drone by the global allocator.
    Returns:
    - Any: dizionario/oggetto con il task assegnato, o None.
    """
    return _assigned_task
