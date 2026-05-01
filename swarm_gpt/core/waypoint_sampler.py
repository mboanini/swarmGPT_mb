"""WaypointSampler: esegue global_skill.py generato da GenSwarm in un ambiente
mock 3D e campiona le posizioni dei droni → waypoints compatibili con SwarmGPT."""

from __future__ import annotations

import ast
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WaypointSampler:
    """Esegue global_skill.py in un simulatore mock e produce waypoints.

    Le posizioni interne sono in CENTIMETRI (convenzione GenSwarm, 1-indexed).
    L'output _to_waypoints() converte in METRI (convenzione SwarmGPT, 0-indexed time).

    Args:
        n_drones: numero di droni.
        initial_positions_cm: {drone_id(1-indexed): np.array([x,y,z])} in cm.
        settings: dizionario settings.yaml (contiene axswarm.pos_min/pos_max).
    """

    def __init__(
        self,
        n_drones: int,
        initial_positions_cm: dict[int, np.ndarray],
        settings: dict,
    ):
        self.n_drones = n_drones
        self.drone_ids: list[int] = sorted(initial_positions_cm.keys())
        self.initial_positions: dict[int, np.ndarray] = {
            k: v.copy() for k, v in initial_positions_cm.items()
        }
        self.positions: dict[int, np.ndarray] = {
            k: v.copy() for k, v in initial_positions_cm.items()
        }

        lim = settings["axswarm"]
        self.env_bounds = {
            "x_min": float(lim["pos_min"][0]) * 100,
            "x_max": float(lim["pos_max"][0]) * 100,
            "y_min": float(lim["pos_min"][1]) * 100,
            "y_max": float(lim["pos_max"][1]) * 100,
            "z_min": float(lim["pos_min"][2]) * 100,
            "z_max": float(lim["pos_max"][2]) * 100,
        }

        self._snapshots: list[dict[int, np.ndarray]] = []
        self._in_highlevel: bool = False

    # ─── snapshot helpers ────────────────────────────────────────────────────

    def _snapshot(self) -> None:
        self._snapshots.append({k: v.copy() for k, v in self.positions.items()})

    def _clamp(self, pos: np.ndarray) -> np.ndarray:
        lo = np.array([self.env_bounds["x_min"], self.env_bounds["y_min"], self.env_bounds["z_min"]])
        hi = np.array([self.env_bounds["x_max"], self.env_bounds["y_max"], self.env_bounds["z_max"]])
        return np.clip(pos, lo, hi)

    # ─── API base ────────────────────────────────────────────────────────────

    def get_self_id(self) -> int:
        return self.drone_ids[0]

    def get_all_drones_id(self) -> list[int]:
        return list(self.drone_ids)

    def get_self_position(self) -> np.ndarray:
        return self.positions[self.drone_ids[0]].copy()

    def get_self_velocity(self) -> np.ndarray:
        return np.zeros(3)

    def set_self_velocity(self, velocity: Any) -> None:
        pass

    def stop_self(self) -> None:
        pass

    def get_self_radius(self) -> float:
        return 20.0

    def get_surrounding_environment_info(self) -> list:
        return []

    def get_all_drones_initial_position(self) -> dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self.initial_positions.items()}

    def get_environment_range(self) -> dict:
        return dict(self.env_bounds)

    def get_target_formation_points(self) -> list:
        return []

    def get_target_position(self) -> np.ndarray:
        return np.zeros(3)

    def get_prey_position(self) -> np.ndarray:
        return np.zeros(3)

    def get_lead_position(self) -> np.ndarray:
        return np.zeros(3)

    def get_surrounding_unexplored_area(self) -> list:
        return []

    def get_initial_unexplored_areas(self) -> list:
        return []

    def get_quadrant_target_position(self) -> dict:
        return {}

    def get_prey_initial_position(self) -> np.ndarray:
        return np.zeros(3)

    # ─── API comandi posizione ────────────────────────────────────────────────

    def move(self, x: float, y: float, z: float, drone_id: int) -> None:
        pos = self._clamp(np.array([float(x), float(y), float(z)]))
        self.positions[drone_id] = pos
        if not self._in_highlevel:
            self._snapshot()

    def move_z(self, drone_ids: list[int], distance: float) -> None:
        self._in_highlevel = True
        targets = drone_ids if drone_ids else self.drone_ids
        for did in targets:
            p = self.positions[did].copy()
            p[2] = np.clip(
                p[2] + distance,
                self.env_bounds["z_min"],
                self.env_bounds["z_max"],
            )
            self.positions[did] = p
        self._in_highlevel = False
        self._snapshot()

    def rotate(self, angle: float, axis: str) -> None:
        self._in_highlevel = True
        centroid = np.mean(list(self.positions.values()), axis=0)
        theta = np.radians(float(angle))
        c, s = np.cos(theta), np.sin(theta)
        if axis == "z":
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif axis == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            self._in_highlevel = False
            return
        for did in self.drone_ids:
            self.positions[did] = self._clamp(centroid + R @ (self.positions[did] - centroid))
        self._in_highlevel = False
        self._snapshot()

    def form_circle(self, drone_ids: list[int], z_coord: float) -> None:
        self._in_highlevel = True
        dids = drone_ids if drone_ids else self.drone_ids
        n = len(dids)
        # raggio minimo per spacing >= 80 cm tra droni adiacenti
        radius = max(80.0 * n / (2 * np.pi), 80.0)
        cx, cy = np.mean([self.positions[did][:2] for did in dids], axis=0)
        z = float(z_coord)
        for i, did in enumerate(dids):
            a = 2 * np.pi * i / n
            self.positions[did] = self._clamp(
                np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z])
            )
        self._in_highlevel = False
        self._snapshot()

    def center(self, drone_ids: list[int]) -> None:
        self._in_highlevel = True
        dids = drone_ids if drone_ids else self.drone_ids
        centroid = np.mean([self.positions[did] for did in dids], axis=0)
        for did in dids:
            self.positions[did] = self._clamp(centroid.copy())
        self._in_highlevel = False
        self._snapshot()

    def swap(self, drone_id_1: int, drone_id_2: int) -> None:
        self._in_highlevel = True
        self.positions[drone_id_1], self.positions[drone_id_2] = (
            self.positions[drone_id_2].copy(),
            self.positions[drone_id_1].copy(),
        )
        self._in_highlevel = False
        self._snapshot()

    def polygon(self, n_sides: int, height: float) -> None:
        self._in_highlevel = True
        dids = self.drone_ids[:n_sides]
        radius = max(60.0 * len(dids) / (2 * np.pi), 60.0)
        cx, cy = np.mean([self.positions[did][:2] for did in self.drone_ids], axis=0)
        z = float(height)
        for i, did in enumerate(dids):
            a = 2 * np.pi * i / len(dids)
            self.positions[did] = self._clamp(
                np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z])
            )
        self._in_highlevel = False
        self._snapshot()

    def form_star(self, height: float, min_spacing: float, delta_radius: float) -> None:
        self._in_highlevel = True
        n = len(self.drone_ids)
        n_inner = n // 2
        n_outer = n - n_inner
        r_inner = max(float(min_spacing) * n_inner / (2 * np.pi), float(min_spacing))
        r_outer = r_inner + float(delta_radius)
        cx, cy = np.mean([self.positions[did][:2] for did in self.drone_ids], axis=0)
        z = float(height)
        for i, did in enumerate(self.drone_ids):
            if i < n_inner:
                r = r_inner
                a = 2 * np.pi * i / max(n_inner, 1)
            else:
                r = r_outer
                j = i - n_inner
                a = 2 * np.pi * j / max(n_outer, 1) + np.pi / max(n_outer, 1)
            self.positions[did] = self._clamp(
                np.array([cx + r * np.cos(a), cy + r * np.sin(a), z])
            )
        self._in_highlevel = False
        self._snapshot()

    def form_cone(self, delta_height: float, spacing: float, is_inverted: bool) -> None:
        self._in_highlevel = True
        n = len(self.drone_ids)
        cx, cy = np.mean([self.positions[did][:2] for did in self.drone_ids], axis=0)
        base_z = float(np.mean([self.positions[did][2] for did in self.drone_ids]))
        dh = float(delta_height)
        sp = float(spacing)

        assigned = 0
        layer = 0
        while assigned < n:
            n_in_layer = 1 if layer == 0 else min(layer * 4, n - assigned)
            radius = layer * sp
            z_offset = layer * dh * (1 if is_inverted else -1)
            z = base_z + z_offset
            for j in range(n_in_layer):
                if assigned >= n:
                    break
                did = self.drone_ids[assigned]
                if n_in_layer == 1:
                    pos = np.array([cx, cy, z])
                else:
                    a = 2 * np.pi * j / n_in_layer
                    pos = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z])
                self.positions[did] = self._clamp(pos)
                assigned += 1
            layer += 1

        self._in_highlevel = False
        self._snapshot()

    # ─── esecuzione ──────────────────────────────────────────────────────────

    def _build_namespace(self) -> dict:
        return {
            "np": np,
            "numpy": np,
            "get_self_id": self.get_self_id,
            "get_all_drones_id": self.get_all_drones_id,
            "get_self_position": self.get_self_position,
            "get_self_velocity": self.get_self_velocity,
            "set_self_velocity": self.set_self_velocity,
            "stop_self": self.stop_self,
            "get_self_radius": self.get_self_radius,
            "get_environment_range": self.get_environment_range,
            "get_surrounding_environment_info": self.get_surrounding_environment_info,
            "get_all_drones_initial_position": self.get_all_drones_initial_position,
            "get_target_formation_points": self.get_target_formation_points,
            "get_target_position": self.get_target_position,
            "get_prey_position": self.get_prey_position,
            "get_lead_position": self.get_lead_position,
            "get_surrounding_unexplored_area": self.get_surrounding_unexplored_area,
            "get_initial_unexplored_areas": self.get_initial_unexplored_areas,
            "get_quadrant_target_position": self.get_quadrant_target_position,
            "get_prey_initial_position": self.get_prey_initial_position,
            "move": self.move,
            "move_z": self.move_z,
            "rotate": self.rotate,
            "form_circle": self.form_circle,
            "center": self.center,
            "swap": self.swap,
            "form_star": self.form_star,
            "form_cone": self.form_cone,
            "polygon": self.polygon,
        }

    @staticmethod
    def _find_entry_points(code: str) -> list[str]:
        """Funzioni definite ma non chiamate da altre funzioni nel file."""
        tree = ast.parse(code)
        defined: list[str] = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        called: set[str] = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        return [name for name in defined if name not in called]

    def run(self, global_skill_code: str) -> dict | None:
        """Esegue global_skill_code e restituisce il dict waypoints (metri)."""
        if not global_skill_code.strip():
            logger.warning("[WaypointSampler] global_skill_code è vuoto, skip.")
            return None

        namespace: dict = {"__builtins__": __builtins__}
        namespace.update(self._build_namespace())

        try:
            exec(compile(global_skill_code, "global_skill.py", "exec"), namespace)
        except Exception as e:
            logger.error("[WaypointSampler] Errore nella compilazione/exec: %s", e)
            return None

        entry_points = self._find_entry_points(global_skill_code)
        logger.info("[WaypointSampler] Entry points trovati: %s", entry_points)

        for name in entry_points:
            fn = namespace.get(name)
            if callable(fn):
                try:
                    fn()
                    logger.info("[WaypointSampler] Eseguito '%s'", name)
                except Exception as e:
                    logger.warning("[WaypointSampler] '%s' fallita: %s", name, e)

        # snapshot finale se nessuno è stato registrato o le posizioni sono cambiate
        last = self._snapshots[-1] if self._snapshots else {}
        if not self._snapshots or any(
            not np.array_equal(last.get(did), self.positions[did]) for did in self.drone_ids
        ):
            self._snapshot()

        logger.info("[WaypointSampler] %d snapshot registrati", len(self._snapshots))
        return self._to_waypoints()

    # ─── conversione output ───────────────────────────────────────────────────

    def _to_waypoints(self) -> dict | None:
        """Converte i snapshot in formato waypoints SwarmGPT (metri, time 0-indexed)."""
        if not self._snapshots:
            return None

        initial_snap = {did: self.initial_positions[did].copy() for did in self.drone_ids}
        all_snaps = [initial_snap] + self._snapshots

        T = len(all_snaps)
        n = len(self.drone_ids)
        pos = np.zeros((n, T, 3))

        for t, snap in enumerate(all_snaps):
            for di, did in enumerate(self.drone_ids):
                pos[di, t, :] = snap.get(did, self.initial_positions[did]) / 100.0

        time_step = 0.5
        timestamps = np.arange(T, dtype=float) * time_step
        time_arr = np.tile(timestamps, (n, 1))

        return {
            "time": time_arr,
            "pos": pos,
            "vel": np.zeros_like(pos),
            "acc": np.zeros_like(pos),
        }
