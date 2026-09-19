from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import win32con
import win32gui

from game_interface import BlackBoxGameInterface, GameStatus


@dataclass(frozen=True)
class FinalRewardConfig:
    """Reward for the learned movement policy.

    Combat is handled by the deterministic visual aiming layer, so combat score is
    deliberately a *small* signal here.  The learned policy is paid primarily for
    staying alive and actually traversing free space.  This prevents the degenerate
    strategy where PPO holds one direction into a wall while the aim controller
    continues farming score.
    """

    score_scale: float = 0.00025
    hp_loss_scale: float = 0.250
    death_penalty: float = 12.0
    living_bonus: float = 0.0020
    movement_bonus: float = 0.0250
    idle_penalty: float = 0.0150
    stagnation_penalty: float = 0.0800
    blocked_move_penalty: float = 0.1200
    low_clearance_penalty: float = 0.0600
    edge_penalty_scale: float = 0.0250
    corner_penalty: float = 0.0800
    danger_penalty_scale: float = 0.0200


class FinalBlackBoxEnv(BlackBoxGameInterface):
    """Black-box perception/control environment for SpaceProx.

    The executable remains an external process. The wrapper captures rendered RGB
    pixels with PrintWindow, extracts player/enemy geometry with deterministic CV,
    and sends ordinary Windows keyboard/mouse input. The learned policy chooses one
    of nine movement actions. Target engagement is automatic whenever a valid
    rendered enemy is visible.

    Target selection and mouse aiming are handled by a low-level visual controller.
    This deliberately reduces the sample complexity of learning against one real-time
    Windows game instance while keeping the complete system black-box with respect to
    process memory and game internals.
    """

    MAX_VISIBLE_ENEMIES = 7
    ENEMY_FEATURES_PER_SLOT = 6  # dx, dy, distance, delta-x, delta-y, present
    BASE_VISUAL_FEATURES = 58
    CLEARANCE_FEATURES = 8
    HISTORY_FEATURES = 1
    LAST_ACTION_FEATURES = 9
    OBS_FEATURES = (
        BASE_VISUAL_FEATURES
        + CLEARANCE_FEATURES
        + HISTORY_FEATURES
        + LAST_ACTION_FEATURES
    )
    MOVEMENT_ACTIONS = 9
    ACTIONS = MOVEMENT_ACTIONS
    STAGNATION_WINDOW = 12
    BULLET_SPEED_GAME_PX_S = 560.0
    MAX_TRACK_STEP_PX = 70.0
    CLEARANCE_RAY_PX = 180.0
    MOVEMENT_VECTORS = {
        1: (0.0, -1.0),
        2: (0.0, 1.0),
        3: (-1.0, 0.0),
        4: (1.0, 0.0),
        5: (-0.70710678, -0.70710678),
        6: (0.70710678, -0.70710678),
        7: (-0.70710678, 0.70710678),
        8: (0.70710678, 0.70710678),
    }

    def __init__(
        self,
        game_exe: str | Path | None = None,
        step_seconds: float = 0.080,
        auto_launch: bool = True,
        bring_to_front: bool = False,
        reward_config: FinalRewardConfig | None = None,
        fresh_process_on_first_reset: bool = True,
    ) -> None:
        self.final_reward_config = reward_config or FinalRewardConfig()
        self.fresh_process_on_first_reset = bool(fresh_process_on_first_reset)
        self._fresh_start_done = False
        self._fresh_player: tuple[float, float] | None = None
        self._previous_player: tuple[float, float] | None = None
        self._fresh_enemies: list[tuple[float, float]] = []
        self._previous_enemies: list[tuple[float, float]] = []
        self._enemy_step_deltas: list[tuple[float, float]] = []
        self._position_history: deque[tuple[float, float]] = deque(
            maxlen=self.STAGNATION_WINDOW
        )
        self._visual_game_over = False
        self._force_retry = False
        self._target_center: tuple[float, float] | None = None
        self._target_intercept: tuple[float, float] | None = None
        self._engage_actuated = False
        self._movement_clearances = np.ones((self.CLEARANCE_FEATURES,), dtype=np.float32)
        self._last_movement_action = 0
        super().__init__(
            game_exe=game_exe,
            step_seconds=step_seconds,
            auto_launch=auto_launch,
            bring_to_front=bring_to_front,
        )
        self.action_space = gym.spaces.Discrete(self.MOVEMENT_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBS_FEATURES,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Pixel perception
    # ------------------------------------------------------------------
    @staticmethod
    def detect_player(rgb: np.ndarray) -> tuple[float, float] | None:
        """Detect the green player body, preferring its white center disk."""

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        green = (g > 175) & (g > r * 1.55) & (g > b * 1.35) & (r < 150)
        green[: min(60, green.shape[0]), :] = False
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            (green.astype(np.uint8) * 255), connectivity=8
        )

        candidates: list[tuple[int, int]] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if 80 <= area <= 2200:
                candidates.append((area, index))
        if not candidates:
            return None

        _, body_index = max(candidates)
        x = int(stats[body_index, cv2.CC_STAT_LEFT])
        y = int(stats[body_index, cv2.CC_STAT_TOP])
        width = int(stats[body_index, cv2.CC_STAT_WIDTH])
        height = int(stats[body_index, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - 14)
        y0 = max(60, y - 14)
        x1 = min(rgb.shape[1], x + width + 14)
        y1 = min(rgb.shape[0], y + height + 14)
        roi = rgb[y0:y1, x0:x1]
        rr = roi[:, :, 0]
        gg = roi[:, :, 1]
        bb = roi[:, :, 2]
        white = (rr > 210) & (gg > 225) & (bb > 215)
        w_count, _, w_stats, w_centroids = cv2.connectedComponentsWithStats(
            (white.astype(np.uint8) * 255), connectivity=8
        )
        white_candidates: list[tuple[int, int]] = []
        for index in range(1, w_count):
            area = int(w_stats[index, cv2.CC_STAT_AREA])
            if 25 <= area <= 450:
                white_candidates.append((area, index))
        if white_candidates:
            _, core_index = max(white_candidates)
            cx, cy = w_centroids[core_index]
            return float(x0 + cx), float(y0 + cy)

        cx, cy = centroids[body_index]
        return float(cx), float(cy)

    @staticmethod
    def detect_enemies(rgb: np.ndarray) -> list[tuple[float, float]]:
        """Detect visible red enemy bodies from the rendered frame."""

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        red = (r > 155) & (r > g * 1.65) & (r > b * 1.35) & (g < 125)
        red[: min(60, red.shape[0]), :] = False
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            (red.astype(np.uint8) * 255), connectivity=8
        )
        centers: list[tuple[float, float]] = []
        for index in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if not (70 <= area <= 2800):
                continue
            if width > 80 or height > 80:
                continue
            cx, cy = centroids[index]
            centers.append((float(cx), float(cy)))
        return centers[: FinalBlackBoxEnv.MAX_VISIBLE_ENEMIES]

    @staticmethod
    def detect_visual_game_over(rgb: np.ndarray, player_present: bool) -> bool:
        """Detect the red/dark GAME OVER overlay before title telemetry catches up."""

        if player_present:
            return False
        height, width = rgb.shape[:2]
        border = max(6, int(min(width, height) * 0.025))
        strips = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[-border:, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        r = strips[:, 0].astype(np.int16)
        g = strips[:, 1].astype(np.int16)
        b = strips[:, 2].astype(np.int16)
        red_fraction = float(np.mean((r > 100) & (r > g * 1.7) & (r > b * 1.3)))
        center = rgb[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        center_brightness = float(np.mean(center))
        return bool(red_fraction > 0.12 and center_brightness < 85.0)

    # ------------------------------------------------------------------
    # Tracking and semantic observation
    # ------------------------------------------------------------------
    @staticmethod
    def _signed(value: float, scale: float) -> float:
        return float(np.clip(value / max(scale, 1e-6), -1.0, 1.0))

    def _match_enemy_deltas(
        self,
        current: list[tuple[float, float]],
        previous: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not current or not previous:
            return [(0.0, 0.0) for _ in current]
        unmatched = set(range(len(previous)))
        deltas: list[tuple[float, float]] = []
        for cx, cy in current:
            best_index: int | None = None
            best_distance = float("inf")
            for index in unmatched:
                px, py = previous[index]
                distance = math.hypot(cx - px, cy - py)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is not None and best_distance <= self.MAX_TRACK_STEP_PX:
                px, py = previous[best_index]
                unmatched.remove(best_index)
                deltas.append((cx - px, cy - py))
            else:
                deltas.append((0.0, 0.0))
        return deltas

    @staticmethod
    def _wall_mask(rgb: np.ndarray) -> np.ndarray:
        """Return a binary mask for the blue-gray arena obstacles.

        The threshold is intentionally based only on rendered pixels.  It includes
        both the dark wall body and the cyan wall highlight, then dilates the result
        by roughly the player's collision radius so ray clearance answers the useful
        question: "can the player body move there?"
        """

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        body = (
            (r >= 28)
            & (r <= 105)
            & (g >= 38)
            & (g <= 125)
            & (b >= 55)
            & (b <= 160)
            & (g > r + 4)
            & (b > g + 8)
        )
        highlight = (
            (r >= 45)
            & (r <= 135)
            & (g >= 120)
            & (b >= 165)
            & (b > g + 20)
        )
        mask = ((body | highlight).astype(np.uint8) * 255)
        # Do not let HUD bars at the very top masquerade as gameplay obstacles.
        mask[:55, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        return cv2.dilate(mask, kernel, iterations=1) > 0

    def _movement_clearance_features(
        self,
        rgb: np.ndarray,
        player: tuple[float, float] | None,
    ) -> np.ndarray:
        """Ray-cast free-space estimates for actions W/S/A/D and diagonals."""

        if player is None:
            return np.zeros((self.CLEARANCE_FEATURES,), dtype=np.float32)
        mask = self._wall_mask(rgb)
        height, width = mask.shape
        px, py = player
        result: list[float] = []
        max_ray = float(self.CLEARANCE_RAY_PX)
        for action in range(1, 9):
            dx, dy = self.MOVEMENT_VECTORS[action]
            clear = max_ray
            for distance in np.arange(4.0, max_ray + 0.1, 4.0):
                x = int(round(px + dx * distance))
                y = int(round(py + dy * distance))
                if x < 18 or x >= width - 18 or y < 68 or y >= height - 18:
                    clear = float(distance)
                    break
                if mask[y, x]:
                    clear = float(distance)
                    break
            result.append(float(np.clip(clear / max_ray, 0.0, 1.0)))
        return np.asarray(result, dtype=np.float32)

    def _history_extent_feature(self) -> float:
        if len(self._position_history) < 2:
            return 0.0
        xs = [point[0] for point in self._position_history]
        ys = [point[1] for point in self._position_history]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return float(np.clip(extent / 180.0, 0.0, 1.0))

    def _semantic_features(
        self,
        rgb: np.ndarray,
        player: tuple[float, float] | None,
        previous_player: tuple[float, float] | None,
        enemies: list[tuple[float, float]],
        enemy_deltas: list[tuple[float, float]],
    ) -> np.ndarray:
        height, width = rgb.shape[:2]
        features: list[float] = []
        if player is None:
            return np.zeros((self.OBS_FEATURES,), dtype=np.float32)

        px, py = player
        player_dx = 0.0 if previous_player is None else px - previous_player[0]
        player_dy = 0.0 if previous_player is None else py - previous_player[1]
        left = 0.025 * width
        right = 0.975 * width
        top = 0.105 * height
        bottom = 0.965 * height
        margin_x = 0.10 * width
        margin_y = 0.10 * height

        # 9 player/arena features.
        features.extend(
            [
                float(np.clip((2.0 * px / width) - 1.0, -1.0, 1.0)),
                float(np.clip((2.0 * py / height) - 1.0, -1.0, 1.0)),
                1.0,
                self._signed(player_dx, 0.05 * width),
                self._signed(player_dy, 0.05 * height),
                float(np.clip(1.0 - (px - left) / margin_x, 0.0, 1.0)),
                float(np.clip(1.0 - (right - px) / margin_x, 0.0, 1.0)),
                float(np.clip(1.0 - (py - top) / margin_y, 0.0, 1.0)),
                float(np.clip(1.0 - (bottom - py) / margin_y, 0.0, 1.0)),
            ]
        )

        indexed = list(zip(enemies, enemy_deltas))
        indexed.sort(key=lambda item: math.dist(player, item[0]))
        indexed = indexed[: self.MAX_VISIBLE_ENEMIES]

        # Target summary: count + nearest target dx/dy/distance/delta-x/delta-y/present.
        features.append(float(np.clip(len(indexed) / self.MAX_VISIBLE_ENEMIES, 0.0, 1.0)))
        if indexed:
            (tx, ty), (tdx, tdy) = indexed[0]
            distance = math.hypot(tx - px, ty - py)
            diagonal = math.hypot(width, height)
            features.extend(
                [
                    self._signed(tx - px, 0.5 * width),
                    self._signed(ty - py, 0.5 * height),
                    float(np.clip(distance / diagonal, 0.0, 1.0)),
                    self._signed(tdx, 0.05 * width),
                    self._signed(tdy, 0.05 * height),
                    1.0,
                ]
            )
        else:
            features.extend([0.0] * 6)

        # Seven distance-sorted enemy slots x six features = 42.
        diagonal = math.hypot(width, height)
        for (ex, ey), (edx, edy) in indexed:
            features.extend(
                [
                    self._signed(ex - px, 0.5 * width),
                    self._signed(ey - py, 0.5 * height),
                    float(np.clip(math.hypot(ex - px, ey - py) / diagonal, 0.0, 1.0)),
                    self._signed(edx, 0.05 * width),
                    self._signed(edy, 0.05 * height),
                    1.0,
                ]
            )
        for _ in range(self.MAX_VISIBLE_ENEMIES - len(indexed)):
            features.extend([0.0] * self.ENEMY_FEATURES_PER_SLOT)

        # Movement-specific information.  The old 58-D observation told the policy
        # where enemies were but not where the *walls* were, which made a repeated W
        # action hard to distinguish from a useful W action until after collision.
        # Eight screenshot-derived clearance rays make obstacles observable.
        clearances = self._movement_clearance_features(rgb, player)
        self._movement_clearances = clearances
        features.extend(float(value) for value in clearances)
        features.append(self._history_extent_feature())

        last_action = [0.0] * self.LAST_ACTION_FEATURES
        last_action[int(np.clip(self._last_movement_action, 0, 8))] = 1.0
        features.extend(last_action)

        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.OBS_FEATURES,):
            raise RuntimeError(
                f"Final observation feature bug: expected {(self.OBS_FEATURES,)}, got {result.shape}"
            )
        return result

    def _observation_from_rgb(self, rgb: np.ndarray, initialize: bool = False) -> np.ndarray:
        previous_player = None if initialize else self._fresh_player
        previous_enemies = [] if initialize else list(self._fresh_enemies)
        player = self.detect_player(rgb)
        visual_game_over = self.detect_visual_game_over(rgb, player is not None)
        enemies = self.detect_enemies(rgb) if player is not None and not visual_game_over else []
        deltas = self._match_enemy_deltas(enemies, previous_enemies)

        self._previous_player = previous_player
        self._previous_enemies = previous_enemies
        self._fresh_player = player
        self._fresh_enemies = enemies
        self._enemy_step_deltas = deltas
        self._visual_game_over = visual_game_over
        if player is not None:
            self._last_player_center = player

        return self._semantic_features(rgb, player, previous_player, enemies, deltas)

    # ------------------------------------------------------------------
    # Hierarchical action: learned movement, automatic visual low-level combat
    # ------------------------------------------------------------------
    def decode_action(self, action: int | np.integer | np.ndarray) -> int:
        value = int(np.asarray(action).reshape(-1)[0])
        if value < 0 or value >= int(self.action_space.n):
            raise ValueError(f"Action out of range: {value}")
        return value

    def _nearest_target_index(self) -> int | None:
        if self._fresh_player is None or not self._fresh_enemies:
            return None
        px, py = self._fresh_player
        return min(
            range(len(self._fresh_enemies)),
            key=lambda index: math.hypot(
                self._fresh_enemies[index][0] - px,
                self._fresh_enemies[index][1] - py,
            ),
        )

    def _intercept_point(
        self,
        player: tuple[float, float],
        target: tuple[float, float],
        target_step_delta: tuple[float, float],
    ) -> tuple[float, float]:
        region = self._client_region()
        px, py = player
        tx, ty = target
        vx = target_step_delta[0] / max(self.step_seconds, 1e-3)
        vy = target_step_delta[1] / max(self.step_seconds, 1e-3)
        bullet_speed = self.BULLET_SPEED_GAME_PX_S * (region["width"] / 960.0)
        rx = tx - px
        ry = ty - py

        # Solve ||r + v*t|| = s*t. Fall back to direct aim when tracking is
        # insufficient or the quadratic has no useful positive root.
        a = vx * vx + vy * vy - bullet_speed * bullet_speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        t = 0.0
        if abs(a) < 1e-8:
            if abs(b) > 1e-8:
                candidate = -c / b
                if candidate > 0.0:
                    t = candidate
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
                positive = [value for value in roots if value > 0.0]
                if positive:
                    t = min(positive)
        t = float(np.clip(t, 0.0, 1.0))
        ix = float(np.clip(tx + vx * t, 0.0, region["width"] - 1.0))
        iy = float(np.clip(ty + vy * t, 0.0, region["height"] - 1.0))
        return ix, iy

    def _aim_at_client_point(self, point: tuple[float, float]) -> None:
        import win32api

        region = self._client_region()
        screen_x = int(region["left"] + point[0])
        screen_y = int(region["top"] + point[1])
        for attempt in range(3):
            try:
                win32api.SetCursorPos((screen_x, screen_y))
                return
            except Exception:
                if attempt < 2:
                    time.sleep(0.005)

    def _apply_action(self, action) -> None:
        movement = self.decode_action(action)
        target_index = self._nearest_target_index()
        target_valid = target_index is not None and self._fresh_player is not None
        wanted = set(self.MOVEMENT_KEYS[movement])

        # Combat is intentionally deterministic now. PPO learns only movement.
        should_engage = bool(target_valid)
        self._engage_actuated = should_engage
        self._target_center = None
        self._target_intercept = None

        if should_engage:
            assert target_index is not None
            target = self._fresh_enemies[target_index]
            delta = self._enemy_step_deltas[target_index]
            intercept = self._intercept_point(self._fresh_player, target, delta)
            self._target_center = target
            self._target_intercept = intercept
            self._aim_at_client_point(intercept)
            wanted.add(self.VK_SPACE)

        self._set_pressed_keys(wanted)
        self._last_movement_action = movement

    # ------------------------------------------------------------------
    # Reward and episode handling
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_metrics(
        player: tuple[float, float] | None,
        width: int,
        height: int,
    ) -> tuple[float, bool]:
        if player is None:
            return 0.0, False
        x, y = player
        left = 0.025 * width
        right = 0.975 * width
        top = 0.105 * height
        bottom = 0.965 * height
        mx = 0.07 * width
        my = 0.08 * height
        dx = max(0.0, min(x - left, right - x))
        dy = max(0.0, min(y - top, bottom - y))
        sx = max(0.0, 1.0 - dx / max(mx, 1.0))
        sy = max(0.0, 1.0 - dy / max(my, 1.0))
        return float(min(1.0, max(sx, sy))), bool(sx > 0.35 and sy > 0.35)

    def _stagnant(self) -> tuple[bool, float | None]:
        if len(self._position_history) < self.STAGNATION_WINDOW:
            return False, None
        xs = [point[0] for point in self._position_history]
        ys = [point[1] for point in self._position_history]
        spatial_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return spatial_extent < 35.0, spatial_extent

    def _danger_severity(
        self,
        player: tuple[float, float] | None,
        enemies: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> float:
        """Normalized visual proximity risk in [0, 1].

        This is intentionally a simple external heuristic. It uses only rendered
        player/enemy centers and becomes non-zero when the nearest visible enemy is
        inside roughly 22% of the frame diagonal.
        """

        if player is None or not enemies:
            return 0.0
        px, py = player
        nearest = min(math.hypot(ex - px, ey - py) for ex, ey in enemies)
        threshold = max(1.0, 0.22 * math.hypot(width, height))
        return float(np.clip(1.0 - nearest / threshold, 0.0, 1.0))

    def _reward_terms(
        self,
        previous: GameStatus,
        current: GameStatus,
        movement: int,
        width: int,
        height: int,
        visual_game_over: bool,
        selected_clearance: float,
    ) -> tuple[dict[str, float], dict[str, float | bool | int | None]]:
        cfg = self.final_reward_config
        score_gain = max(0, current.score - previous.score)
        hp_loss = max(0, previous.hp - current.hp)
        died_now = bool(
            visual_game_over
            or (current.wasted and not previous.wasted)
            or (current.hp <= 0 < previous.hp)
        )
        geometry_valid = self._fresh_player is not None and not visual_game_over

        displacement: float | None = None
        if geometry_valid and self._previous_player is not None:
            displacement = math.dist(self._previous_player, self._fresh_player)
        if geometry_valid:
            self._position_history.append(self._fresh_player)

        stagnant, spatial_extent = self._stagnant() if geometry_valid else (False, None)
        blocked = bool(
            geometry_valid
            and movement != 0
            and displacement is not None
            and displacement < 3.0
        )
        edge_severity, in_corner = self._edge_metrics(
            self._fresh_player if geometry_valid else None, width, height
        )
        danger_severity = (
            self._danger_severity(self._fresh_player, self._fresh_enemies, width, height)
            if geometry_valid
            else 0.0
        )
        action_shaping_allowed = geometry_valid and not died_now
        movement_progress = (
            float(np.clip((displacement or 0.0) / 18.0, 0.0, 1.0))
            if action_shaping_allowed and movement != 0
            else 0.0
        )
        clearance_badness = (
            float(np.clip((0.30 - selected_clearance) / 0.30, 0.0, 1.0))
            if action_shaping_allowed and movement != 0
            else 0.0
        )
        terms = {
            "score": score_gain * cfg.score_scale,
            "damage": -hp_loss * cfg.hp_loss_scale,
            "death": -cfg.death_penalty if died_now else 0.0,
            "living": cfg.living_bonus if action_shaping_allowed else 0.0,
            "movement": cfg.movement_bonus * movement_progress,
            "idle": -cfg.idle_penalty if action_shaping_allowed and movement == 0 else 0.0,
            "stagnation": -cfg.stagnation_penalty if action_shaping_allowed and stagnant else 0.0,
            "blocked_move": -cfg.blocked_move_penalty if action_shaping_allowed and blocked else 0.0,
            "low_clearance": -cfg.low_clearance_penalty * clearance_badness,
            "edge": -cfg.edge_penalty_scale * edge_severity if action_shaping_allowed else 0.0,
            "corner": -cfg.corner_penalty if action_shaping_allowed and in_corner else 0.0,
            "danger": -cfg.danger_penalty_scale * danger_severity if action_shaping_allowed else 0.0,
        }
        geometry: dict[str, float | bool | int | None] = {
            "geometry_valid": geometry_valid,
            "visual_game_over": visual_game_over,
            "player_displacement": displacement,
            "spatial_extent": spatial_extent,
            "stagnant": stagnant,
            "blocked_move": blocked,
            "edge_severity": edge_severity,
            "in_corner": in_corner,
            "enemy_count": len(self._fresh_enemies) if geometry_valid else 0,
            "danger_severity": danger_severity,
            "selected_clearance": float(selected_clearance),
            "engage_actuated": bool(self._engage_actuated),
        }
        return terms, geometry

    def _restart_game_process(self) -> None:
        self._release_all_keys()
        hwnd = self._ensure_game_window()
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not win32gui.IsWindow(hwnd):
                break
            time.sleep(0.05)
        self._hwnd = None
        self._ensure_game_window()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.fresh_process_on_first_reset and not self._fresh_start_done:
            self._restart_game_process()
            self._fresh_start_done = True

        self._release_all_keys()
        self._ensure_game_window()
        status = self.read_status()
        if status.paused:
            self._tap_key(self.VK_P)
            time.sleep(0.08)
            status = self.read_status()

        if self._force_retry or status.wasted or status.hp <= 0:
            self._tap_key(self.VK_R)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                time.sleep(0.05)
                status = self.read_status()
                if status.hp > 0 and not status.wasted:
                    break
            if status.hp <= 0 or status.wasted:
                raise RuntimeError("Game did not reset after pressing R.")

        self._fresh_player = None
        self._previous_player = None
        self._fresh_enemies = []
        self._previous_enemies = []
        self._enemy_step_deltas = []
        self._position_history.clear()
        self._visual_game_over = False
        self._force_retry = False
        self._target_center = None
        self._target_intercept = None
        self._engage_actuated = False
        self._movement_clearances = np.ones((self.CLEARANCE_FEATURES,), dtype=np.float32)
        self._last_movement_action = 0

        rgb = self.capture_rgb()
        observation = self._observation_from_rgb(rgb, initialize=True)
        self._last_status = status
        self._episode_steps = 0
        self._episode_reward = 0.0
        info = self._make_info(status)
        info.update(
            {
                "geometry_valid": self._fresh_player is not None,
                "enemy_count": len(self._fresh_enemies),
                "visual_game_over": False,
            }
        )
        return observation, info

    def step(self, action):
        if self._last_status is None:
            raise RuntimeError("reset() must be called before step().")
        previous_status = self._last_status
        movement = self.decode_action(action)
        selected_clearance = (
            float(self._movement_clearances[movement - 1])
            if movement != 0 and len(self._movement_clearances) >= movement
            else 1.0
        )
        self._apply_action(action)
        time.sleep(self.step_seconds)
        rgb = self.capture_rgb()
        observation = self._observation_from_rgb(rgb)
        current_status = self.read_status()
        terms, geometry = self._reward_terms(
            previous_status,
            current_status,
            movement,
            rgb.shape[1],
            rgb.shape[0],
            self._visual_game_over,
            selected_clearance,
        )
        reward = float(sum(terms.values()))
        terminated = bool(
            self._visual_game_over or current_status.wasted or current_status.hp <= 0
        )

        self._episode_steps += 1
        self._episode_reward += reward
        self._last_status = current_status
        if terminated:
            self._force_retry = bool(self._visual_game_over and not current_status.wasted)
            self._release_all_keys()

        info = self._make_info(current_status, terms)
        info.update(geometry)
        info["action_mode"] = "movement_only_auto_combat"
        if self._fresh_player is not None:
            info["player_x"] = float(self._fresh_player[0])
            info["player_y"] = float(self._fresh_player[1])
        if self._target_center is not None:
            info["target_x"] = float(self._target_center[0])
            info["target_y"] = float(self._target_center[1])
        if self._target_intercept is not None:
            info["intercept_x"] = float(self._target_intercept[0])
            info["intercept_y"] = float(self._target_intercept[1])
        return observation, reward, terminated, False, info
