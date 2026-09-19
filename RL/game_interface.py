from __future__ import annotations

import ctypes
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


TITLE_RE = re.compile(r"HP\s+(\d+)\s+\|\s+Ammo\s+(\d+)\s+\|\s+Score\s+(\d+)")
BEST_RE = re.compile(r"\|\s+Best\s+(\d+)")


@dataclass(frozen=True)
class GameStatus:
    hp: int
    ammo: int
    score: int
    best: int
    wasted: bool
    paused: bool
    title: str


class BlackBoxGameInterface(gym.Env):
    """Low-level Win32 interface shared by the single current RL environment.

    This class intentionally contains no policy observation, reward or learning
    logic. It only owns the black-box boundary: window discovery, PrintWindow RGB
    capture, public title telemetry, and ordinary keyboard input.
    """

    metadata = {"render_modes": []}

    VK_W = 0x57
    VK_A = 0x41
    VK_S = 0x53
    VK_D = 0x44
    VK_R = 0x52
    VK_P = 0x50
    VK_SPACE = win32con.VK_SPACE

    MOVEMENT_KEYS = {
        0: (),
        1: (VK_W,),
        2: (VK_S,),
        3: (VK_A,),
        4: (VK_D,),
        5: (VK_W, VK_A),
        6: (VK_W, VK_D),
        7: (VK_S, VK_A),
        8: (VK_S, VK_D),
    }

    def __init__(
        self,
        game_exe: str | Path | None = None,
        step_seconds: float = 0.080,
        auto_launch: bool = True,
        bring_to_front: bool = False,
    ) -> None:
        super().__init__()
        rl_root = Path(__file__).resolve().parents[1]
        default_exe = rl_root / "Game" / "SpaceProx.exe"
        self.game_exe = Path(game_exe) if game_exe else default_exe
        self.step_seconds = float(step_seconds)
        self.auto_launch = bool(auto_launch)
        self.bring_to_front = bool(bring_to_front)

        self._hwnd: int | None = None
        self._pressed_keys: set[int] = set()
        self._last_player_center: tuple[float, float] | None = None
        self._last_status: GameStatus | None = None
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._ensure_game_window()

    @staticmethod
    def enum_game_windows() -> list[int]:
        matches: list[int] = []

        def callback(hwnd: int, _: object) -> bool:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.startswith("SpaceProx") and "D3D11 Arena" in title:
                    matches.append(hwnd)
            return True

        win32gui.EnumWindows(callback, None)
        return matches

    def _ensure_game_window(self) -> int:
        if self._hwnd and win32gui.IsWindow(self._hwnd):
            return self._hwnd

        matches = self.enum_game_windows()
        if not matches and self.auto_launch:
            if not self.game_exe.exists():
                raise FileNotFoundError(f"SpaceProx executable not found: {self.game_exe}")
            subprocess.Popen([str(self.game_exe)], cwd=str(self.game_exe.parent))
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                time.sleep(0.08)
                matches = self.enum_game_windows()
                if matches:
                    break
        if not matches:
            raise RuntimeError("Could not find a visible SpaceProx window.")

        self._hwnd = matches[0]
        if self.bring_to_front:
            try:
                win32gui.ShowWindow(self._hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self._hwnd)
            except Exception:
                pass
        return self._hwnd

    @property
    def hwnd(self) -> int:
        return self._ensure_game_window()

    def _client_region(self) -> dict[str, int]:
        hwnd = self._ensure_game_window()
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        _, _, width, height = win32gui.GetClientRect(hwnd)
        return {
            "left": left,
            "top": top,
            "width": max(1, width),
            "height": max(1, height),
        }

    def capture_rgb(self) -> np.ndarray:
        """Capture the game client directly from its HWND with PrintWindow."""
        hwnd = self._ensure_game_window()
        _, _, width, height = win32gui.GetClientRect(hwnd)
        window_dc = win32gui.GetDC(hwnd)
        source_dc = win32ui.CreateDCFromHandle(window_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bitmap)
        try:
            ok = ctypes.windll.user32.PrintWindow(hwnd, memory_dc.GetSafeHdc(), 3)
            if not ok:
                raise RuntimeError("PrintWindow failed to capture SpaceProx")
            raw = bitmap.GetBitmapBits(True)
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        finally:
            for cleanup in (
                lambda: win32gui.DeleteObject(bitmap.GetHandle()),
                memory_dc.DeleteDC,
                source_dc.DeleteDC,
                lambda: win32gui.ReleaseDC(hwnd, window_dc),
            ):
                try:
                    cleanup()
                except Exception:
                    pass

    def read_status(self) -> GameStatus:
        title = win32gui.GetWindowText(self._ensure_game_window())
        match = TITLE_RE.search(title)
        if match:
            hp, ammo, score = (int(match.group(i)) for i in range(1, 4))
        elif self._last_status is not None:
            hp, ammo, score = (
                self._last_status.hp,
                self._last_status.ammo,
                self._last_status.score,
            )
        else:
            hp, ammo, score = 100, 24, 0
        best_match = BEST_RE.search(title)
        best = (
            int(best_match.group(1))
            if best_match
            else max(score, self._last_status.best if self._last_status else 0)
        )
        return GameStatus(
            hp=hp,
            ammo=ammo,
            score=score,
            best=best,
            wasted="WASTED" in title,
            paused="PAUSED" in title,
            title=title,
        )

    @staticmethod
    def _key_down(vk: int) -> None:
        win32api.keybd_event(vk, 0, 0, 0)

    @staticmethod
    def _key_up(vk: int) -> None:
        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

    def _set_pressed_keys(self, wanted: set[int]) -> None:
        for vk in self._pressed_keys - wanted:
            self._key_up(vk)
        for vk in wanted - self._pressed_keys:
            self._key_down(vk)
        self._pressed_keys = set(wanted)

    def _release_all_keys(self) -> None:
        self._set_pressed_keys(set())

    def _tap_key(self, vk: int, duration: float = 0.035) -> None:
        self._key_down(vk)
        time.sleep(duration)
        self._key_up(vk)

    def _make_info(
        self,
        status: GameStatus,
        reward_terms: dict[str, float] | None = None,
    ) -> dict:
        info = {
            "hp": status.hp,
            "ammo": status.ammo,
            "score": status.score,
            "best": status.best,
            "wasted": status.wasted,
            "episode_steps": self._episode_steps,
            "episode_reward_live": self._episode_reward,
        }
        if reward_terms is not None:
            info["reward_terms"] = reward_terms
        return info

    def close(self) -> None:
        self._release_all_keys()
        super().close()

