"""
alert_system.py

Implements the "intelligent alert mechanism ... based on the detected level of
drowsiness" described in the project abstract. This directly replaces the base
paper's single binary alarm (which fires only when eyes are closed for a hard-coded
10 consecutive frames) with a THREE-LEVEL, cooldown-managed, escalating response:

    0 Alert          -> no alert, dashboard stays green
    1 Drowsy         -> amber visual warning + soft chime
    2 Highly Drowsy  -> red visual warning + escalating audible alarm

A cooldown per level prevents alert spamming once the smoothed prediction settles,
while still re-firing if drowsiness persists past the cooldown window.
"""

import time
import sys

try:
    from playsound import playsound
except ImportError:
    playsound = None


LEVEL_NAMES = {0: "ALERT", 1: "DROWSY", 2: "HIGHLY DROWSY"}


class AlertManager:
    def __init__(self, drowsy_cooldown_sec=4, highly_drowsy_cooldown_sec=2, sound_dir=None):
        self.drowsy_cooldown = drowsy_cooldown_sec
        self.highly_drowsy_cooldown = highly_drowsy_cooldown_sec
        self.last_alert_time = {1: 0.0, 2: 0.0}
        self.sound_dir = sound_dir  # optional dir with chime.wav / siren.wav
        self.event_log = []

    def _cooldown_for(self, level: int) -> float:
        return self.drowsy_cooldown if level == 1 else self.highly_drowsy_cooldown

    def _play_sound(self, level: int):
        if playsound is None or self.sound_dir is None:
            return
        filename = "chime.wav" if level == 1 else "siren.wav"
        path = f"{self.sound_dir}/{filename}"
        try:
            playsound(path, block=False)
        except Exception as e:  # pragma: no cover - best-effort audio
            print(f"[AlertManager] could not play sound: {e}", file=sys.stderr)

    def process(self, level: int) -> dict:
        """
        Called once per smoothed prediction. Returns a dict describing what UI/audio
        action (if any) should be taken this tick.
        """
        now = time.time()
        action = {"level": level, "level_name": LEVEL_NAMES.get(level, "UNKNOWN"), "fire": False}

        if level == 0:
            return action  # nothing to do, driver is alert

        cooldown = self._cooldown_for(level)
        if now - self.last_alert_time[level] >= cooldown:
            self.last_alert_time[level] = now
            action["fire"] = True
            self._play_sound(level)
            self.event_log.append({"time": now, "level": level})

        return action

    def summary(self):
        drowsy_events = sum(1 for e in self.event_log if e["level"] == 1)
        highly_drowsy_events = sum(1 for e in self.event_log if e["level"] == 2)
        return {
            "total_drowsy_alerts": drowsy_events,
            "total_highly_drowsy_alerts": highly_drowsy_events,
        }
