"""Mirror display: fullscreen pygame, portrait, black background (only lit pixels show through the glass).

States: idle -> thinking -> reveal -> idle. All animation clusters around display.focus_y,
the camera's position, so people look toward the lens.

Standalone, this is build step 1-3 (the "dumb" version, no recognition):
    python mirror/display.py --windowed           # keys drive the show and the hat
    python mirror/display.py --windowed --no-hat  # display only
    python mirror/display.py --windowed --no-hat --demo   # cycles through sortings on its own

Keys:  1-9, 0  sort person with id 1-10      SPACE  random person
       F1-F4   guest -> house 1-4            S      stop / back to idle
       D       debug overlay                 ESC    quit
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from common import Config, House, People, Person, load_config, setup_logging  # noqa: E402

log = logging.getLogger("sortinghat.display")

Color = tuple[int, int, int]
GOLD: Color = (255, 205, 130)
PALE: Color = (190, 210, 255)
DIM: Color = (150, 140, 120)


def ease_out_back(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ramp(t: float, start: float, dur: float) -> float:
    return min(max((t - start) / dur, 0.0), 1.0)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    radius: int
    color: Color
    drag: float = 0.0
    gravity: float = 0.0
    twinkle: float = 0.0
    age: float = 0.0
    orbit_phase: float | None = None  # set = positioned by the thinking ring, not by physics
    orbit_r: float = 1.0

    def alpha(self) -> float:
        # fade in over the first 20% of life, out over the last 40%
        p = 1 - self.life / self.max_life
        a = 1.0 if self.orbit_phase is not None else min(p / 0.2, 1.0, self.life / (self.max_life * 0.4))
        if self.twinkle:
            a *= 0.65 + 0.35 * math.sin(self.age * self.twinkle)
        return max(0.0, min(1.0, a))


class Sprites:
    """Cache of soft glow dots and halos (built once with numpy, reused every frame)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, Color, bool], pygame.Surface] = {}

    def glow(self, radius: int, color: Color, core: bool = True) -> pygame.Surface:
        key = (radius, color, core)
        surf = self._cache.get(key)
        if surf is None:
            r = max(2, radius)
            yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32) + 0.5
            d = np.sqrt(xx * xx + yy * yy) / r
            alpha = np.clip(1 - d, 0, 1) ** 2.2
            surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            rgb = np.empty((2 * r, 2 * r, 3), np.float32)
            rgb[:] = color
            if core:  # white-hot centre so sparks read as light, not paint
                c = np.clip(1 - d * 4, 0, 1)[..., None]
                rgb = rgb * (1 - c) + 255 * c
            px = pygame.surfarray.pixels3d(surf)
            px[:] = rgb.transpose(1, 0, 2).astype(np.uint8)
            del px
            pa = pygame.surfarray.pixels_alpha(surf)
            pa[:] = (alpha.T * 255).astype(np.uint8)
            del pa
            self._cache[key] = surf
        return surf


class MirrorDisplay:
    def __init__(self, cfg: Config, windowed: bool = False):
        self.cfg = cfg
        self.dcfg = cfg.display
        self.houses = cfg.houses
        self.windowed = windowed or not self.dcfg.fullscreen
        self._events: queue.Queue[tuple[str, Person | None]] = queue.Queue()
        self.state = "idle"
        self.person: Person | None = None
        self.state_t0 = time.monotonic()
        self.particles: list[Particle] = []
        self.sprites = Sprites()
        self.debug = False
        self.caption: str | None = None  # what the hat is currently saying (set from the show thread)
        self.caption_t0 = 0.0
        self._fonts: dict[tuple[bool, int], pygame.font.Font] = {}
        self._text_cache: dict[tuple[str, int, Color, Color | None], pygame.Surface] = {}
        self._crest_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._spawn_acc = 0.0
        self._err_count = 0
        self._phrase_order: list[str] = []
        self.status_line: Callable[[], str] | None = None  # extra debug text, set by main.py

    # ------------------------------------------------------------------ thread-safe API

    def post(self, state: str, person: Person | None) -> None:
        """Called from any thread (show controller)."""
        self._events.put((state, person))

    def set_caption(self, text: str | None) -> None:
        """Show what the hat is saying under the thinking ring (None = clear)."""
        self.caption = text
        self.caption_t0 = time.monotonic()

    # ------------------------------------------------------------------ setup

    def _open(self) -> None:
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Sorting Hat Mirror")
        if self.windowed:
            w, h = self.dcfg.window_size
            if self.dcfg.rotate in (90, 270):
                w, h = h, w
            self.screen = pygame.display.set_mode((w, h))
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        sw, sh = self.screen.get_size()
        if self.dcfg.rotate in (90, 270):
            self.canvas = pygame.Surface((sh, sw))
        elif self.dcfg.rotate == 180:
            self.canvas = pygame.Surface((sw, sh))
        else:
            self.canvas = self.screen
        self.W, self.H = self.canvas.get_size()
        self.fx, self.fy = self.W / 2, self.H * self.dcfg.focus_y
        self.unit = self.W / 1080  # scale everything off a 1080px-wide portrait screen
        log.info("display %dx%d (canvas %dx%d, rotate %d)", sw, sh, self.W, self.H, self.dcfg.rotate)
        self._report_crests()

    def font(self, size: int, title: bool = False) -> pygame.font.Font:
        """Body font, or the display face for headings (falls back to the body font)."""
        key = (title, size)
        f = self._fonts.get(key)
        if f is None:
            for path in ((self.dcfg.title_font, self.dcfg.font) if title else (self.dcfg.font,)):
                if not path:
                    continue
                p = Path(path)
                if not p.is_absolute():
                    p = self.cfg.resolve(self.cfg.paths.assets) / p
                    if not p.exists():
                        p = self.cfg.resolve(path)
                try:
                    f = pygame.font.Font(str(p), size)
                    break
                except (OSError, FileNotFoundError) as e:
                    log.warning("font %s unusable (%s)", path, e)
            if f is None:
                f = pygame.font.SysFont("cinzel,garamond,palatino,georgia,dejavuserif,liberationserif,serif", size)
            self._fonts[key] = f
        return f

    # ------------------------------------------------------------------ main loop

    def run(self, on_key: Callable[[int], None] | None = None, should_stop: Callable[[], bool] = lambda: False,
            dev: bool = False) -> None:
        self._open()
        clock = pygame.time.Clock()
        last = time.monotonic()
        while not should_stop():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if ev.key == pygame.K_d:
                        self.debug = not self.debug
                    elif on_key:
                        on_key(ev.key)
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            try:
                self._drain_events()
                self._frame(dt)
            except Exception:
                if dev:
                    raise
                # party mode: never die on a render glitch; reset to a known state
                self._err_count += 1
                if self._err_count < 20 or self._err_count % 500 == 0:
                    log.exception("render error #%d", self._err_count)
                self._enter("idle", None)
            pygame.display.flip()
            clock.tick(self.dcfg.fps)
        pygame.quit()

    def _drain_events(self) -> None:
        while True:
            try:
                state, person = self._events.get_nowait()
            except queue.Empty:
                return
            self._enter(state, person)

    def _enter(self, state: str, person: Person | None) -> None:
        self.state, self.person = state, person
        self.state_t0 = time.monotonic()
        self.caption = None
        if state == "thinking":
            self.particles.clear()
            self._phrase_order = random.sample(self.dcfg.thinking_phrases, len(self.dcfg.thinking_phrases))
            self._spawn_orbit()
        elif state == "reveal" and person:
            self._burst(self.houses[person.house])
        elif state == "idle":
            # let reveal sparks die out naturally rather than vanishing
            self.particles = [p for p in self.particles if p.orbit_phase is None]
            for p in self.particles:
                p.life = min(p.life, 1.0)

    # ------------------------------------------------------------------ frame

    def _frame(self, dt: float) -> None:
        t = time.monotonic() - self.state_t0
        c = self.canvas
        c.fill((0, 0, 0))
        if self.state == "idle":
            self._draw_idle(t, dt)
        elif self.state == "thinking":
            self._draw_thinking(t, dt)
        elif self.state == "reveal" and self.person:
            self._draw_reveal(t, dt, self.person)
        self._update_particles(dt)
        if self.debug:
            self._draw_debug(dt)
        if self.canvas is not self.screen:
            rotated = pygame.transform.rotate(self.canvas, -self.dcfg.rotate)
            self.screen.blit(rotated, (0, 0))

    def _update_particles(self, dt: float) -> None:
        alive: list[Particle] = []
        for p in self.particles:
            p.age += dt
            if p.orbit_phase is None:
                p.life -= dt
            if p.life <= 0:
                continue
            if p.orbit_phase is None:  # free particle: physics. Orbit dots are placed by _draw_thinking.
                if p.drag:
                    k = math.exp(-p.drag * dt)
                    p.vx *= k
                    p.vy *= k
                p.vy += p.gravity * dt
                p.x += p.vx * dt
                p.y += p.vy * dt
            a = p.alpha()
            if a > 0.01:
                s = self.sprites.glow(p.radius, p.color)
                s.set_alpha(int(255 * a))
                self.canvas.blit(s, (p.x - p.radius, p.y - p.radius))
            alive.append(p)
        self.particles = alive

    # ------------------------------------------------------------------ idle

    def _draw_idle(self, t: float, dt: float) -> None:
        u = self.unit
        # sparse drifting motes, concentrated around the camera
        target = 18
        self._spawn_acc += dt * target / 6.0
        while self._spawn_acc >= 1 and len(self.particles) < target:
            self._spawn_acc -= 1
            life = random.uniform(4, 9)
            self.particles.append(Particle(
                x=random.gauss(self.fx, self.W * 0.22), y=random.gauss(self.fy + 80 * u, self.H * 0.12),
                vx=random.uniform(-6, 6) * u, vy=random.uniform(-18, -6) * u,
                life=life, max_life=life, radius=int(random.uniform(5, 13) * u) + 2,
                color=random.choice([GOLD, PALE, GOLD]), twinkle=random.uniform(1.5, 4)))
        self._spawn_acc = min(self._spawn_acc, 1.0)
        # a faint breathing glow at the lens: draws the eye without being "a screen"
        breathe = 0.5 + 0.5 * math.sin(t * 0.8)
        self._blit_halo(GOLD, int(110 * u), 0.10 + 0.08 * breathe)
        # occasional hint text
        every = self.dcfg.idle_text_every_s
        if every > 0 and self.dcfg.idle_text:
            phase = (t + every - 6) % every  # first appearance ~6s after going idle
            if phase < 5:
                a = math.sin(math.pi * phase / 5) * 0.6
                self._blit_text(self.dcfg.idle_text, int(54 * u), DIM, self.fx, self.fy + 230 * u, a)

    # ------------------------------------------------------------------ thinking

    def _spawn_orbit(self) -> None:
        for _ in range(56):
            self.particles.append(Particle(
                x=self.fx, y=self.fy, vx=0, vy=0, life=1.0, max_life=1.0,
                radius=int(random.uniform(5, 12) * self.unit) + 2,
                color=random.choice([GOLD, GOLD, PALE, (255, 240, 210)]), twinkle=random.uniform(3, 7),
                orbit_phase=random.uniform(0, 2 * math.pi), orbit_r=random.uniform(0.85, 1.15)))

    def _draw_thinking(self, t: float, dt: float) -> None:
        u = self.unit
        settle = ease_out_back(ramp(t, 0, 1.2))
        base_r = (520 - 330 * settle) * u  # ring contracts onto the lens
        spin = 0.6 + 0.5 * min(t / 4, 1.0)  # speeds up as the hat "decides"
        for p in self.particles:
            if p.orbit_phase is None:
                continue
            phase = p.orbit_phase + t * spin * (1.2 if p.orbit_r > 1 else 0.9)
            r = base_r * p.orbit_r * (1 + 0.08 * math.sin(t * 2.3 + p.orbit_phase * 3))
            p.x = self.fx + math.cos(phase) * r
            p.y = self.fy + math.sin(phase) * r * 0.82
        pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        self._blit_halo(GOLD, int(220 * u), (0.12 + 0.12 * pulse) * ramp(t, 0, 0.8))
        if self.caption:  # the hat's actual line, wrapped under the ring
            a = min((time.monotonic() - self.caption_t0) / 0.6, 1.0)
            size = int(44 * u)
            maxw = int(self.W * 0.88)
            lines = self._wrap(self.caption, self.font(size), maxw)[:4]
            lh = int(size * 1.3)
            y0 = self.fy + 330 * u - (len(lines) - 1) * lh / 2
            for j, ln in enumerate(lines):
                self._blit_text(ln, size, GOLD, self.fx, y0 + j * lh, a * 0.95)
            return
        phrases = self._phrase_order or ["Hmm..."]
        per = 2.4
        idx = int(t // per) % len(phrases)
        local = t % per
        a = min(local / 0.5, 1.0, (per - local) / 0.5) * ramp(t, 0.6, 0.6)
        self._blit_text(phrases[idx], int(58 * u), GOLD, self.fx, self.fy + 330 * u, a * 0.85)

    def _wrap(self, text: str, font: pygame.font.Font, maxw: int) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            joined = f"{cur} {w}".strip()
            if not cur or font.size(joined)[0] <= maxw:
                cur = joined
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # ------------------------------------------------------------------ reveal

    def _burst(self, house: House) -> None:
        self.particles = [p for p in self.particles if p.orbit_phase is None]  # drop the thinking ring
        prim, sec = House.rgb(house.primary), House.rgb(house.secondary)
        u = self.unit
        for _ in range(160):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(200, 900) * u
            life = random.uniform(1.5, 3.5)
            self.particles.append(Particle(
                x=self.fx, y=self.fy + 150 * u, vx=math.cos(ang) * spd, vy=math.sin(ang) * spd,
                life=life, max_life=life, radius=int(random.uniform(5, 14) * u) + 2,
                color=random.choice([prim, prim, sec, (255, 255, 255)]), drag=1.8, gravity=120 * u))

    def _draw_reveal(self, t: float, dt: float, person: Person) -> None:
        u = self.unit
        house = self.houses[person.house]
        prim, sec = House.rgb(house.primary), House.rgb(house.secondary)
        hold = self.cfg.show.reveal_s
        fade = 1.0 - ramp(t, hold - 1.2, 1.2)  # fade out at the end of the hold
        cy = self.fy + 150 * u  # crest sits just below the lens

        # colour wash: bright flash that settles into a slow pulse
        flash = 1 - ramp(t, 0, 1.0)
        pulse = 0.5 + 0.5 * math.sin(t * 1.6)
        self._blit_halo(prim, int(760 * u), (0.30 + 0.45 * flash + 0.08 * pulse) * fade, cy)
        self._blit_halo((255, 255, 255), int(320 * u), 0.5 * flash, cy)
        # faint wash anchoring the title block into the lower half of a tall screen
        self._blit_halo(prim, int(700 * u), (0.09 + 0.04 * pulse) * fade, cy + 460 * u)

        # trickle of embers rising off the crest
        self._spawn_acc += dt * 14
        while self._spawn_acc >= 1:
            self._spawn_acc -= 1
            life = random.uniform(2, 4)
            self.particles.append(Particle(
                x=random.gauss(self.fx, 190 * u), y=cy + random.uniform(-160, 260) * u,
                vx=random.uniform(-15, 15) * u, vy=random.uniform(-90, -40) * u, life=life, max_life=life,
                radius=int(random.uniform(4, 10) * u) + 2, color=random.choice([prim, sec]), twinkle=5))

        # crest
        size = int(520 * u)
        scale = ease_out_back(ramp(t, 0.1, 0.9))
        if scale > 0.02:
            crest = self._crest(house, size)
            cw, ch = crest.get_size()
            sc = pygame.transform.smoothscale(crest, (max(1, int(cw * scale)), max(1, int(ch * scale))))
            sc.set_alpha(int(255 * fade))
            self.canvas.blit(sc, (self.fx - sc.get_width() / 2, cy - sc.get_height() / 2))

        # names
        if person.name:
            self._blit_text(person.name, int(84 * u), (255, 245, 225), self.fx, self.fy - 175 * u,
                            ramp(t, 0.3, 0.7) * fade, glow=sec, title=True)
        self._blit_text(house.label.upper(), int(150 * u), sec, self.fx, cy + 448 * u,
                        ramp(t, 0.5, 0.8) * fade, glow=prim, title=True)
        tagline = "  ·  ".join(x for x in (house.animal, house.motto) if x)
        if tagline:
            self._blit_text(tagline, int(52 * u), DIM, self.fx, cy + 590 * u, ramp(t, 1.4, 1.0) * fade * 0.8)

    # ------------------------------------------------------------------ drawing helpers

    def _blit_halo(self, color: Color, radius: int, alpha: float, cy: float | None = None) -> None:
        if alpha <= 0.01:
            return
        halo = self.sprites.glow(radius, color, core=False)
        halo.set_alpha(int(255 * min(alpha, 1.0)))
        y = self.fy if cy is None else cy
        self.canvas.blit(halo, (self.fx - radius, y - radius))

    def _blit_text(self, text: str, size: int, color: Color, cx: float, cy: float, alpha: float,
                   glow: Color | None = None, title: bool = False) -> None:
        if alpha <= 0.01 or not text:
            return
        key = (text, size, color, glow, title)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = self._render_text(text, size, color, glow or color, title)
            if len(self._text_cache) > 200:
                self._text_cache.clear()
            self._text_cache[key] = surf
        # shrink to fit the screen width
        maxw = self.W * 0.92
        if surf.get_width() > maxw:
            k = maxw / surf.get_width()
            surf = pygame.transform.smoothscale(surf, (int(surf.get_width() * k), int(surf.get_height() * k)))
        surf.set_alpha(int(255 * min(alpha, 1.0)))
        self.canvas.blit(surf, (cx - surf.get_width() / 2, cy - surf.get_height() / 2))

    def _render_text(self, text: str, size: int, color: Color, glow: Color, title: bool = False) -> pygame.Surface:
        font = self.font(size, title)
        base = font.render(text, True, color)
        pad = max(4, size // 3)
        w, h = base.get_width() + 2 * pad, base.get_height() + 2 * pad
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        g = pygame.Surface((w, h), pygame.SRCALPHA)
        g.blit(font.render(text, True, glow), (pad, pad))
        small = pygame.transform.smoothscale(g, (max(1, w // 8), max(1, h // 8)))
        blur = pygame.transform.smoothscale(small, (w, h))
        out.blit(blur, (0, 0))
        out.blit(blur, (0, 0))
        out.blit(base, (pad, pad))
        return out

    def crest_file(self, house: House) -> Path | None:
        """assets/crests/<house>.png (or .webp/.jpg), matched case-insensitively."""
        folder = self.cfg.resolve(self.cfg.paths.assets) / "crests"
        if not folder.is_dir():
            return None
        for ext in (".png", ".webp", ".jpg", ".jpeg"):
            for p in folder.iterdir():
                if p.stem.lower() == house.key.lower() and p.suffix.lower() == ext:
                    return p
        return None

    def _load_crest_image(self, path: Path, size: int) -> pygame.Surface | None:
        try:
            img = pygame.image.load(str(path)).convert_alpha()
        except (pygame.error, FileNotFoundError) as e:
            log.warning("crest %s unreadable (%s): using the drawn shield", path.name, e)
            return None
        # A solid light background shows up as a glowing rectangle behind the two-way mirror.
        w, h = img.get_size()
        corners = [img.get_at((x, y)) for x in (0, w - 1) for y in (0, h - 1)]
        if all(c.a > 200 and (c.r + c.g + c.b) / 3 > 180 for c in corners):
            log.warning("crest %s has a light, opaque background: it will glow as a box on the mirror. "
                        "Use a PNG with a transparent (or black) background.", path.name)
        # trim transparent margins so every house's crest renders at the same visual size
        bounds = img.get_bounding_rect(min_alpha=8)
        if bounds.width > 0 and bounds.height > 0:
            img = img.subsurface(bounds).copy()
        k = size / max(img.get_size())
        return pygame.transform.smoothscale(img, (max(1, int(img.get_width() * k)), max(1, int(img.get_height() * k))))

    def _crest(self, house: House, size: int) -> pygame.Surface:
        key = (house.key, size)
        if key in self._crest_cache:
            return self._crest_cache[key]
        path = self.crest_file(house)
        surf = self._load_crest_image(path, size) if path else None
        if surf is None:
            surf = self._draw_shield(house, size)
        self._crest_cache[key] = surf
        return surf

    def _report_crests(self) -> None:
        folder = self.cfg.resolve(self.cfg.paths.assets) / "crests"
        for house in self.houses.values():
            path = self.crest_file(house)
            if path:
                log.info("crest %-10s -> %s", house.key, path.name)
            else:
                log.info("crest %-10s -> drawn shield (add %s/%s.png for the real crest)", house.key, folder, house.key)
            self._crest(house, int(400 * self.unit))  # preload now: no hitch at reveal time

    def _draw_shield(self, house: House, size: int) -> pygame.Surface:
        """Procedural crest: shield in house colours with the house initial. Drawn 2x then downscaled (AA)."""
        s = size * 2
        w, h = int(s * 0.82), s
        prim, sec = House.rgb(house.primary), House.rgb(house.secondary)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        def shield(inset: float) -> list[tuple[float, float]]:
            pts: list[tuple[float, float]] = []
            l, r, top = inset, w - inset, inset
            # top edge with a shallow dip in the middle
            for i in range(21):
                x = l + (r - l) * i / 20
                pts.append((x, top + 0.04 * h * math.sin(math.pi * i / 20)))
            # right side down then curve to the point
            for i in range(1, 31):
                k = i / 30
                x = r - (r - w / 2) * (k ** 2.2)
                y = top + (h - 2 * inset) * (0.35 + 0.65 * k)
                pts.append((x, y))
            for i in range(29, 0, -1):
                k = i / 30
                x = l + (w / 2 - l) * (k ** 2.2)
                y = top + (h - 2 * inset) * (0.35 + 0.65 * k)
                pts.append((x, y))
            pts.append((l, top + (h - 2 * inset) * 0.35))
            return pts

        pygame.draw.polygon(surf, sec, shield(0))
        pygame.draw.polygon(surf, prim, shield(s * 0.035))
        inner = [(x, y) for x, y in shield(s * 0.07)]
        darker = tuple(int(c * 0.55) for c in prim)
        pygame.draw.polygon(surf, darker, inner)
        # diagonal band
        band = pygame.Surface((w, h), pygame.SRCALPHA)
        bw = s * 0.11
        pygame.draw.polygon(band, (*sec, 170), [(0, h * 0.18), (bw, h * 0.18 - bw * 0.3), (w, h * 0.78), (w - bw, h * 0.78 + bw * 0.3)])
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), inner)
        band.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(band, (0, 0))
        # house initial
        letter = self.font(int(s * 0.45)).render(house.label[:1].upper(), True, sec)
        surf.blit(letter, (w / 2 - letter.get_width() / 2, h * 0.42 - letter.get_height() / 2))
        return pygame.transform.smoothscale(surf, (w // 2, h // 2))

    def _draw_debug(self, dt: float) -> None:
        line = f"{self.state}  {1 / max(dt, 1e-3):.0f}fps  particles={len(self.particles)}"
        if self.status_line:
            line += "  " + self.status_line()
        txt = self.font(int(24 * self.unit) + 8).render(line, True, (120, 120, 120))
        self.canvas.blit(txt, (10, self.H - txt.get_height() - 10))


# ---------------------------------------------------------------------------- standalone (dumb version)


def key_handler(people: People, cfg: Config, trigger: Callable[[Person], None], stop: Callable[[], None]
                ) -> Callable[[int], None]:
    """Shared keyboard mapping for display.py and main.py."""
    digit_keys = {getattr(pygame, f"K_{d}"): (d if d else 10) for d in range(10)}
    house_keys = dict(zip([pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4], cfg.houses))

    def on_key(key: int) -> None:
        if key in digit_keys:
            p = people.by_id(digit_keys[key])
            if p:
                trigger(p)
            else:
                log.warning("no person with id %d in people.json", digit_keys[key])
        elif key == pygame.K_SPACE:
            everyone = people.all()
            trigger(random.choice(everyone) if everyone else people.guest())
        elif key in house_keys:
            trigger(people.guest(house_keys[key]))
        elif key == pygame.K_s:
            stop()

    return on_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--no-hat", action="store_true", help="don't contact the ESP32 (display only)")
    ap.add_argument("--dev", action="store_true", help="crash loudly on errors")
    ap.add_argument("--demo", action="store_true", help="sort everyone in people.json in turn, then a guest, forever")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, name="display")
    from hat import HatClient
    from show import ShowController

    people = People(cfg.resolve(cfg.paths.people), cfg.houses)
    display = MirrorDisplay(cfg, windowed=args.windowed)
    hat = HatClient(cfg, enabled=not args.no_hat)
    if args.no_hat:
        hat.cfg.fallback_audio = False
    show = ShowController(cfg, hat, display.post, display.set_caption)
    on_key = key_handler(people, cfg, lambda p: show.trigger(p, "keyboard", preempt=True), show.stop)
    display.status_line = lambda: f"hat={'on' if hat.online else 'off' if hat.online is False else '?'}"
    log.info("keys: 1-9/0 = person id, SPACE = random, F1-F4 = guest house, S = stop, ESC = quit")
    if args.demo:
        import threading

        def demo() -> None:
            cast = people.all() + [people.guest()]
            i = 0
            time.sleep(4)  # show a little idle first
            while True:
                if not show.busy:
                    time.sleep(3)  # idle gap between sortings
                    show.trigger(cast[i % len(cast)], "demo")
                    i += 1
                time.sleep(0.5)

        threading.Thread(target=demo, name="demo", daemon=True).start()
    display.run(on_key=on_key, dev=args.dev)


if __name__ == "__main__":
    sys.exit(main())
