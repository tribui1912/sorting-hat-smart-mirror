# Sorting Hat + Smart Mirror

The mirror recognises a friend, the hat mutters, then announces their house with a moving jaw while the mirror shows the crest.

```
Pi 5: camera → detect → track → embed → match → vote ──HTTP GET──▶ ESP32 (hat)
                                                │                   play <id>.wav + step jaw from <id>.env
                                                ▼
                                          mirror display
Phone: http://<pi>:8000/  ──▶ force any reveal (bypasses recognition)
```

## Layout

```
config.yaml              every tunable, commented with which way to move it
dialogue.yaml            the hat's script: 8 openings, 5 suspense lines, 7 announcements per house
people.json              name -> {id, house}; written by enroll.py (the sample entries are placeholders)
mirror/
  main.py                the party app: display + recognition + override page, one process
  display.py             pygame mirror UI. Standalone = the "dumb" keyboard-driven show
  show.py                sequence: thinking (dialogue) -> reveal -> idle
  hat.py                 ESP32 client + fallback audio through the Pi. CLI for testing
  server.py              override page backend;  override.html  the phone page
  recognize.py           recognition service. Standalone --preview for tuning
  enroll.py              capture from the mirror camera / import photos / quality report
  calibrate.py           sweeps threshold & margin, writes them into config.yaml
  make_envelope.py       voice clips -> ESP32 WAV + jaw envelope
  camera.py faces.py tracker.py common.py
hat_firmware/            Arduino sketch (hat_firmware.ino, config.h, secrets.h.example)
audio/source/            your recordings go here;  audio/sd/  generated files = SD card contents
assets/crests/<house>.png  crest art: transparent PNG (also .webp/.jpg), e.g. gryffindor.png.
                         Missing = a shield is drawn in the house colours. The log says which each house uses.
assets/fonts/            optional font; set display.font
deploy/sortinghat.service  autostart on the Pi
```

Every Python script takes `--config path` and runs standalone. Run them from the project root.

## Build order

Get the dumb chain working end to end before touching recognition. If enrollment or lighting goes wrong the week of the party, steps 1–4 are a complete show run from your phone.

### 1. Pi setup

```bash
./setup_pi.sh
```

Installs apt deps (picamera2 comes from apt), makes `.venv` with `--system-site-packages`, pip installs the rest, and downloads the face models while there's internet. Then:

- **Rotate the screen to portrait** in Pi OS (Screen Configuration → Orientation). If you can't, set `display.rotate`.
- **Fit the active cooler.** Sustained ONNX inference will throttle a bare Pi 5.
- Set `display.focus_y` to where the camera sits (fraction of screen height from the top).

### 2. Mirror alone

Replace the placeholder `people.json` with your real guest list first: `{"Alice": {"id": 1, "house": "gryffindor"}, ...}`. The ids name the SD clips (`1.wav`), and enrollment keeps an existing person's id.

```bash
.venv/bin/python mirror/display.py --windowed --no-hat
```

Keys `1`–`9`/`0` sort person id 1–10, `SPACE` random, `F1`–`F4` guest into a house, `S` stop, `D` debug overlay, `ESC` quit.

### 3. Hat firmware

Arduino IDE or arduino-cli, ESP32 core **3.x**, libraries **ESP8266Audio ≥ 2.0** and **ESP32Servo ≥ 3.0**. (Verified to compile with esp32 3.3.12, ESP8266Audio 2.4.1, ESP32Servo 3.2.1. ESP8266Audio 1.x won't work, it lacks `SetBuffers`.)

```bash
cp hat_firmware/secrets.h.example hat_firmware/secrets.h   # add your Wi-Fi
arduino-cli lib install ESP8266Audio ESP32Servo
arduino-cli compile --fqbn esp32:esp32:esp32 hat_firmware
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 hat_firmware
```

**Wiring** (pins are in `config.h`):

| Part | Pin | ESP32 |
|---|---|---|
| MAX98357A | BCLK / LRC / DIN | 26 / 25 / 22 |
| | VIN / GND | 5V / GND. Leave SD unconnected (mono L+R mix), GAIN unconnected (9 dB) |
| microSD (SPI) | CS / SCK / MISO / MOSI | 5 / 18 / 19 / 23 |
| MG90S | signal | 13. Power it from the 5V bank rail, not the ESP32's 3V3, with the 1000µF cap across it. Common ground. |
| Button | one leg / other leg | 32 / GND (internal pull-up) |

**Find the jaw limits**, with the linkage attached and the hat on Wi-Fi:

```bash
.venv/bin/python mirror/hat.py jaw 90     # nudge until the jaw is just shut
.venv/bin/python mirror/hat.py jaw 125    # ...and fully open without straining
```

Put those in `config.h` (`JAW_CLOSED_DEG`, `JAW_OPEN_DEG`, with `JAW_MIN/MAX_DEG` a few degrees outside them as hard stops) **and** in `config.yaml` `envelope.closed_deg/open_deg`. Then reflash.

Give the ESP32 a DHCP reservation and put its IP in `hat.host`. mDNS works, but an IP never has a lookup stall.

### 4. Audio

Record clips into `audio/source/`, any format ffmpeg reads:

| File | Used for |
|---|---|
| `Alice.m4a` (a name from people.json) | Alice's personal announcement. Written out as `<her id>.wav/.env`. Takes precedence over the dialogue script |
| `mutter1.wav`, `mutter2.wav`, … | thinking phase for personal clips, picked at random |
| `dlg_open_01.wav` … `dlg_hufflepuff_07.wav` | the dialogue script's lines, named by the ids in `dialogue.yaml` |
| `dlg_open_01__alice.wav` | optional per-person variant of a personalized line; used for Alice instead of the generic `dlg_open_01.wav` |
| `sfx_reveal.wav` | optional sting played as the crest appears |

```bash
.venv/bin/python mirror/make_envelope.py audio/source/ --preview
```

Copy everything in `audio/sd/` to the root of a FAT32 microSD and power-cycle the hat. To add a person later: record, re-run, copy two files, power-cycle. No reflash.

Set `show.reveal_delay_s` to when the house name is spoken in a personal clip (or set `reveal_delay_s` per person in people.json) so the crest lands on the word. Dialogue clips don't need this: they're synced with `show.reveal_lead_s`.

**Dialogue before voices:** with no `dlg_*.wav` generated yet, the show still runs the full sequence — opening, optional suspense, announcement — on the script's timings, with each line shown as a caption on the mirror (`show.captions: false` hides them). Generate the voices whenever, name the files by their ids, re-run `make_envelope.py`, and the same sequence switches to real audio automatically; the crest then lands `show.reveal_lead_s` before the clip ends, right on the shouted house name.

**Test the chain:**

```bash
.venv/bin/python mirror/hat.py status
.venv/bin/python mirror/hat.py play 1
.venv/bin/python mirror/display.py --windowed       # keypress -> mirror + hat
.venv/bin/python mirror/main.py --no-recognition    # plus the phone page at http://<pi>.local:8000/
```

That's the fallback show, complete. Everything from here on is the automatic trigger.

### 5. Enrollment

Enroll on the mirror itself, standing on the mark, under the party lighting. The angle, lens and light have to match what the camera sees at the party.

```bash
.venv/bin/python mirror/recognize.py --preview     # first: check faces at the mark are > min_face_px
.venv/bin/python mirror/enroll.py capture --name Alice --house gryffindor
```

`SPACE` captures, `BACKSPACE` undoes, `ENTER` saves. Take 6–10 shots: looking at the lens, slightly left/right/up/down, neutral and smiling, glasses on if they'll wear them. Each capture prints its similarity to the earlier shots, so you can undo outliers on the spot. Over SSH, use `--headless`.

No access to someone before the party? `enroll.py folder photos/bob --name Bob --house hufflepuff` works, just less well. Anyone not enrolled can still be sorted from the phone.

`enroll.py list` prints the quality report: high-variance captures, outlier shots, lookalike pairs.

### 6. Calibration

```bash
.venv/bin/python mirror/enroll.py snap data/strangers     # non-attendees on the mirror camera
.venv/bin/python mirror/enroll.py snap data/genuine/Alice # optional: held-out shots, another day
.venv/bin/python mirror/calibrate.py --strangers data/strangers --genuine data/genuine
```

You can top up `data/strangers` with any public face photos. A few hundred faces gives usable numbers. Calibration reports per-pair FAR, the per-stranger FAR (a stranger gets one chance per enrolled person, ~20×), FRR and misidentification. It recommends a threshold in the middle of the safe window plus a margin, then writes both into `config.yaml` (with a `.bak`). Use `--dry-run` to only report.

### 7. Party mode

```bash
mkdir -p ~/.config/systemd/user && cp deploy/sortinghat.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now sortinghat
```

Runs `main.py` fullscreen, restarts on exit, logs to `logs/sortinghat.log`. A thread that crashes (camera, recognition, server) is logged and restarted, and the display keeps running. Use `--dev` while building to make any crash fatal and loud.

## How a sorting runs

1. Recognition commits (or the phone/keyboard triggers): the mirror goes to **thinking**, and the show draws a script from `dialogue.yaml` — a random opening (never the same one twice in a row) personalized with the guest's name from people.json, then a suspense line on ~25% of sortings (`show.suspense_prob`), then an announcement from their house's pool. The pool only reshuffles once all 7 lines of that house have been used. A guest's house always comes from people.json — recognition only picks the name, and a re-sort keeps the same house. Guests sorted without a name (keyboard F1–F4, generic button) get the generic opening/suspense lines. Unrecognized faces are not guessed at: recognition stays quiet and the operator triggers the right person from the phone.
2. Each line plays as `GET /play?id=<clip id>`. With no ACK within `hat.timeout_s` (1.5 s), the clip plays from `audio/sd/` through the Pi's audio output instead. The Pi 5 has no headphone jack, so that's HDMI audio or a USB speaker. Lines without a clip yet hold for the script's caption timing.
3. `show.reveal_lead_s` before the announcement clip ends — as the house name is shouted — the crest appears, optionally followed by `sfx_reveal`, and holds for `show.reveal_s`.

People with a personal clip (`audio/sd/<id>.wav`) keep the classic single-clip flow: mutter, announcement, crest at `reveal_delay_s`.

Recognition triggers are refused while a sorting is running. Phone and keyboard triggers preempt it, so if recognition picks the wrong person you just tap the right one.

## Tuning cheat-sheet

| Symptom | Change |
|---|---|
| Friends not recognised | `calibrate.py` with a `--genuine` set. Re-enroll under party light. Lower `threshold` a little. Lower `vote_min` |
| Strangers or wrong person sorted | Raise `threshold` / `margin`. Check `enroll.py list` for lookalikes and bad shots |
| Triggers on people walking past | Raise `min_face_px`, `stable_frames` |
| Slow to trigger | Lower `vote_window`/`vote_min`. Try `model: buffalo_s` (then `enroll.py rebuild`) |
| Jaw chatters | Raise `envelope.smooth_frames`, `noise_floor` |
| Jaw hangs open | Raise `noise_floor` |
| Jaw late/early vs voice | `envelope.lead_ms` (re-run make_envelope), or `AUDIO_SYNC_TRIM_MS` in firmware |
| Audio crackles on the hat | Raise `DMA_BUFFERS` (sync compensates automatically) |
| Crest appears before/after the house name | `show.reveal_delay_s` or per-person `reveal_delay_s` |

## Party-day checklist

- [ ] Override page bookmarked on your phone, phone on the same Wi-Fi (`http://<pi>.local:8000/`)
- [ ] `hat.py status` answers. Power bank charged, plus a spare
- [ ] `hat.host` is an IP, not `.local`
- [ ] Hat button tested: it sorts a random house with no Wi-Fi at all
- [ ] `recognize.py --preview` under the final lighting: faces at the mark are well above `min_face_px`
- [ ] Turn recognition off from the phone if it misbehaves. The show carries on manually and nobody can tell
