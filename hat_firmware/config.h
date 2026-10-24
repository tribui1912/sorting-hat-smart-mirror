// Hat firmware tunables. Wi-Fi credentials live in secrets.h (copy secrets.h.example).
#pragma once

// ----------------------------------------------------------------- network
#define HOSTNAME "sortinghat"  // -> http://sortinghat.local/ (mDNS)
// Optional static IP: more reliable than mDNS at a party. Or set a DHCP reservation on the router.
#define USE_STATIC_IP 0
#define STATIC_IP 192, 168, 1, 50
#define STATIC_GATEWAY 192, 168, 1, 1
#define STATIC_SUBNET 255, 255, 255, 0

// ----------------------------------------------------------------- pins (classic ESP32 DevKit)
// MAX98357A
#define I2S_BCLK 26
#define I2S_LRC 25
#define I2S_DOUT 22
// microSD on VSPI
#define SD_SCK 18
#define SD_MISO 19
#define SD_MOSI 23
#define SD_CS 5
#define SD_SPI_HZ 10000000  // 22 kHz mono needs ~44 kB/s; drop to 4000000 if cards fail to mount
// Servo signal. Avoid strapping pins 0, 2, 12, 15.
#define SERVO_PIN 13
// Manual fallback button: momentary, between this pin and GND (internal pull-up)
#define BUTTON_PIN 32
#define LED_PIN 2  // on-board LED: on = Wi-Fi connected

// ----------------------------------------------------------------- jaw servo
// Find these with `python mirror/hat.py jaw <deg>` (the /jaw endpoint), then copy closed/open
// into config.yaml's envelope section and re-run make_envelope.py.
#define JAW_CLOSED_DEG 90  // jaw shut, resting
#define JAW_OPEN_DEG 130   // jaw fully open (may be numerically lower than closed: direction doesn't matter)
// Hard software stops. Every angle the servo is sent is clamped to this range, including envelope
// values from the SD card, so a bad .env can't drive the linkage into its mechanical stops.
#define JAW_MIN_DEG 85
#define JAW_MAX_DEG 135
#define SERVO_MIN_US 500   // MG90S pulse range
#define SERVO_MAX_US 2400
// Detach the servo after this long idle (stops hum/jitter). 0 = always hold position.
// Idle twitches re-attach it briefly.
#define SERVO_DETACH_IDLE_MS 0

// ----------------------------------------------------------------- audio / sync
#define VOLUME 0.8f  // 0.0-4.0 software gain. >1 clips normalised clips; use the amp's GAIN pin instead.
// I2S DMA buffering. The envelope is indexed off the sample position read from the WAV file, and
// those samples reach the speaker one full DMA queue later; the firmware subtracts that automatically.
// 8 x 512 bytes = 1024 stereo frames = ~46 ms at 22050 Hz. (+) more robust to Wi-Fi hiccups
// (crackles = raise it), (-) tighter. Library default is ~130 ms.
#define DMA_BUFFERS 8
#define DMA_BUFFER_BYTES 512
// Manual sync trim on top of that. (+) jaw moves later relative to sound, (-) earlier.
// make_envelope.py already leads the jaw by ~40 ms; only touch this if sync looks off by eye.
#define AUDIO_SYNC_TRIM_MS 0
#define ENV_FRAME_MS 20         // must match envelope.frame_ms in config.yaml
#define MAX_ENV_FRAMES 3000     // 60 s at 50 Hz

// ----------------------------------------------------------------- behaviour
// Idle: tiny random jaw movements so the hat looks alive. Also keeps a USB power bank from
// auto-shutting off on low current draw.
#define IDLE_TWITCH_MIN_MS 2500
#define IDLE_TWITCH_MAX_MS 7000
#define IDLE_TWITCH_TRAVEL 0.12f  // fraction of closed->open travel. (+) more visible twitch
#define IDLE_TWITCH_HOLD_MS 180
// Mutter with no mutter*.wav on the card: random jaw movement for this long, silent.
#define SILENT_MUTTER_MS 3000
#define BUTTON_DEBOUNCE_MS 40
#define BUTTON_LOCKOUT_MS 2000  // ignore re-presses while a button sorting is starting
