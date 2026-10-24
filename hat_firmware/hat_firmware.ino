// Sorting Hat: audio + jaw on one ESP32.
//
// SD card root:
//   <id>.wav / <id>.env          per-person announcement (id = number from people.json)
//   house_<house>.wav / .env     generic announcements (guests, physical button)
//   mutter*.wav / .env           "thinking" clips, one picked at random
// WAVs are 16-bit mono 22050 Hz, envelopes one servo angle byte per 20 ms (make_envelope.py).
// Adding a person = copy two files + power-cycle. No reflash.
//
// HTTP:  /play?id=N   /mutter   /stop   /status   /jaw?deg=N (servo calibration)
//
// Threading: the web server runs in loop(); a higher-priority audio task owns the SD card, the
// I2S output and the servo. They talk through a FreeRTOS queue, so a slow HTTP client can't
// starve the audio, and audio + jaw are driven from one place off one sample position.
//
// Libraries: ESP8266Audio (Earle Philhower), ESP32Servo (Kevin Harrington). See README.

#include <Arduino.h>
#include <ESPmDNS.h>
#include <ESP32Servo.h>
#include <SD.h>
#include <SPI.h>
#include <WebServer.h>
#include <WiFi.h>

#include <vector>

#include "AudioFileSourceSD.h"
#include "AudioGeneratorWAV.h"
#include "AudioOutputI2S.h"
#include "config.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Copy secrets.h.example to secrets.h and set your Wi-Fi credentials"
#endif

// ======================================================================= shared state

enum CmdType : uint8_t { CMD_PLAY, CMD_MUTTER, CMD_STOP, CMD_BUTTON, CMD_JAW };
struct Cmd {
  CmdType type;
  char id[32];
  int arg;
};

// (types live up here: the Arduino IDE generates function prototypes above the first function)
struct WavInfo {
  uint32_t dataOffset = 44;
  uint32_t sampleRate = 22050;
  uint16_t blockAlign = 2;  // bytes per sample frame (all channels)
};

static QueueHandle_t cmdQueue;
static portMUX_TYPE statusMux = portMUX_INITIALIZER_UNLOCKED;
static char statusState[12] = "boot";  // idle | playing | muttering
static char statusClip[32] = "";
static uint32_t clipsPlayed = 0;

// Catalogue of clips found on the card at boot (read-only afterwards).
static std::vector<String> clipIds;
static std::vector<String> mutterIds;
static std::vector<String> houseIds;
static bool sdOk = false;

WebServer server(80);
Servo jaw;

static void setStatus(const char *state, const char *clip) {
  portENTER_CRITICAL(&statusMux);
  strlcpy(statusState, state, sizeof(statusState));
  strlcpy(statusClip, clip, sizeof(statusClip));
  portEXIT_CRITICAL(&statusMux);
}

static bool hasClip(const std::vector<String> &v, const String &id) {
  for (const auto &s : v)
    if (s == id) return true;
  return false;
}

// ======================================================================= jaw

static int lastAngle = -1;
static uint32_t lastJawMoveMs = 0;

static int clampAngle(int deg) {
  const int lo = min(JAW_MIN_DEG, JAW_MAX_DEG), hi = max(JAW_MIN_DEG, JAW_MAX_DEG);
  return constrain(deg, lo, hi);
}

static int travelToAngle(float frac) {
  return (int)lroundf(JAW_CLOSED_DEG + frac * (JAW_OPEN_DEG - JAW_CLOSED_DEG));
}

static void setJaw(int deg) {
  deg = clampAngle(deg);
  if (!jaw.attached()) {
    jaw.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    lastAngle = -1;
  }
  if (deg != lastAngle) {
    jaw.write(deg);
    lastAngle = deg;
  }
  lastJawMoveMs = millis();
}

// ======================================================================= playback (audio task only)

// Walk the RIFF chunks to find where samples start; don't assume a 44-byte header.
static bool readWavInfo(const String &path, WavInfo &info) {
  File f = SD.open(path);
  if (!f) return false;
  uint8_t hdr[12];
  if (f.read(hdr, 12) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
    f.close();
    return false;
  }
  bool gotFmt = false, gotData = false;
  while (f.available() >= 8) {
    uint8_t ch[8];
    if (f.read(ch, 8) != 8) break;
    uint32_t sz = ch[4] | (ch[5] << 8) | (ch[6] << 16) | ((uint32_t)ch[7] << 24);
    if (!memcmp(ch, "fmt ", 4) && sz >= 16) {
      uint8_t fmt[16];
      f.read(fmt, 16);
      info.sampleRate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
      info.blockAlign = fmt[12] | (fmt[13] << 8);
      f.seek(f.position() + sz - 16 + (sz & 1));
      gotFmt = true;
    } else if (!memcmp(ch, "data", 4)) {
      info.dataOffset = f.position();
      gotData = true;
      break;
    } else {
      f.seek(f.position() + sz + (sz & 1));
    }
  }
  f.close();
  return gotFmt && gotData && info.sampleRate > 0 && info.blockAlign > 0;
}

static const int WAV_READAHEAD_BYTES = 128;  // AudioGeneratorWAV's internal read buffer
static AudioOutputI2S *out = nullptr;
static AudioGeneratorWAV *wav = nullptr;
static AudioFileSourceSD *src = nullptr;
static WavInfo curInfo;
static uint8_t env[MAX_ENV_FRAMES];
static size_t envLen = 0;
static uint32_t silentUntil = 0;       // silent procedural mutter (no clip on card)
static uint32_t nextProceduralMs = 0;  // next random jaw target when there's no envelope
static String pendingClip;             // played when the current clip ends (button sequence)

static void stopClip() {
  if (wav) {
    if (wav->isRunning()) wav->stop();
    delete wav;
    wav = nullptr;
  }
  if (src) {
    delete src;
    src = nullptr;
  }
  envLen = 0;
  silentUntil = 0;
}

static bool startClip(const String &id, bool isMutter) {
  stopClip();
  const String wavPath = "/" + id + ".wav";
  if (!readWavInfo(wavPath, curInfo)) {
    Serial.printf("[audio] %s missing or not a PCM WAV\n", wavPath.c_str());
    setStatus("idle", "");
    return false;
  }
  envLen = 0;
  File e = SD.open("/" + id + ".env");
  if (e) {
    envLen = e.read(env, MAX_ENV_FRAMES);
    e.close();
  } else {
    Serial.printf("[audio] no %s.env: using random jaw\n", id.c_str());
  }
  src = new AudioFileSourceSD(wavPath.c_str());
  wav = new AudioGeneratorWAV();
  if (!wav->begin(src, out)) {
    Serial.printf("[audio] failed to start %s\n", wavPath.c_str());
    stopClip();
    setStatus("idle", "");
    return false;
  }
  clipsPlayed++;
  setStatus(isMutter ? "muttering" : "playing", id.c_str());
  Serial.printf("[audio] playing %s (%lu Hz, %u env frames)\n", id.c_str(), (unsigned long)curInfo.sampleRate, (unsigned)envLen);
  return true;
}

static void startMutter() {
  if (!mutterIds.empty() && startClip(mutterIds[random(mutterIds.size())], true)) return;
  stopClip();
  silentUntil = millis() + SILENT_MUTTER_MS;
  setStatus("muttering", "");
}

// Random jaw movement for clips without an envelope (and the silent mutter).
static void proceduralJaw() {
  if ((int32_t)(millis() - nextProceduralMs) < 0) return;
  nextProceduralMs = millis() + random(70, 170);
  setJaw(travelToAngle(random(0, 2) ? random(15, 55) / 100.0f : 0.0f));
}

// The heart of sync: jaw frame derived from the same sample position the audio is playing from.
static void updateJawFromAudio() {
  if (!envLen) {
    proceduralJaw();
    return;
  }
  const int64_t bytesRead = (int64_t)src->getPos() - curInfo.dataOffset;
  // samples read from the file, minus what's still queued in the generator and the DMA buffers
  const int64_t queued = DMA_BUFFERS * DMA_BUFFER_BYTES / 4 + WAV_READAHEAD_BYTES / curInfo.blockAlign +
                         (int64_t)curInfo.sampleRate * AUDIO_SYNC_TRIM_MS / 1000;
  const int64_t samples = bytesRead / curInfo.blockAlign - queued;
  if (samples < 0) return;
  size_t frame = (size_t)(samples * 1000 / ((int64_t)curInfo.sampleRate * ENV_FRAME_MS));
  if (frame >= envLen) frame = envLen - 1;
  setJaw(env[frame]);
}

// ======================================================================= idle behaviour

static uint32_t nextTwitchMs = 0;
static uint32_t twitchReturnMs = 0;

static void idleTick() {
  const uint32_t now = millis();
  if (twitchReturnMs && (int32_t)(now - twitchReturnMs) >= 0) {
    setJaw(JAW_CLOSED_DEG);
    twitchReturnMs = 0;
  }
  if ((int32_t)(now - nextTwitchMs) >= 0) {
    setJaw(travelToAngle(IDLE_TWITCH_TRAVEL * random(30, 101) / 100.0f));
    twitchReturnMs = now + IDLE_TWITCH_HOLD_MS;
    nextTwitchMs = now + random(IDLE_TWITCH_MIN_MS, IDLE_TWITCH_MAX_MS);
  }
  if (SERVO_DETACH_IDLE_MS > 0 && jaw.attached() && !twitchReturnMs && now - lastJawMoveMs > SERVO_DETACH_IDLE_MS) {
    jaw.detach();
  }
}

static void finishedClip() {
  stopClip();
  setJaw(JAW_CLOSED_DEG);
  nextTwitchMs = millis() + random(IDLE_TWITCH_MIN_MS, IDLE_TWITCH_MAX_MS);
  if (pendingClip.length()) {
    String next = pendingClip;
    pendingClip = "";
    if (startClip(next, false)) return;
  }
  setStatus("idle", "");
}

static void handleCmd(const Cmd &c) {
  switch (c.type) {
    case CMD_PLAY:
      pendingClip = "";
      if (!startClip(String(c.id), false)) finishedClip();
      break;
    case CMD_MUTTER:
      pendingClip = "";
      startMutter();
      break;
    case CMD_STOP:
      pendingClip = "";
      finishedClip();
      break;
    case CMD_BUTTON:  // manual fallback: mutter, then a random house
      if (houseIds.empty()) {
        Serial.println("[button] no house_*.wav on the card");
        startMutter();
        break;
      }
      pendingClip = houseIds[random(houseIds.size())];
      Serial.printf("[button] sorting a guest into %s\n", pendingClip.c_str());
      startMutter();
      break;
    case CMD_JAW:
      stopClip();
      setStatus("idle", "");
      setJaw(c.arg);
      nextTwitchMs = millis() + 15000;  // hold still while you're calibrating
      break;
  }
}

static void audioTask(void *) {
  for (;;) {
    Cmd c;
    const bool active = wav || silentUntil;
    // Block on the queue when idle; when playing just peek so the DMA buffers keep filling.
    if (xQueueReceive(cmdQueue, &c, active ? 0 : pdMS_TO_TICKS(10)) == pdTRUE) handleCmd(c);

    if (wav) {
      if (wav->isRunning() && wav->loop()) {
        updateJawFromAudio();
      } else {
        finishedClip();  // starts pendingClip if the button queued one
      }
    } else if (silentUntil) {
      if ((int32_t)(millis() - silentUntil) < 0) {
        proceduralJaw();
      } else {
        silentUntil = 0;
        finishedClip();
      }
    } else {
      idleTick();
    }
    vTaskDelay(1);  // let loop() (web server) run; DMA buffers cover ~45 ms
  }
}

// ======================================================================= web server (loop task)

static bool enqueue(const Cmd &c) { return xQueueSend(cmdQueue, &c, 0) == pdTRUE; }

static bool validId(const String &id) {
  if (id.length() == 0 || id.length() >= sizeof(Cmd::id)) return false;
  for (char ch : id)
    if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-') return false;
  return true;
}

static void handlePlay() {
  const String id = server.arg("id");
  if (!validId(id)) return server.send(400, "text/plain", "bad id\n");
  if (!hasClip(clipIds, id)) return server.send(404, "text/plain", "no /" + id + ".wav on SD\n");
  Cmd c{CMD_PLAY, {0}, 0};
  strlcpy(c.id, id.c_str(), sizeof(c.id));
  if (!enqueue(c)) return server.send(503, "text/plain", "busy\n");
  server.send(200, "text/plain", "OK " + id + "\n");
}

static void handleSimple(CmdType t) {
  Cmd c{t, {0}, 0};
  if (!enqueue(c)) return server.send(503, "text/plain", "busy\n");
  server.send(200, "text/plain", "OK\n");
}

static void handleJaw() {
  if (!server.hasArg("deg")) return server.send(400, "text/plain", "need ?deg=\n");
  const int requested = server.arg("deg").toInt();
  Cmd c{CMD_JAW, {0}, requested};
  if (!enqueue(c)) return server.send(503, "text/plain", "busy\n");
  server.send(200, "text/plain", "OK jaw " + String(clampAngle(requested)) +
                                     (clampAngle(requested) != requested ? " (clamped by JAW_MIN/MAX_DEG)\n" : "\n"));
}

static String jsonList(const std::vector<String> &v) {
  String s = "[";
  for (size_t i = 0; i < v.size(); i++) s += (i ? ",\"" : "\"") + v[i] + "\"";
  return s + "]";
}

static void handleStatus() {
  char state[12], clip[32];
  portENTER_CRITICAL(&statusMux);
  strlcpy(state, statusState, sizeof(state));
  strlcpy(clip, statusClip, sizeof(clip));
  portEXIT_CRITICAL(&statusMux);
  String j = "{";
  j += "\"state\":\"" + String(state) + "\",";
  j += "\"clip\":\"" + String(clip) + "\",";
  j += "\"sd\":" + String(sdOk ? "true" : "false") + ",";
  j += "\"clips\":" + jsonList(clipIds) + ",";
  j += "\"mutters\":" + String(mutterIds.size()) + ",";
  j += "\"houses\":" + jsonList(houseIds) + ",";
  j += "\"played\":" + String(clipsPlayed) + ",";
  j += "\"uptime_s\":" + String(millis() / 1000) + ",";
  j += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  j += "\"heap\":" + String(ESP.getFreeHeap()) + ",";
  j += "\"jaw\":{\"closed\":" + String(JAW_CLOSED_DEG) + ",\"open\":" + String(JAW_OPEN_DEG) +
       ",\"min\":" + String(JAW_MIN_DEG) + ",\"max\":" + String(JAW_MAX_DEG) + "}";
  j += "}";
  server.send(200, "application/json", j);
}

// ======================================================================= setup / loop

static void scanCard() {
  File root = SD.open("/");
  if (!root) return;
  for (File f = root.openNextFile(); f; f = root.openNextFile()) {
    String name = f.name();
    f.close();
    if (name.startsWith("/")) name = name.substring(1);  // older cores return a leading slash
    if (name.startsWith(".") || !name.endsWith(".wav")) continue;
    const String id = name.substring(0, name.length() - 4);
    if (!validId(id)) {
      Serial.printf("[sd] skipping %s: ids may only use letters, digits, _ and -\n", name.c_str());
      continue;
    }
    clipIds.push_back(id);
    if (id.startsWith("mutter")) mutterIds.push_back(id);
    if (id.startsWith("house_")) houseIds.push_back(id);
  }
  root.close();
  Serial.printf("[sd] %u clips, %u mutters, %u house clips\n", clipIds.size(), mutterIds.size(), houseIds.size());
}

static bool mdnsStarted = false;

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
#if USE_STATIC_IP
  WiFi.config(IPAddress(STATIC_IP), IPAddress(STATIC_GATEWAY), IPAddress(STATIC_SUBNET), IPAddress(STATIC_GATEWAY));
#endif
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);  // modem sleep adds 100+ ms latency: the Pi only waits 1.5 s for an ACK
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  const uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[wifi] %s  http://%s.local/\n", WiFi.localIP().toString().c_str(), HOSTNAME);
  } else {
    Serial.println("[wifi] not connected yet; button still works, will keep retrying");
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n== Sorting Hat ==");
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  randomSeed(esp_random());

  jaw.setPeriodHertz(50);
  setJaw(JAW_CLOSED_DEG);

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  sdOk = SD.begin(SD_CS, SPI, SD_SPI_HZ);
  if (sdOk) {
    scanCard();
  } else {
    Serial.println("[sd] MOUNT FAILED: check wiring/format (FAT32). Running with no clips.");
  }

  out = new AudioOutputI2S();
  out->SetBuffers(DMA_BUFFERS, DMA_BUFFER_BYTES);  // known depth, so the jaw sync can subtract it
  out->SetPinout(I2S_BCLK, I2S_LRC, I2S_DOUT);
  out->SetGain(VOLUME);

  cmdQueue = xQueueCreate(4, sizeof(Cmd));
  setStatus("idle", "");
  // Same core as loop() but higher priority: audio preempts the web server, never the reverse.
  xTaskCreatePinnedToCore(audioTask, "audio", 8192, nullptr, 3, nullptr, 1);

  connectWifi();
  server.on("/", [] {
    server.send(200, "text/plain", "Sorting Hat. /play?id=N /mutter /stop /status /jaw?deg=N\n");
  });
  server.on("/play", handlePlay);
  server.on("/mutter", [] { handleSimple(CMD_MUTTER); });
  server.on("/stop", [] { handleSimple(CMD_STOP); });
  server.on("/status", handleStatus);
  server.on("/jaw", handleJaw);
  server.onNotFound([] { server.send(404, "text/plain", "not found\n"); });
  server.begin();
}

static void pollButton() {
  static bool lastRaw = HIGH, stable = HIGH;
  static uint32_t changedMs = 0, lastFireMs = 0;
  const bool raw = digitalRead(BUTTON_PIN);
  const uint32_t now = millis();
  if (raw != lastRaw) {
    lastRaw = raw;
    changedMs = now;
  }
  if (now - changedMs >= BUTTON_DEBOUNCE_MS && raw != stable) {
    stable = raw;
    if (stable == LOW && now - lastFireMs > BUTTON_LOCKOUT_MS) {
      lastFireMs = now;
      Cmd c{CMD_BUTTON, {0}, 0};
      enqueue(c);
    }
  }
}

void loop() {
  server.handleClient();
  pollButton();
  const bool connected = WiFi.status() == WL_CONNECTED;
  digitalWrite(LED_PIN, connected ? HIGH : LOW);
  if (connected && !mdnsStarted) {
    mdnsStarted = MDNS.begin(HOSTNAME);
    if (mdnsStarted) MDNS.addService("http", "tcp", 80);
  }
  delay(2);
}
