/*
 * Sorting Hat — WiFi VOICE STREAMER (XIAO ESP32-C5)
 * -----------------------------------------------------------------
 * Joins WiFi (secrets.h) and listens on TCP 5005. The Jetson/PC side
 * (jetson/send_audio.py) streams 16-bit mono 22050 Hz PCM; the hat
 * plays it through the MAX98357A while the mouth motor flaps with the
 * voice loudness.
 *
 * Protocol (commands are newline-terminated ASCII):
 *   PING                              -> PONG reset=<why last boot> up=<s> boots=<n>
 *                                        (reset=BROWNOUT => the 5V rail is sagging)
 *   CLIP                              -> plays the embedded bench clip, then DONE
 *   PLAY <n>  followed by n PCM bytes -> OK, streams to speaker, then DONE
 * A client that disconnects mid-PLAY aborts playback cleanly.
 *
 * WiFi: copy secrets.h.example -> secrets.h and fill in. The IP is
 * printed on the Serial Monitor when it joins — pass it to
 * send_audio.py --host.
 *
 * Pins: I2S BCLK=D3 LRC=D4 DIN=D5 ; motor SIG=D6 ; button=D7.
 * Board: XIAO_ESP32C5, USB CDC On Boot enabled.
 * -----------------------------------------------------------------
 */

#include <ESP_I2S.h>
#include <WiFi.h>
#include <esp_system.h>
#include "secrets.h"          // WIFI_SSID / WIFI_PASSWORD
#include "audio/test_line.h"  // bench clip (CLIP command / D7 button)

const int I2S_BCLK = D3, I2S_LRC = D4, I2S_DOUT = D5;
const int PIN_MOTOR = D6, PIN_BTN = D7;
const int PIN_LED = LED_BUILTIN;
const int LED_ON = LOW, LED_OFF = HIGH;

const uint32_t SAMPLE_RATE = 22050;  // must match send_audio.py conversion
const int PWM_FREQ = 20000, PWM_RES = 8;
const uint16_t TCP_PORT = 5005;

const int MOTOR_DUTY = 240;        // 240 ≈ 4.2V — proven OEM value; 215 stalls under load
// Each flap starts the motor from rest, pulling near-stall current off the shared
// 5V rail. A short hardware fade 0 -> MOTOR_DUTY softens that spike so the ESP32
// doesn't brown out. 0 = hard switch-on (old behaviour). Keep < MOUTH_HOLD * 12 ms.
const int MOTOR_RAMP_MS = 15;
const float VOLUME = 1.0f;         // 1.0 = as recorded (full scale); 0.5 ≈ -6 dB; amp gain set by GAIN pin
// Mouth follows the voice: motor ON while the clip is loud, OFF in pauses.
const int   MOUTH_GATE  = 1200;   // loudness floor (of 32767) below which mouth stops
const float MOUTH_RELAX = 0.70f;  // envelope release per chunk (~12 ms)
const int   MOUTH_HOLD  = 3;      // minimum ON time in chunks (~35 ms) per flap

const size_t CH = 512;            // PCM bytes per chunk (~12 ms of audio)

I2SClass I2S;
WiFiServer server(TCP_PORT);

// Why did we last boot? Off USB there's no Serial Monitor, so PING reports it.
RTC_NOINIT_ATTR uint32_t bootCount;   // survives brownout/panic resets, not power-on
const char* resetName(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:  return "POWERON";
    case ESP_RST_BROWNOUT: return "BROWNOUT";
    case ESP_RST_PANIC:    return "PANIC";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT:      return "WATCHDOG";
    case ESP_RST_SW:       return "SOFTWARE";
    case ESP_RST_EXT:      return "EXTERNAL";
    default:               return "OTHER";
  }
}
esp_reset_reason_t bootReason;

// ---- shared playback path: mouth tracking + volume, used by CLIP and PLAY
int16_t outBuf[CH / 2];
static int16_t inS[CH / 2];       // network staging buffer (int16-aligned)
float env = 0.0f;
int onFor = 0;
bool motorOn = false;
uint32_t dbgT = 0;

void mouthStart() { env = 0.0f; onFor = 0; motorOn = false; }
void mouthStop()  { ledcWrite(PIN_MOTOR, 0); motorOn = false; }

void setMotor(bool on) {
  if (on == motorOn) return;           // only act on edges so the fade can finish
  motorOn = on;
  if (on && MOTOR_RAMP_MS > 0) ledcFade(PIN_MOTOR, 0, MOTOR_DUTY, MOTOR_RAMP_MS);
  else ledcWrite(PIN_MOTOR, on ? MOTOR_DUTY : 0);
}

void emitChunk(const int16_t* s, size_t samples) {
  int32_t peak = 0;
  for (size_t i = 0; i < samples; i++) {
    int32_t o = s[i];
    int32_t a = o < 0 ? -o : o;
    if (a > peak) peak = a;            // envelope uses pre-volume level,
    outBuf[i] = (int16_t)(o * VOLUME); // so MOUTH_GATE is volume-independent
  }
  env = (float)peak > env ? (float)peak : env * MOUTH_RELAX;

  // motor stalls below full duty, so drive it binary: loud = flap, pause = rest
  if (env > MOUTH_GATE) onFor = MOUTH_HOLD;
  else if (onFor > 0) onFor--;
  setMotor(onFor > 0);

  if (millis() - dbgT > 400) {         // ~2-3 debug lines/s while playing
    dbgT = millis();
    Serial.printf("env %5d  duty %d\n", (int)env, onFor > 0 ? MOTOR_DUTY : 0);
  }
  I2S.write((uint8_t*)outBuf, samples * 2);   // blocks via DMA = paces playback
}

void playClip() {                      // embedded bench clip
  Serial.println("play embedded clip");
  digitalWrite(PIN_LED, LED_ON);
  mouthStart();
  const size_t LEN = test_line_len * sizeof(test_line_pcm[0]);
  for (size_t off = 0; off < LEN; off += CH)
    emitChunk(test_line_pcm + off / 2, min(CH, LEN - off) / 2);
  mouthStop();
  digitalWrite(PIN_LED, LED_OFF);
}

// Stream `total` PCM bytes from the TCP client to the speaker.
void playStream(WiFiClient& client, size_t total) {
  Serial.printf("stream %u bytes (%.1f s)\n", (unsigned)total, total / (SAMPLE_RATE * 2.0));
  digitalWrite(PIN_LED, LED_ON);
  mouthStart();
  size_t got = 0, fill = 0;
  while (got < total && client.connected()) {
    int r = client.read((uint8_t*)inS + fill, min(CH - fill, total - got));
    if (r > 0) {
      fill += r;
      got += r;
      if (fill == CH || got == total) {          // chunk full or stream ended
        emitChunk(inS, fill / 2);
        if (fill & 1) { inS[0] = (int16_t)((uint8_t*)inS)[fill - 1]; fill = 1; }
        else fill = 0;
      }
    } else if (r < 0) break;                     // socket error -> abort
    else delay(1);                               // no data yet
  }
  if (fill >= 2) emitChunk(inS, fill / 2);       // flush final partial chunk
  mouthStop();
  digitalWrite(PIN_LED, LED_OFF);
  Serial.println("playback done");
}

String readLine(WiFiClient& c) {
  String line;
  uint32_t idle = millis();
  while (millis() - idle < 3000) {               // 3 s idle timeout
    while (c.available()) {
      char ch = c.read();
      if (ch == '\n') return line;
      if (ch != '\r') line += ch;
      idle = millis();
    }
    if (!c.connected()) break;
    delay(1);
  }
  return line;
}

void handleClient(WiFiClient& client) {
  while (client.connected()) {
    String line = readLine(client);
    if (line.length() == 0) return;              // timeout / dead link
    if (line == "PING") {
      client.printf("PONG reset=%s up=%lus boots=%lu\n", resetName(bootReason),
                    millis() / 1000, (unsigned long)bootCount);
    } else if (line == "CLIP") {
      playClip();
      client.print("DONE\n");
    } else if (line.startsWith("PLAY ")) {
      size_t total = strtoul(line.c_str() + 5, nullptr, 10);
      if (total == 0 || total & 1) { client.print("ERR byte count must be even\n"); return; }
      client.print("OK\n");                      // sender pushes the n bytes now
      playStream(client, total);
      client.print("DONE\n");
      return;                                    // one PLAY per connection
    } else {
      client.print("ERR unknown command\n");
      return;
    }
  }
}

void setup() {
  // motor off FIRST: D6 floats through the MOSFET module's input until PWM
  // attaches, which spins the motor on direct-5V power during boot
  pinMode(PIN_MOTOR, OUTPUT);
  digitalWrite(PIN_MOTOR, LOW);

  Serial.begin(115200);
  delay(300);
  Serial.println("\nSORTING HAT — wifi voice streamer (XIAO ESP32-C5)");
  bootReason = esp_reset_reason();
  if (bootReason == ESP_RST_POWERON) bootCount = 0;   // RTC_NOINIT is garbage after power-on
  bootCount++;
  Serial.printf("reset reason: %s (boot #%lu)\n", resetName(bootReason), (unsigned long)bootCount);

  pinMode(PIN_LED, OUTPUT); digitalWrite(PIN_LED, LED_OFF);
  pinMode(PIN_BTN, INPUT_PULLUP);
  ledcAttach(PIN_MOTOR, PWM_FREQ, PWM_RES);
  mouthStop();

  // Silence or wrong pitch? change MONO -> STEREO here.
  I2S.setPins(I2S_BCLK, I2S_LRC, I2S_DOUT, -1, -1);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO))
    Serial.println("I2S begin FAILED");

  WiFi.mode(WIFI_STA);
  // Lower TX power cuts the radio's ~300+ mA transmit spikes, which on a shared
  // motor/amp 5V rail are enough to trip the brownout detector. 8.5 dBm is plenty
  // for a router in the same room; raise it if RSSI is poor.
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.setSleep(false);                          // no modem-sleep stalls mid-stream
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("joining wifi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(400); Serial.print(".");
    if (millis() - t0 > 20000) ESP.restart();    // stuck association -> start clean
  }
  Serial.println();
  Serial.print("HAT IP: "); Serial.println(WiFi.localIP());   // <- for send_audio.py
  server.begin();
  Serial.println("listening on TCP 5005 — press D7 for the bench clip");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {           // keep WiFi alive
    Serial.println("wifi lost — reconnecting");
    WiFi.reconnect();
    delay(500);
    return;
  }

  WiFiClient client = server.accept();
  if (client) {
    Serial.printf("client %s\n", client.remoteIP().toString().c_str());
    handleClient(client);
    client.stop();
    Serial.println("client gone");
  }

  // bench button still plays the embedded clip without the PC
  static bool btnWas = HIGH;
  bool btn = digitalRead(PIN_BTN);
  if (btnWas == HIGH && btn == LOW) playClip();
  btnWas = btn;
}
