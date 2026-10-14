/*
 * fan_monitor.ino - Stage 3/4 firmware for the self-baselining vibration monitor.
 *
 * ESP32-DevKitC + MPU-6050, 1 kHz interrupt-driven sampling.
 * On every 256-sample window (50% overlap) it computes the same 24 features
 * as fan_features.py, then scores the window with BOTH detectors:
 *     Mahalanobis   d^2  = || A (x - mu) / sigma ||^2
 *     Autoencoder   MSE  = mean((decode(encode(z)) - z)^2)
 * An alarm needs ALARM_CONSEC consecutive anomalous windows, which kills the
 * single-window false positives you would otherwise get from door slams.
 *
 * Wiring (unchanged from C_mpu_stream.ino):
 *     MPU VCC -> 3V3     MPU GND -> GND
 *     MPU SDA -> GPIO21  MPU SCL -> GPIO22   MPU INT -> GPIO19
 *     LED anode -> GPIO2 through a 220 ohm resistor -> GND
 *
 * Before flashing:
 *     1. run fan_train.py then export_header.py
 *     2. copy the generated baseline.h into this sketch folder
 *     3. fill in WIFI_SSID / WIFI_PASS / MQTT_HOST below
 *
 * Library Manager: PubSubClient by Nick O'Leary.
 * Set ENABLE_MQTT to 0 to run detection standalone over serial only.
 */

#include <Wire.h>
#include <math.h>
#include "baseline.h"

#define ENABLE_MQTT 1

#if ENABLE_MQTT
#include <WiFi.h>
#include <PubSubClient.h>
static const char *WIFI_SSID = "YOUR_WIFI_SSID";
static const char *WIFI_PASS = "YOUR_WIFI_PASSWORD";
static const char *MQTT_HOST = "192.168.1.50";   // laptop running Mosquitto
static const uint16_t MQTT_PORT = 1883;
static const char *MQTT_CLIENT_ID = "fan_rig_01";
static const char *MQTT_TOPIC = "sensor/fan_rig_01/data";
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
#endif

// ----------------------------------------------------------------- hardware
static const uint8_t MPU_ADDR = 0x68;
static const uint8_t PIN_INT = 19;
static const uint8_t PIN_LED = 2;
static const float ACCEL_LSB_PER_G = 16384.0f;   // +/-2 g full scale

// ----------------------------------------------------------------- tuning
static const uint8_t ALARM_CONSEC = 3;     // consecutive bad windows before alarm
static const uint32_t PUBLISH_MS = 1000;   // MQTT rate limit

// ----------------------------------------------------------------- buffers
#define N_FFT BASE_WINDOW                  // 256
static const float BAND_EDGES[5] = {5.0f, 50.0f, 100.0f, 200.0f, 500.0f};
static const int N_BANDS = 4;

static float ring[3][BASE_WINDOW];
static volatile uint32_t sampleCount = 0;
static uint16_t fillIndex = 0;
static uint32_t windowsSinceHop = 0;
static bool windowPrimed = false;

static float fftRe[N_FFT], fftIm[N_FFT];
static float hann[N_FFT];
static float feat[BASE_N_FEAT];
static float zfeat[BASE_N_FEAT];
static float recon[BASE_N_FEAT];
static float hidden[BASE_LATENT];

static volatile bool dataReady = false;
static uint32_t i2cErrors = 0;
static uint8_t badStreak = 0;
static bool alarmLatched = false;
static uint32_t lastPublish = 0;
static uint32_t windowIndex = 0;

void IRAM_ATTR onMpuInterrupt() { dataReady = true; }

// ----------------------------------------------------------------- MPU-6050
static bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool mpuInit() {
  Wire.begin(21, 22);
  Wire.setClock(400000);
  if (!mpuWrite(0x6B, 0x80)) return false;   // device reset
  delay(100);
  if (!mpuWrite(0x6B, 0x01)) return false;   // wake, PLL with X gyro
  delay(10);
  mpuWrite(0x1A, 0x01);                      // DLPF 184 Hz, 1 kHz internal rate
  mpuWrite(0x19, 0x00);                      // sample rate divider = 1 -> 1 kHz
  mpuWrite(0x1C, 0x00);                      // accel +/-2 g
  mpuWrite(0x37, 0x20);                      // INT latched until read
  mpuWrite(0x38, 0x01);                      // data-ready interrupt enable
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((int)MPU_ADDR, 1);
  return Wire.available() && Wire.read() == 0x68;
}

static bool mpuReadAccel(float *ax, float *ay, float *az) {
  uint8_t buf[6];
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = Wire.read();
  *ax = (int16_t)((buf[0] << 8) | buf[1]) / ACCEL_LSB_PER_G;
  *ay = (int16_t)((buf[2] << 8) | buf[3]) / ACCEL_LSB_PER_G;
  *az = (int16_t)((buf[4] << 8) | buf[5]) / ACCEL_LSB_PER_G;
  return true;
}

// ----------------------------------------------------------------- FFT
// In-place iterative radix-2 complex FFT. N_FFT is a power of two.
static void fft(float *re, float *im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / (float)len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = i + k + len / 2;
        float tr = re[b] * cr - im[b] * ci;
        float ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
        float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

// ----------------------------------------------------------------- features
// Must stay bit-for-bit equivalent to fan_features.window_features().
static void computeFeatures() {
  const float binHz = BASE_FS / (float)N_FFT;
  int k = 0;
  for (int axis = 0; axis < 3; axis++) {
    float mean = 0.0f;
    for (int i = 0; i < BASE_WINDOW; i++) mean += ring[axis][i];
    mean /= (float)BASE_WINDOW;

    float sumsq = 0.0f, mx = -1e30f, mn = 1e30f, sum4 = 0.0f;
    for (int i = 0; i < BASE_WINDOW; i++) {
      float x = ring[axis][i] - mean;
      sumsq += x * x;
      sum4 += x * x * x * x;
      if (x > mx) mx = x;
      if (x < mn) mn = x;
      fftRe[i] = x * hann[i];
      fftIm[i] = 0.0f;
    }
    float rms = sqrtf(sumsq / (float)BASE_WINDOW);
    float peak = fmaxf(fabsf(mx), fabsf(mn));
    float m4 = sum4 / (float)BASE_WINDOW;

    feat[k + 0] = rms;
    feat[k + 1] = mx - mn;
    feat[k + 2] = m4 / (powf(rms, 4.0f) + 1e-12f);
    feat[k + 3] = peak / (rms + 1e-12f);

    fft(fftRe, fftIm, N_FFT);
    float band[N_BANDS] = {0, 0, 0, 0};
    float total = 0.0f;
    for (int bin = 0; bin <= N_FFT / 2; bin++) {
      float p = fftRe[bin] * fftRe[bin] + fftIm[bin] * fftIm[bin];
      total += p;
      float f = bin * binHz;
      for (int b = 0; b < N_BANDS; b++) {
        if (f >= BAND_EDGES[b] && f < BAND_EDGES[b + 1]) { band[b] += p; break; }
      }
    }
    total += 1e-12f;
    for (int b = 0; b < N_BANDS; b++)
      feat[k + 4 + b] = logf(band[b] / total + 1e-12f);

    k += BASE_N_FEAT / 3;
  }
}

// ----------------------------------------------------------------- detectors
static float mahalanobis() {
  float d2 = 0.0f;
  for (int i = 0; i < BASE_N_FEAT; i++) {
    float acc = 0.0f;
    const float *row = &BASE_A[i * BASE_N_FEAT];
    for (int j = 0; j < BASE_N_FEAT; j++) acc += pgm_read_float(&row[j]) * zfeat[j];
    d2 += acc * acc;
  }
  return d2;
}

static float autoencoder() {
  for (int h = 0; h < BASE_LATENT; h++) {
    float acc = pgm_read_float(&BASE_B1[h]);
    for (int i = 0; i < BASE_N_FEAT; i++)
      acc += zfeat[i] * pgm_read_float(&BASE_W1[i * BASE_LATENT + h]);
    hidden[h] = tanhf(acc);
  }
  float mse = 0.0f;
  for (int o = 0; o < BASE_N_FEAT; o++) {
    float acc = pgm_read_float(&BASE_B2[o]);
    for (int h = 0; h < BASE_LATENT; h++)
      acc += hidden[h] * pgm_read_float(&BASE_W2[h * BASE_N_FEAT + o]);
    recon[o] = acc;
    float e = acc - zfeat[o];
    mse += e * e;
  }
  return mse / (float)BASE_N_FEAT;
}

// ----------------------------------------------------------------- MQTT
#if ENABLE_MQTT
static void mqttEnsure() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 8000) delay(200);
  }
  if (WiFi.status() == WL_CONNECTED && !mqtt.connected()) {
    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    mqtt.connect(MQTT_CLIENT_ID);
  }
}

static void publish(float d2, float mse, bool anomaly) {
  if (!mqtt.connected()) return;
  char payload[256];
  snprintf(payload, sizeof(payload),
           "{\"device_id\":\"%s\",\"window\":%lu,\"maha\":%.3f,\"maha_thr\":%.3f,"
           "\"ae\":%.6f,\"ae_thr\":%.6f,\"rms_x\":%.4f,\"rms_y\":%.4f,\"rms_z\":%.4f,"
           "\"anomaly\":%s,\"i2c_errors\":%lu}",
           MQTT_CLIENT_ID, (unsigned long)windowIndex, d2, BASE_MAHA_THR,
           mse, BASE_AE_THR, feat[0], feat[BASE_N_FEAT / 3], feat[2 * BASE_N_FEAT / 3],
           anomaly ? "true" : "false", (unsigned long)i2cErrors);
  mqtt.publish(MQTT_TOPIC, payload);
}
#endif

// ----------------------------------------------------------------- setup
void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  for (int i = 0; i < N_FFT; i++)
    hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)N_FFT);

  Serial.println();
  Serial.println("fan_monitor - self-baselining vibration anomaly detector");
  Serial.printf("features %d   latent %d   window %d @ %.0f Hz\n",
                BASE_N_FEAT, BASE_LATENT, BASE_WINDOW, BASE_FS);
  Serial.printf("thresholds   maha %.3f   ae %.6f\n", BASE_MAHA_THR, BASE_AE_THR);

  if (!mpuInit()) {
    Serial.println("MPU-6050 not responding - check wiring, halting");
    while (true) {
      digitalWrite(PIN_LED, !digitalRead(PIN_LED));
      delay(150);
    }
  }
  Serial.println("MPU-6050 ok, streaming at 1 kHz");

  pinMode(PIN_INT, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_INT), onMpuInterrupt, RISING);

#if ENABLE_MQTT
  mqttEnsure();
  Serial.println(WiFi.status() == WL_CONNECTED
                 ? "wifi connected" : "wifi not connected - serial only");
#endif
}

// ----------------------------------------------------------------- loop
void loop() {
  if (dataReady) {
    dataReady = false;
    float ax, ay, az;
    if (mpuReadAccel(&ax, &ay, &az)) {
      ring[0][fillIndex] = ax;
      ring[1][fillIndex] = ay;
      ring[2][fillIndex] = az;
      fillIndex = (fillIndex + 1) % BASE_WINDOW;
      sampleCount++;
      windowsSinceHop++;
      if (!windowPrimed && sampleCount >= BASE_WINDOW) windowPrimed = true;
    } else {
      i2cErrors++;
    }
  }

  if (windowPrimed && windowsSinceHop >= BASE_HOP) {
    windowsSinceHop = 0;
    windowIndex++;

    // Rotate the ring so index 0 is the oldest sample, matching Python.
    static float ordered[3][BASE_WINDOW];
    for (int axis = 0; axis < 3; axis++)
      for (int i = 0; i < BASE_WINDOW; i++)
        ordered[axis][i] = ring[axis][(fillIndex + i) % BASE_WINDOW];
    memcpy(ring, ordered, sizeof(ring));
    uint16_t savedFill = fillIndex;
    fillIndex = 0;

    computeFeatures();
    fillIndex = savedFill;

    for (int i = 0; i < BASE_N_FEAT; i++)
      zfeat[i] = (feat[i] - pgm_read_float(&BASE_MU[i])) / pgm_read_float(&BASE_SIGMA[i]);

    float d2 = mahalanobis();
    float mse = autoencoder();
    bool bad = (d2 > BASE_MAHA_THR) || (mse > BASE_AE_THR);

    if (bad) {
      if (badStreak < 255) badStreak++;
    } else {
      badStreak = 0;
    }
    alarmLatched = badStreak >= ALARM_CONSEC;
    digitalWrite(PIN_LED, alarmLatched ? HIGH : LOW);

    Serial.printf("w=%lu maha=%.2f/%.2f ae=%.5f/%.5f %s%s\n",
                  (unsigned long)windowIndex, d2, BASE_MAHA_THR, mse, BASE_AE_THR,
                  bad ? "ANOM" : "ok", alarmLatched ? " [ALARM]" : "");

#if ENABLE_MQTT
    if (millis() - lastPublish >= PUBLISH_MS || alarmLatched) {
      lastPublish = millis();
      mqttEnsure();
      publish(d2, mse, alarmLatched);
    }
#endif
  }

#if ENABLE_MQTT
  if (mqtt.connected()) mqtt.loop();
#endif
}
