# vibration-anomaly-detector
Built a full-stack anomaly detection pipeline (ESP32 → MQTT → Python → HTML5 dashboard) using two unsupervised detectors: Mahalanobis and a 356-byte TensorFlow Lite autoencoder. Self-baselines on-device with no labeled fault data. Tested on 6 telemetry blocks: 0 false positives, under 5 second detection latency.
