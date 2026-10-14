#include <Wire.h>

void setup() {
  Serial.begin(115200);
  delay(1000);                         // WAIT for Serial to initialize
  Wire.begin(21, 22);
  Serial.println("\nScanning...");
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("Device found at 0x");
      Serial.println(addr, HEX);
    }
  }
  Serial.println("Done.");
}
void loop() {}