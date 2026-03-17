#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;  

unsigned long sampleIntervalUs = 3333; // Default 300Hz
unsigned long durationMs = 300000;     // Default 5 mins
unsigned long previousMicros = 0;
unsigned long startTime = 0;
bool isRecording = false;

void setup() {
  Serial.begin(115200); 
  
  // Locked to 860 SPS to support variable sampling rates up to 860 Hz
  ads.setDataRate(RATE_ADS1115_860SPS);
  
  if (!ads.begin()) {
    Serial.println("Error: Failed to initialize ADS1115.");
    while (1);
  }
  
  Serial.println("System Ready. Waiting for dynamic configuration...");
}

void loop() {
  // 1. Wait for configuration command from Python
  if (!isRecording) {
    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\n');
      command.trim();
      
      // Expected format: "S,RATE_HZ,DURATION_SEC" (e.g., "S,300,300")
      if (command.startsWith("S,")) {
        int firstComma = command.indexOf(',');
        int secondComma = command.indexOf(',', firstComma + 1);
        
        if (firstComma > 0 && secondComma > 0) {
          int rateHz = command.substring(firstComma + 1, secondComma).toInt();
          int durSec = command.substring(secondComma + 1).toInt();
          
          if (rateHz > 0 && durSec > 0) {
            // Dynamically calculate the timer and duration
            sampleIntervalUs = 1000000UL / rateHz;
            durationMs = durSec * 1000UL;
            
            isRecording = true;
            startTime = millis();
            previousMicros = micros();
            Serial.println("START");
          }
        }
      }
    }
    return;
  }

  // 2. Check if requested duration has elapsed
  if (millis() - startTime >= durationMs) {
    isRecording = false;
    Serial.println("END");
    return;
  }

  // 3. Execute precise timing based on requested Hz
  unsigned long currentMicros = micros();
  if (currentMicros - previousMicros >= sampleIntervalUs) {
    previousMicros += sampleIntervalUs; 
    
    int16_t adc0 = ads.readADC_SingleEnded(0);
    float volts = ads.computeVolts(adc0);
    Serial.println(volts, 6);
  }
}