// Connect 555 Pin 3 to A0
const int inputPin = A0; 

void setup() {
  // Use a high baud rate. This MUST match the Python script.
  Serial.begin(115200); 
  
  // Set pin mode explicitly
  pinMode(inputPin, INPUT);
}

void loop() {
  // Read the PWM output (0 to 5V becomes 0 to 1023)
  int val = analogRead(inputPin); 
  
  // Print the raw value followed by a newline
  // Python's readline() depends on this newline to know a sample is finished.
  Serial.println(val); 
  
  // No delay here! We need maximum sampling speed to see 
  // the high-frequency pulses of your 555 modulator.
}