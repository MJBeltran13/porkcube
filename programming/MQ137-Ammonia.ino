const int mq137Pin = A0; // Analog pin connected to the sensor

void setup() {
  Serial.begin(9600); // Start serial communication
}

void loop() {
  int sensorValue = analogRead(mq137Pin); // Read sensor value
  float voltage = sensorValue * (5.0 / 1023.0); // Convert sensor value to voltage (assuming 5V Arduino)
  
  Serial.print("MQ-137 PPM: ");
  Serial.println(voltage);
  
  delay(1000); // Wait for a second before taking the next reading
}
