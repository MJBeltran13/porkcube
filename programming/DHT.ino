#include <dht11.h>

#define DHT11_PIN 2

dht11 DHT11;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int DATA = DHT11.read(DHT11_PIN);

  Serial.print("Temperature (C): ");
  Serial.println((float) DHT11.temperature, 2);

  delay(2000);
}
