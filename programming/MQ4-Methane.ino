/* Using a MQ-4 Sensor with Arduino to detect Methane (CH4) gas levels in the air */
/* Jenna Epstein */
/* Sensing the City - Project Tutorial Component | University of Pennsylvania */

/* MQ-4 sensors detect the presence of methane gas (CH4) in the air. This code establishes a connection between an MQ-4 sensor and an Arduino
and prints out an analog output to the serial monitor. The analog signal that is generated and printed is proportional to the amount of gas detected in the air. 
The code also establishes a routine for a digital output. When digital output is HIGH (representing a high amount of gas detected), this triggers a red LED light to illuminate. */ 

/* This code was adapted from https://microcontrollerslab.com/mq4-methane-gas-sensor-pinout-interfacing-with-arduino-features/ */

const int AO_Pin= A0; // Connect the AO of MQ-4 sensor with analog channel 0 pin (A0) of Arduino // Connect an LED with D13 pin of Arduino
int AO_Out; // Create a variable to store the analog output of the MQ-4 sensor

// Set up
void setup() {
Serial.begin(115200);  // Initialize serial monitor using a baud rate of 115200
}

// Main loop
void loop()
{
AO_Out= analogRead(AO_Pin); // Read the analog output measurement sample from the MQ4 sensor's AO pin
Serial.print("Ammonia Conentration: "); // Print out the text "Methane Concentration: "
Serial.println(AO_Out); // Print out the methane value - the analog output - beteewn 0 and 1023
delay(1000); // Set a 10 second delay
}