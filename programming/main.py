import serial
import time

# Define the serial port and baud rate
ser = serial.Serial('COM3', 9600) # Change 'COM3' to the appropriate port

def read_sensor_data():
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            if data.startswith("Temperature"):
                print("Received sensor data:", data)
                break

try:
    while True:
        read_sensor_data()
        time.sleep(2)  # Wait for 2 seconds

except KeyboardInterrupt:
    print("\nExiting program.")
    ser.close()  # Close the serial connection when Ctrl+C is pressed
