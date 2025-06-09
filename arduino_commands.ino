#include <Servo.h>

// Servo objects
Servo Servo_0; // Base
Servo Servo_1; // Ombro (Shoulder)
Servo Servo_2; // Cotovelo (Elbow)
Servo Servo_3; // Garra (Gripper)

// Current positions of servos
int currentPos_0 = 90;
int currentPos_1 = 90;
int currentPos_2 = 90;
int currentPos_3 = 170; // Gripper open

// Maximum buffer size for incoming serial data
const int MAX_SERIAL_BUFFER = 64;
char serialBuffer[MAX_SERIAL_BUFFER]; // Buffer to store incoming chars
int serialBufferIndex = 0;           // Current position in the buffer

void setup() {
  Serial.begin(9600); // Start serial communication at 9600 baud

  // Attach servos to their respective pins
  Servo_0.attach(4);
  Servo_1.attach(5);
  Servo_2.attach(6);
  Servo_3.attach(7);

  // Set servos to initial positions
  Servo_0.write(currentPos_0);
  Servo_1.write(currentPos_1);
  Servo_2.write(currentPos_2);
  Servo_3.write(currentPos_3);

  Serial.println("Arduino ready. Waiting for commands in format <step_id:velocity>");
  // Clear the serial buffer initially
  memset(serialBuffer, 0, MAX_SERIAL_BUFFER);
  serialBufferIndex = 0;
}

void loop() {
  // Check if data is available to read from serial port
  while (Serial.available() > 0) {
    char incomingChar = Serial.read(); // Read one character

    // Check for the end-of-message marker (newline character)
    if (incomingChar == '\n') {
      serialBuffer[serialBufferIndex] = '\0'; // Null-terminate the string

      processCommand(serialBuffer); // Process the complete command

      // Reset buffer for the next message
      memset(serialBuffer, 0, MAX_SERIAL_BUFFER);
      serialBufferIndex = 0;
    } else {
      // Add character to the buffer if there's space
      if (serialBufferIndex < MAX_SERIAL_BUFFER - 1) {
        serialBuffer[serialBufferIndex++] = incomingChar;
      } else {
        // Buffer overflow: discard message and reset
        Serial.println("Error: Buffer overflow. Message discarded.");
        memset(serialBuffer, 0, MAX_SERIAL_BUFFER);
        serialBufferIndex = 0;
      }
    }
  }
}

void processCommand(char* command) {
  Serial.print("Received raw: ");
  Serial.println(command); // Debug: Print raw received string

  // Validate command frame: must start with '<' and end with '>'
  if (command[0] == '<' && command[strlen(command) - 1] == '>') {
    char* colonPtr = strchr(command, ':'); // Find the colon separator

    if (colonPtr != NULL) { // Colon found
      int colonIndex = colonPtr - command;
      command[colonIndex] = '\0'; // Temporarily split string at colon
      int step_id = atoi(command + 1); // Convert part after '<' to step_id

      char* velocityStr = colonPtr + 1; // Part after colon is velocity
      char* endMarkerPtr = strchr(velocityStr, '>');
      if (endMarkerPtr != NULL) {
        *endMarkerPtr = '\0'; // Remove the trailing '>'
        int velocity = atoi(velocityStr); // Convert velocity string to int

        Serial.print("Parsed Step ID: "); Serial.print(step_id);
        Serial.print(", Velocity: "); Serial.println(velocity);

        // Define base delay values (milliseconds) for servo movement
        int minServoDelay = 15; // Shortest delay
        int maxServoDelay = 70; // Longest delay

        // Control servos based on step_id
        switch (step_id) {
          case 0: // Rest
            moveServoSmooth(&Servo_0, &currentPos_0, 90, velocity, minServoDelay, maxServoDelay);
            moveServoSmooth(&Servo_1, &currentPos_1, 90, velocity, minServoDelay, maxServoDelay);
            moveServoSmooth(&Servo_2, &currentPos_2, 90, velocity, minServoDelay, maxServoDelay);
            moveServoSmooth(&Servo_3, &currentPos_3, 170, velocity, minServoDelay, maxServoDelay); // Open gripper
            break;
          
          // CUP SEQUENCE
          case 1: // Reach for Cup
            moveServoSmooth(&Servo_1, &currentPos_1, 120, velocity, minServoDelay, maxServoDelay);
            moveServoSmooth(&Servo_2, &currentPos_2, 110, velocity, minServoDelay, maxServoDelay);
            break;
          case 2: // Grasp Cup
            moveServoSmooth(&Servo_3, &currentPos_3, 90, velocity, minServoDelay, maxServoDelay);
            break;
          case 3: // Lift Cup
            moveServoSmooth(&Servo_1, &currentPos_1, 70, velocity, minServoDelay, maxServoDelay);
            break;
          case 4: // Drink
            moveServoSmooth(&Servo_2, &currentPos_2, 60, velocity, minServoDelay, maxServoDelay);
            break;
          case 5: // Lower Cup
            moveServoSmooth(&Servo_2, &currentPos_2, 110, velocity, minServoDelay, maxServoDelay);
            moveServoSmooth(&Servo_1, &currentPos_1, 120, velocity, minServoDelay, maxServoDelay);
            break;

          // SOUP STEPS
          case 6: // Reach for Spoon
            moveServoSmooth(&Servo_1, &currentPos_1, 120, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_2, &currentPos_2, 110, velocity, minServoDelay, maxServoDelay); 
            break;
          case 7: // Grasp Spoon
            moveServoSmooth(&Servo_3, &currentPos_3, 90, velocity, minServoDelay, maxServoDelay); 
            break;
          case 8: // Scoop Soup
            moveServoSmooth(&Servo_0, &currentPos_0, 70, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_1, &currentPos_1, 80, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_2, &currentPos_2, 100, velocity, minServoDelay, maxServoDelay); 
            break;
          case 9: // Bring Spoon to Mouth
            moveServoSmooth(&Servo_1, &currentPos_1, 70, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_2, &currentPos_2, 70, velocity, minServoDelay, maxServoDelay);  
            break;
          case 10: // Return Spoon to Bowl
            moveServoSmooth(&Servo_2, &currentPos_2, 100, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_1, &currentPos_1, 80, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_0, &currentPos_0, 90, velocity, minServoDelay, maxServoDelay);  
            break;
          case 11: // Lower Spoon (and release)
            moveServoSmooth(&Servo_2, &currentPos_2, 110, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_1, &currentPos_1, 120, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_3, &currentPos_3, 170, velocity, minServoDelay, maxServoDelay); 
            break;

          // BOOK GRAB STEPS
          case 12: // Reach for Book
            moveServoSmooth(&Servo_1, &currentPos_1, 130, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_2, &currentPos_2, 120, velocity, minServoDelay, maxServoDelay); 
            break;
          case 13: // Grasp Book
            moveServoSmooth(&Servo_3, &currentPos_3, 90, velocity, minServoDelay, maxServoDelay);  
            break;
          case 14: // Lift Book
            moveServoSmooth(&Servo_1, &currentPos_1, 60, velocity, minServoDelay, maxServoDelay);  
            break;
          case 15: // Turn Book
            moveServoSmooth(&Servo_2, &currentPos_2, 50, velocity, minServoDelay, maxServoDelay); 
            break;
          case 16: // Lower Book (and release)
            moveServoSmooth(&Servo_2, &currentPos_2, 120, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_1, &currentPos_1, 130, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_3, &currentPos_3, 170, velocity, minServoDelay, maxServoDelay); 
            break;

          // DOOR KNOB STEPS
          case 17: // Reach for Door Knob
            moveServoSmooth(&Servo_1, &currentPos_1, 110, velocity, minServoDelay, maxServoDelay); 
            moveServoSmooth(&Servo_2, &currentPos_2, 100, velocity, minServoDelay, maxServoDelay); 
            break;
          case 18: // Grasp Door Knob
            moveServoSmooth(&Servo_3, &currentPos_3, 80, velocity, minServoDelay, maxServoDelay);  
            break;
          case 19: // Turn Door Knob 
            moveServoSmooth(&Servo_0, &currentPos_0, 70, velocity, minServoDelay, maxServoDelay);  
            break;
          case 20: // Retract Hand 
            moveServoSmooth(&Servo_2, &currentPos_2, 90, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_1, &currentPos_1, 90, velocity, minServoDelay, maxServoDelay);  
            moveServoSmooth(&Servo_0, &currentPos_0, 90, velocity, minServoDelay, maxServoDelay);  
            break;
          case 21: // Release Door Knob
            moveServoSmooth(&Servo_3, &currentPos_3, 170, velocity, minServoDelay, maxServoDelay); 
            break;

          default:
            Serial.println("Warning: Unknown step_id received.");
            break;
        }
      } else {
        Serial.println("Error: Missing end marker '>' after colon.");
      }
    } else {
      Serial.println("Error: Missing colon ':' separator.");
    }
  } else {
    Serial.println("Error: Invalid command frame. Missing '<' or '>'.");
  }
}

// Function to move a servo smoothly to a target position using dynamic speed
void moveServoSmooth(Servo* servo, int* currentPos, int targetAngle, int parsedVelocity, int baseMinDelay, int baseMaxDelay) {
  int stepSize = 1;    // Move 1 degree at a time
  int actualDelayTime; // Calculated delay between steps

  targetAngle = constrain(targetAngle, 0, 180); // Ensure target is within servo limits

  // Map the parsedVelocity (0-255 from Python) to a delay time.
  // Higher velocity -> shorter delay.
  if (parsedVelocity <= 0) { // If velocity is zero or negative, use max delay (slowest)
    actualDelayTime = baseMaxDelay;
  } else {
    actualDelayTime = map(parsedVelocity, 1, 255, baseMaxDelay, baseMinDelay);
  }
  actualDelayTime = constrain(actualDelayTime, baseMinDelay, baseMaxDelay); // Ensure delay is within defined bounds

  Serial.print("  Smooth Move: Servo to "); Serial.print(targetAngle);
  Serial.print(" from "); Serial.print(*currentPos);
  Serial.print(", VelInput="); Serial.print(parsedVelocity);
  Serial.print(", CalcDelay="); Serial.println(actualDelayTime);

  while (*currentPos != targetAngle) {
    if (targetAngle > *currentPos) {
      (*currentPos) += stepSize;
    } else {
      (*currentPos) -= stepSize;
    }
    servo->write(*currentPos);
    delay(actualDelayTime);
  }
  Serial.println("  Smooth Move: Target reached.");
}
