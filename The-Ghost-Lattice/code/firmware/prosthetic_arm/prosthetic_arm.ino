/*
 * Ghost-Lattice Prosthetic Arm Firmware
 * Eric Yaka (Elbalor) - CC BY-NC 4.0
 *
 * One-letter protocol over USB serial (115200 baud):
 *   'O'  -> target OPEN   (position 0)
 *   'C'  -> target CLOSE  (position 180)
 *   'R'  -> target REST   (position 90)
 *   'S'  -> DISTRESS STOP: latch the power relay OFF (failsafe state)
 *   'A'  -> re-arm power AFTER a stop, operator-confirmed only
 *
 * Safety model (mirrors ghost_lattice/safety.py):
 *   - servo power rail goes through a relay on PIN_RAIL. Power exists
 *     ONLY while the latch is armed. Default/fault state = no power.
 *   - PIN_PANIC (physical button to GND, INPUT_PULLUP) cuts power in
 *     hardware, completely independent of the laptop and this code.
 *   - the firmware smooths motion itself (critically damped), so even
 *     a burst of commands can never jitter the limb.
 */

#include <Servo.h>

const int PIN_SERVO = 9;     // signal wire (white/orange)
const int PIN_RAIL  = 7;     // relay / MOSFET gate -> servo power rail
const int PIN_PANIC = 2;     // physical panic button (to GND)

const float DT       = 0.02; // 50 Hz control loop
const float OMEGA_N  = 6.0;  // natural frequency (rad/s)

Servo hand;
bool  rail_armed = false;
float pos  = 90.0;           // current commanded position (deg)
float vel  = 0.0;            // deg/s
float goal = 90.0;

void power_off() {           // the ONLY way power exists is through this
  rail_armed = false;
  digitalWrite(PIN_RAIL, LOW);
}

void power_on() {            // only after explicit 'A' from the operator
  rail_armed = true;
  digitalWrite(PIN_RAIL, HIGH);
}

void setup() {
  pinMode(PIN_RAIL, OUTPUT);
  pinMode(PIN_PANIC, INPUT_PULLUP);
  power_off();                        // FAULT STATE: no power to motors
  hand.attach(PIN_SERVO);
  hand.write(pos);
  Serial.begin(115200);
  Serial.println("GL-ARM ready (power OFF; send A to arm)");
}

void loop() {
  unsigned long t0 = millis();

  // 1. panic button: independent hardware path, checked every cycle
  if (digitalRead(PIN_PANIC) == LOW) {
    power_off();
    Serial.println("PANIC");
  }

  // 2. commands from the Ghost-Lattice laptop
  while (Serial.available()) {
    char c = Serial.read();
    switch (c) {
      case 'O': goal = 0.0;   Serial.println("CMD OPEN");  break;
      case 'C': goal = 180.0; Serial.println("CMD CLOSE"); break;
      case 'R': goal = 90.0;  Serial.println("CMD REST");  break;
      case 'S': power_off();  Serial.println("STOP LATCHED"); break;
      case 'A': power_on();   Serial.println("ARMED (operator)");
                break;
    }
  }

  // 3. critically damped motion: y'' = w^2 (g - y) - 2w y'
  float acc = OMEGA_N * OMEGA_N * (goal - pos) - 2.0 * OMEGA_N * vel;
  vel += acc * DT;
  pos += vel * DT;
  hand.write(pos);

  // 4. hold 50 Hz
  unsigned long dt = millis() - t0;
  if (dt < 20) delay(20 - dt);
}
