# Ghost-Lattice × Arinzay's Arm — the Seamless Setup

*Teaching guide. You are the software side; Arinzay is the hardware side.
The whole point: the quantum reservoir is ALREADY trained. What she
experiences is a 2-minute calibration, not a training session.*

---

## 1. The 30-second version of the architecture

```
headband (EEG) ──LSL──> laptop: Ghost-Lattice ──USB serial──> Arduino ──> servos
                            |                                   |
                 distress stop (jaw clench)          panic button + relay
                 = software flag, parallel           = hardware cutoff,
                   to the decode pipeline              default = NO POWER
```

- **Frozen brain**: gate + reservoir evolved once, offline, on the BCI
  IV 2a dataset. Never touched again at demos.
- **Per-user readout**: a convex ridge fitted on HER 2 minutes of data.
  This is why no "training" is needed per person — the readout solves
  in one closed-form step (that is the barren-plateau bypass in
  practice).
- **Safety**: two independent kill paths. Nothing learned stands
  between a person and a moving motor.

## 2. Who does what

| Step | You (software) | Arinzay (hardware) |
|---|---|---|
| Firmware | Send her `firmware/prosthetic_arm/prosthetic_arm.ino` | Flash it in Arduino IDE, report back "GL-ARM ready" |
| Wiring | — | Servo signal→D9, servo power→relay NO contact, relay coil→D7, panic button→D2+GND |
| Bench test | `run_live_bci.py --simulate --port COMx` | Watch the arm obey O/C/R and the relay click on S |
| Calibration | Run the wizard | Wear the headband, follow the 4 prompts |
| Live demo | Same command | Move the hand by thought; clench = e-stop |

## 3. Wiring (Arinzay's checklist — 10 minutes)

1. **Servo signal** → Arduino D9 (brown=GND, red=5V, orange=D9).
2. **Relay module**: VCC→5V, GND→GND, IN→D7. Servo's red wire goes
   through the relay's **NO contact** (battery + → relay → servo red).
   NEVER feed servos from the Arduino 5V pin.
3. **Panic button**: one leg → D2, other → GND (firmware uses
   INPUT_PULLUP; pressed = LOW).
4. Serial: laptop USB → Arduino (that is the whole link, 115200 baud).

## 4. Running a session (your checklist)

```bash
# rehearsal, zero hardware: simulated headband AND no serial
python run_live_bci.py --simulate

# bench test with her arm on COM5 (rehearsal brain, real arm)
python run_live_bci.py --simulate --port COM5

# the real thing
python run_live_bci.py --lsl --port COM5 --profile profiles/arinzay.npz
```

First run for any user launches the **calibration wizard**:

1. `REST` — relax 20 s
2. `CLOSE` — imagine slowly closing the hand 20 s
3. `OPEN` — imagine slowly opening the hand 20 s
4. `JAW CLENCH` — 10 s hard clench (sets the e-stop threshold)

Profile saved to `profiles/arinzay.npz`. Next time the same person
skips straight to live. Different person = different profile file —
the frozen pipeline never changes.

## 5. The serial contract (tell her this, it's 4 letters)

| Laptop sends | Arm does |
|---|---|
| `O` / `C` / `R` | glide to OPEN / CLOSE / REST (critically damped, no jitter) |
| `S` | **relay cuts servo power, latched** |
| `A` | re-arm power — operator action only, never automatic |

The arm boots **unpowered**. Power exists only after an explicit `A`.
The panic button cuts power even if the laptop dies mid-demo.

## 6. Safety rules (non-negotiable, from the paper)

1. The distress stop is a **fixed threshold detector**, calibrated on
   her clench — deliberately not learned. You can test it in 5 minutes:
   `python -c "from ghost_lattice.safety import selftest; print(selftest())"`
2. The e-stop path is **parallel** to the decode pipeline, never
   downstream of it.
3. Every live session starts with the arm **unpowered** until you send
   `A`.
4. Range-test the panic button before every audience.

## 7. If something misbehaves

| Symptom | Likely cause | Fix |
|---|---|---|
| No LSL stream found | headband app not streaming | start its LSL outlet, `--lsl` again |
| Commands right in rehearsal, wrong live | cap shift / fit | recalibrate (2 min); the manifold handles geometry, the readout handles her |
| Arm jitters | servo power from Arduino 5V | power through the relay/battery, not the board |
| E-stop fires during normal use | clench threshold too low | recalibrate with a harder clench |

## 8. What to say when people ask "how does it work"

"The quantum reservoir was evolved once on a public EEG dataset — it
is frozen. For each person, two minutes of calibration fits a convex
readout with a closed-form global optimum: no neural-network training,
no wait, and mathematically no barren plateau. The limb itself moves
on a critically damped ODE — that is why it glides instead of
twitching — and a fixed-threshold jaw-clench detector wired to a
hardware relay can always cut the motors, independent of every line of
code I wrote."
