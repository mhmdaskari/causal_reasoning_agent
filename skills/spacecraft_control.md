# Spacecraft Control: Throttle, Staging, and Autopilot

## Basic Controls

```python
control = vessel.control

control.throttle = 1.0    # 0.0 to 1.0
control.sas = True        # stability assist
control.rcs = True        # reaction control thrusters
```

## Staging

```python
control.activate_next_stage()
current_stage = control.current_stage   # int, counts down
```

Be careful not to stage too rapidly. Always enforce a minimum interval between stages (e.g., 2 seconds) to let decouplers and fairings clear.

Check whether staging is needed by inspecting fuel resources:
```python
resources = vessel.resources_in_decouple_stage(stage_number, cumulative=False)
liquid = resources.amount("LiquidFuel")
oxidizer = resources.amount("Oxidizer")
solid = resources.amount("SolidFuel")
```

Active engines can be found via:
```python
engines = [e for e in vessel.parts.engines if e.active]
```

A stage is effectively depleted when its liquid propellants are near zero (< 0.1 units) or all active engines report `available_thrust < 1.0`.

## Autopilot

kRPC's built-in autopilot points the vessel in a commanded direction:

```python
ap = vessel.auto_pilot
ap.engage()

# Point at pitch=90 (straight up), heading=90 (east):
ap.target_pitch_and_heading(90.0, 90.0)

# Point toward a maneuver node:
ap.reference_frame = node.reference_frame
ap.target_direction = (0.0, 1.0, 0.0)   # prograde in node frame
ap.wait()   # blocks until the vessel is pointing close to target

ap.disengage()
```

`target_pitch_and_heading(pitch, heading)` is convenient during ascent. For maneuver burns, pointing in the node's prograde direction `(0, 1, 0)` in `node.reference_frame` is the standard approach.

## Thrust and Mass

```python
vessel.available_thrust   # Newtons, at current throttle settings
vessel.thrust             # current actual thrust
vessel.mass               # kg, includes fuel
vessel.specific_impulse   # Isp in seconds (multiply by 9.82 for exhaust velocity)
```

Thrust-to-weight ratio at the surface:
```python
twr = vessel.available_thrust / (vessel.mass * body.surface_gravity)
```

## Flight Data

```python
ref = vessel.surface_reference_frame
flight = vessel.flight(ref)

flight.mean_altitude        # meters above sea level
flight.surface_altitude     # meters above terrain
flight.vertical_speed       # m/s (positive = ascending)
flight.horizontal_speed     # m/s
flight.dynamic_pressure     # Pascal (aerodynamic pressure, "Q")
flight.mach_number
```

## Gravity Turn Pattern

A gravity turn gradually pitches from vertical to horizontal as the rocket climbs through the atmosphere. A common approach:

- Below the turn start altitude: fly straight up (pitch = 90)
- Between turn start and turn end altitudes: interpolate pitch from 90° down toward 0° based on altitude fraction
- Above the turn end altitude: fly horizontal (pitch = 0)

Throttle should be managed to maintain a target thrust-to-weight ratio. Reduce throttle if dynamic pressure is too high.

## MET (Mission Elapsed Time)

```python
vessel.met   # seconds since launch
```
