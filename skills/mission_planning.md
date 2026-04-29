# Mission Planning: Transfer Windows, Phase Angles, and Corrections

## Hohmann Transfer Overview

A Hohmann transfer moves a spacecraft from one circular orbit to another using two burns:
1. **Departure burn** (prograde) at the lower orbit to raise apoapsis to the target orbit
2. **Capture burn** (retrograde) at the target orbit to circularize

The transfer orbit's semi-major axis is the average of the two circular orbit radii:
```
a_transfer = (r_departure + r_target) / 2
```

Transfer time (half the period of the transfer ellipse):
```
t_transfer = π * sqrt(a_transfer³ / μ)
```

## Phase Angle for a Transfer Window

At departure, the target body must be ahead of the spacecraft by the angle it will travel during the transfer. The required phase angle (in radians, measured from spacecraft to target in the prograde direction):

```python
import math

def required_phase_angle(mu, r_departure, r_target):
    a_transfer = 0.5 * (r_departure + r_target)
    t_transfer = math.pi * math.sqrt(a_transfer**3 / mu)
    target_angular_rate = math.sqrt(mu / r_target**3)
    # Angle target travels during transfer; required lead angle
    return math.pi - target_angular_rate * t_transfer
```

The result is the angle the target should be *ahead* of the spacecraft at the moment of the burn.

## Measuring Current Phase Angle

In the parent body's inertial (non-rotating) frame:

```python
frame = parent_body.non_rotating_reference_frame
vessel_pos = vessel.position(frame)
target_pos = target_body.position(frame)
```

Use vector math to find the signed angle from `vessel_pos` to `target_pos` around the orbit normal:
- Orbit normal ≈ `cross(vessel_pos, vessel_velocity)`
- Signed angle accounts for direction (prograde vs retrograde)

## Waiting for the Window

Once you know the required and current phase angles, compute the time until they match:

```python
vessel_mean_motion = math.sqrt(mu / r_departure**3)
target_mean_motion = math.sqrt(mu / r_target**3)
relative_rate = vessel_mean_motion - target_mean_motion   # rad/s (positive if inner orbit)

phase_error = (required_phase - current_phase) % (2 * math.pi)
wait_time_s = phase_error / relative_rate
```

Then warp to `sc.ut + wait_time_s - some_lead_time`.

## Transfer Delta-V

```python
def transfer_delta_v(mu, r_departure, r_target):
    a_transfer = 0.5 * (r_departure + r_target)
    v_circular = math.sqrt(mu / r_departure)
    v_transfer = math.sqrt(mu * (2.0 / r_departure - 1.0 / a_transfer))
    return v_transfer - v_circular
```

This is the prograde delta-v for the departure burn only.

## Capture Burn (Orbit Insertion)

On arrival, the spacecraft enters the target's SOI on a hyperbolic trajectory. To circularize at a target periapsis altitude:

```python
# At periapsis:
mu_target = target_body.gravitational_parameter
r_peri = target_body.equatorial_radius + target_periapsis_altitude

v_hyperbolic = vis_viva_speed(mu_target, r_peri, vessel.orbit.semi_major_axis)
v_circular   = vis_viva_speed(mu_target, r_peri, r_peri)
delta_v = v_hyperbolic - v_circular   # retrograde (negative prograde)
```

Add this node at `sc.ut + vessel.orbit.time_to_periapsis` with `prograde=-delta_v`.

## Midcourse Correction

After the transfer burn, if the predicted closest approach is still far from the target, a small correction burn (prograde, normal, and/or radial components) can improve the trajectory.

Iterate over candidate delta-v vectors and score each based on:
- Predicted separation at closest approach
- Whether the trajectory actually enters the target's SOI
- Relative inclination

The correction node is typically placed at roughly half the remaining flight time to the closest approach point.

## Optimizing a Node for Encounter

To search for a good node:
1. Start from an initial estimate (e.g., pure Hohmann delta-v)
2. Try small perturbations to UT, prograde, normal, and radial
3. Score each result (lower predicted separation = better)
4. Iterate with decreasing step sizes to converge

Use `node.orbit.distance_at_closest_approach(target_body.orbit)` and `node.orbit.next_orbit` to evaluate whether the projected path enters the SOI.

## Reference Frames Cheat Sheet

| Frame | Use |
|---|---|
| `body.non_rotating_reference_frame` | Inertial positions/velocities relative to body center; best for phase angles and vector math |
| `vessel.surface_reference_frame` | Flight data (altitude, vertical speed, dynamic pressure) |
| `node.reference_frame` | Pointing toward a maneuver node's burn direction |
| `body.reference_frame` | Rotates with the body; useful for surface-relative work |
