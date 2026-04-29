# Orbital Mechanics: Reading Orbits and Planning Burns

## Orbit Properties

```python
orbit = vessel.orbit

orbit.apoapsis_altitude      # meters above sea level (not from center)
orbit.periapsis_altitude     # meters above sea level
orbit.apoapsis               # meters from body center
orbit.periapsis              # meters from body center
orbit.semi_major_axis        # meters
orbit.time_to_apoapsis       # seconds until apoapsis
orbit.time_to_periapsis      # seconds until periapsis
orbit.period                 # orbital period in seconds

orbit.body                   # CelestialBody the vessel is orbiting
orbit.body.name              # e.g. "Kerbin", "Mun"
```

## Vis-Viva Equation

Speed at any point in an orbit:

```
v = sqrt(μ * (2/r - 1/a))
```

where:
- μ = `body.gravitational_parameter`
- r = current distance from body center (meters)
- a = semi-major axis (meters)

```python
import math

def vis_viva_speed(mu, radius, semi_major_axis):
    return math.sqrt(mu * (2.0 / radius - 1.0 / semi_major_axis))
```

## Circularization Delta-V

To circularize at apoapsis, compute the delta-v needed to raise periapsis to apoapsis:

```python
mu = vessel.orbit.body.gravitational_parameter
r = vessel.orbit.apoapsis          # burn at apoapsis
a1 = vessel.orbit.semi_major_axis  # current elliptical orbit
a2 = r                             # target circular orbit

v1 = vis_viva_speed(mu, r, a1)
v2 = vis_viva_speed(mu, r, a2)
delta_v = v2 - v1   # positive = prograde burn
```

## Maneuver Nodes

```python
# Add a node at a specific UT with a prograde component:
node = vessel.control.add_node(ut, prograde=delta_v, normal=0.0, radial=0.0)

# Node properties:
node.ut              # time of burn
node.prograde        # m/s component (prograde/retrograde)
node.normal          # m/s component (normal/anti-normal)
node.radial          # m/s component (radial/anti-radial)
node.delta_v         # total magnitude
node.remaining_delta_v   # remaining during burn (use as stream)
node.orbit           # predicted orbit after this node

# Remove when done:
node.remove()
```

Add the node at the right orbital position (e.g., apoapsis for circularization):
```python
node = vessel.control.add_node(
    sc.ut + vessel.orbit.time_to_apoapsis,
    prograde=delta_v
)
```

## Burn Time Estimation

Using the Tsiolkovsky rocket equation:

```python
def burn_time_for_delta_v(vessel, delta_v):
    thrust = vessel.available_thrust        # Newtons
    isp = vessel.specific_impulse * 9.82   # exhaust velocity (m/s)
    m0 = vessel.mass
    mf = m0 / math.exp(delta_v / isp)
    flow_rate = thrust / isp
    return (m0 - mf) / flow_rate
```

Start the burn half of `burn_time` before the node's UT to center it on the optimal point.

## Executing a Maneuver Node

General pattern:
1. Orient toward the node's burn direction: `(0, 1, 0)` in `node.reference_frame`
2. Warp to `node.ut - burn_time/2 - lead_time`
3. Open throttle and monitor `node.remaining_delta_v` (as a stream)
4. Reduce throttle as remaining delta-v decreases to avoid overshoot
5. Cut throttle when remaining delta-v ≈ 0, then `node.remove()`

## SOI Transitions

```python
orbit.time_to_soi_change    # seconds until SOI change (NaN if none)
orbit.next_orbit            # orbit in the next SOI (None if no SOI change)
next_body = orbit.next_orbit.body.name   # e.g., "Mun"
```

## Encounter Assessment

For checking how close a projected orbit comes to a target body:

```python
# Closest approach between this orbit and the target body's orbit:
closest_dist = vessel.orbit.distance_at_closest_approach(target_body.orbit)
closest_ut = vessel.orbit.time_of_closest_approach(target_body.orbit)

# Relative inclination (radians):
rel_inc = vessel.orbit.relative_inclination(target_body.orbit)
```

## Relative Inclination

```python
import math
deg = math.degrees(vessel.orbit.relative_inclination(target_body.orbit))
```

A low relative inclination (< 1–2°) means the trajectory is roughly in the same orbital plane as the target, which reduces the delta-v needed for capture.
