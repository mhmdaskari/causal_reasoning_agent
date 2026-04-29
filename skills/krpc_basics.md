# kRPC Basics: Connecting and Accessing Game Objects

## Connecting

```python
import krpc

conn = krpc.connect(name="My Script")
sc = conn.space_center
```

`krpc.connect()` returns a `Connection`. The `space_center` attribute is the entry point for everything in the game.

## Getting the Active Vessel

```python
vessel = sc.active_vessel
```

The active vessel object exposes control, flight data, parts, and orbit information.

## Accessing Celestial Bodies

```python
bodies = sc.bodies          # dict[str, CelestialBody]
kerbin = bodies["Kerbin"]
mun = bodies["Mun"]
```

Common body attributes:
- `body.gravitational_parameter` — μ (m³/s²)
- `body.equatorial_radius` — meters
- `body.surface_gravity` — m/s²
- `body.sphere_of_influence` — SOI radius in meters
- `body.non_rotating_reference_frame` — inertial frame centered on the body

You can also set the active target:
```python
sc.target_body = mun
```

## Universal Time

```python
ut = sc.ut   # float, seconds since epoch
```

## Streams (High-frequency Polling)

Instead of calling a property in a loop (which round-trips over RPC each time), create a stream:

```python
ut_stream = conn.add_stream(getattr, sc, "ut")
altitude_stream = conn.add_stream(getattr, vessel.flight(ref_frame), "mean_altitude")

# Use like a function call:
current_ut = ut_stream()
current_altitude = altitude_stream()
```

Streams are much faster for tight control loops. Always prefer them when reading the same value repeatedly.

## Vessel Situation

```python
situation = str(vessel.situation).lower()
# Possible values: "pre_launch", "landed", "splashed", "flying", "sub_orbital",
#                  "orbiting", "escaping", "docked"
```

## Vessel Position and Velocity

```python
frame = body.non_rotating_reference_frame
pos = vessel.position(frame)    # (x, y, z) tuple in meters
vel = vessel.velocity(frame)    # (x, y, z) tuple in m/s
```

## Time Warp

```python
sc.warp_to(target_ut)   # warps to just before target_ut
```

## Common Pattern: Wait Until a Condition

```python
import time

def wait_until(condition, poll_interval_s=0.1):
    while not condition():
        time.sleep(poll_interval_s)

wait_until(lambda: vessel.orbit.apoapsis_altitude > 80_000)
```
