**Minimum Distance Maintenance**: Each drone must maintain a minimum distance of 40 cm from other drones and obstacles at all times.
**Obstacle Avoidance**: Drones should sense and avoid obstacles within their field of view by adjusting their velocities to maintain a safe distance.
**Velocity Alignment**: Each drone must align its velocity with the average velocity of nearby drones to achieve coordinated movement.
**Cohesion towards Center**: Drones must adjust their movement to stay close to the center of their local group to maintain flock cohesion.
**Boundary Compliance**: Drones must ensure their position remains within the 3D bounded volume defined as X, Y in [-200, 200] cm and Z in [20, 200] cm.
**Real-time Updates**: Drones must update their position and velocity in real-time based on the most current environment and peer information.