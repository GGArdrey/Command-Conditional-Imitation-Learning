import math
import carla
import numpy as np
import sys
sys.path.append('/home/luca/carla/PythonAPI/carla/')
from agents.navigation.global_route_planner import GlobalRoutePlanner

class StanleyController:
    '''
    Stanley Controller for vehicle steering control in CARLA.
    Uses CARLAs GlobalRoutePlanner to generate a route between waypoints.
    The controller computes the steering angle based on the vehicle's position and orientation relative to the path.
    The compute_steering() method returns the normalized steering angle, the closest waypoint,
    and the road option (navigation command).

    '''
    def __init__(self, vehicle, k_e=2, k_s=0.01, k_d=0.1):
        self.waypoint_separation = 0.05 # Waypoint separation in meters
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        self.k_e = k_e  # Cross-track error gain
        self.k_s = k_s  # Softening factor
        self.k_d = k_d  # Derivative gain for steering smoothing
        self.max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle  # Max steering angle in degrees
        #self.wheelbase = 2.875  # Wheelbase in meters
        self.route_waypoints = []
        self.route_road_options = []
        self.closest_waypoint = None
        self.last_steering_angle = 0.0  # For derivative control
        self.closest_idx = 0

        # Generate initial vehicle state
        self.last_vehicle_location, self.last_vehicle_yaw = self._get_vehicle_location()

    def reset(self):
        """Resets the controller state but keep the route."""
        self.closest_waypoint = None
        self.last_steering_angle = 0.0
        self.closest_idx = 0


    def generate_route(self, locations):
        """Generates a route between multiple waypoints using GlobalRoutePlanner."""
        if len(locations) < 2:
            raise ValueError("At least two waypoints are required to generate a route.")

        grp = GlobalRoutePlanner(self.map, sampling_resolution=self.waypoint_separation)
        self.route_waypoints = []
        self.route_road_options = []
        for i in range(len(locations) - 1):
            start_wp = self.map.get_waypoint(locations[i], project_to_road=True, lane_type=carla.LaneType.Any)
            end_wp = self.map.get_waypoint(locations[i + 1], project_to_road=True, lane_type=carla.LaneType.Any)
            route_segment = grp.trace_route(start_wp.transform.location, end_wp.transform.location)

            if not route_segment:
                print(f"Warning: No route segment found between waypoint {i} and {i + 1}.")
                continue  # Skip empty route segments

            self.route_waypoints.extend([wp for wp, _ in route_segment])
            self.route_road_options.extend(roadOptions for _, roadOptions in route_segment)
        if not self.route_waypoints:
            raise ValueError("Route generation failed. No valid route segments were created.")

        # Remove duplicate waypoints
        self._remove_duplicate_waypoints()
        anomaly_waypoints = self._check_waypoint_continuity()
        return anomaly_waypoints

    def _check_waypoint_continuity(self):
        """Checks the continuity of the route waypoints by measuring the spacing between them and detecting anomalies."""
        anomaly_waypoints = []
        anomalies = []
        threshold_multiplier = 2.1  # Adjust this multiplier as needed

        for i in range(len(self.route_waypoints) - 1):
            wp_current = self.route_waypoints[i].transform.location
            wp_next = self.route_waypoints[i + 1].transform.location
            distance = wp_current.distance(wp_next)

            # Detect anomalies if the distance is significantly larger than expected
            if distance > self.waypoint_separation * threshold_multiplier:
                anomalies.append((i, distance))
                anomaly_waypoints.append(self.route_waypoints[i])
                anomaly_waypoints.append(self.route_waypoints[i + 1])

        if anomalies:
            print("-" * 25)
            print("Anomalies detected between waypoints:")
            for idx, dist in anomalies:
                print(f"Waypoint {idx} to {idx + 1}: distance = {dist:.2f} meters")
            print("-" * 25)
        else:
            print("-" * 25)
            print("No anomalies detected in waypoint continuity.")
            print("-" * 25)

        return anomaly_waypoints

    def _remove_duplicate_waypoints(self):
        """Removes duplicate waypoints based on their ID if they occur one after the other."""
        unique_route = []
        len_total_route = len(self.route_waypoints)
        len_unique_route = 0
        last_id = None

        for waypoint in self.route_waypoints:
            if waypoint.id == last_id:
                continue
            else:
                unique_route.append(waypoint)
                len_unique_route += 1
            last_id = waypoint.id
        print("-" * 25)
        print("Removed", len_total_route - len_unique_route, "duplicate waypoints, out of:", len_total_route)
        print("-" * 25)
        print("Route length is approx.: ", round(len_unique_route * self.waypoint_separation,2), "meters")
        print("-" * 25)

        return unique_route

    @staticmethod
    def _normalize_angle(angle):
        """Normalizes an angle to the range [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _calculate_heading_error(self, vehicle_yaw, path_yaw):
        """Calculates the heading error between the vehicle's yaw and the path's yaw."""
        heading_error = path_yaw - vehicle_yaw
        return self._normalize_angle(heading_error)

    def _calculate_cross_track_error(self, vehicle_location, waypoint, path_direction_unit):
        """Calculates the signed perpendicular distance from the vehicle to the path."""
        waypoint_location = waypoint.transform.location
        normal_vector = np.array([path_direction_unit[1], -path_direction_unit[0]])
        vector_wp_to_vehicle = np.array([
            vehicle_location.x - waypoint_location.x,
            vehicle_location.y - waypoint_location.y
        ])
        cross_track_error = np.dot(vector_wp_to_vehicle, normal_vector)
        return cross_track_error

    def _find_closest_waypoints(self, vehicle_location):
        """Finds the closest waypoint and its successor on the path to the vehicle's position."""
        # Initialize self.closest_idx if it doesn't exist
        if not hasattr(self, 'closest_idx'):
            self.closest_idx = 0

        N_BACK = int(1 * (1/self.waypoint_separation)) # Number of waypoints to look back, 5 meters
        N_FORWARD = int(2 * (1/self.waypoint_separation))  # Number of waypoints to look ahead

        start_idx = max(self.closest_idx - N_BACK, 0)
        end_idx = min(self.closest_idx + N_FORWARD, len(self.route_waypoints) - 1)

        min_dist_sqr = float('inf')
        closest_idx = self.closest_idx

        vehicle_x = vehicle_location.x
        vehicle_y = vehicle_location.y

        # Search for the closest waypoint within the window
        for i in range(start_idx, end_idx + 1):
            waypoint = self.route_waypoints[i]
            dx = waypoint.transform.location.x - vehicle_x
            dy = waypoint.transform.location.y - vehicle_y
            dist_sqr = dx ** 2 + dy ** 2
            if dist_sqr < min_dist_sqr:
                min_dist_sqr = dist_sqr
                closest_idx = i

        self.closest_idx = closest_idx

        current_wp = self.route_waypoints[closest_idx]
        current_road_option = self.route_road_options[closest_idx]

        if closest_idx + 1 < len(self.route_waypoints):
            next_wp = self.route_waypoints[closest_idx + 1]
        else:
            next_wp = current_wp  # end of route

        dx = next_wp.transform.location.x - current_wp.transform.location.x
        dy = next_wp.transform.location.y - current_wp.transform.location.y

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            if closest_idx > 0:
                prev_wp = self.route_waypoints[closest_idx - 1]
                dx = current_wp.transform.location.x - prev_wp.transform.location.x
                dy = current_wp.transform.location.y - prev_wp.transform.location.y
            else:
                vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
                dx = math.cos(vehicle_yaw)
                dy = math.sin(vehicle_yaw)

        path_direction = np.array([dx, dy])
        norm = np.linalg.norm(path_direction)
        path_direction_unit = path_direction / (norm + 1e-4)
        path_yaw = math.atan2(path_direction_unit[1], path_direction_unit[0])

        return current_wp, current_road_option, path_direction_unit, path_yaw

    def _get_axle_locations(self):
        """Calculate front and rear axle centers from wheel positions."""
        physics_control = self.vehicle.get_physics_control()
        wheels = physics_control.wheels

        # front wheels: 0 = front-left, 1 = front-right
        fl = wheels[0].position
        fr = wheels[1].position
        front_axle_center_x = (fr.x + 0.5 * (fl.x - fr.x)) / 100.0
        front_axle_center_y = (fr.y + 0.5 * (fl.y - fr.y)) / 100.0
        front_axle_center_z = (fr.z + 0.5 * (fl.z - fr.z)) / 100.0
        front_axle_location = carla.Location(front_axle_center_x, front_axle_center_y, front_axle_center_z)

        # rear wheels: 2 = rear-left, 3 = rear-right
        rl = wheels[2].position
        rr = wheels[3].position
        rear_axle_center_x = (rr.x + 0.5 * (rl.x - rr.x)) / 100.0
        rear_axle_center_y = (rr.y + 0.5 * (rl.y - rr.y)) / 100.0
        rear_axle_center_z = (rr.z + 0.5 * (rl.z - rr.z)) / 100.0
        rear_axle_location = carla.Location(rear_axle_center_x, rear_axle_center_y, rear_axle_center_z)

        return front_axle_location, rear_axle_location

    def _get_vehicle_location(self):
        """Get the vehicle's front axle location and yaw angle in radians."""
        front_axle_location, rear_axle_location = self._get_axle_locations()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        return front_axle_location, vehicle_yaw

    def is_off_road(self, threshold=1.5):
        """Check if the vehicle is off-road based on the distance to the closest waypoint."""
        vehicle_location, _ = self._get_vehicle_location()
        closest_wp, _, _, _ = self._find_closest_waypoints(vehicle_location)
        waypoint_location = closest_wp.transform.location
        distance = vehicle_location.distance(waypoint_location)
        return distance > threshold

    def compute_steering(self):
        """Computes the steering angle for the vehicle using the Stanley control law with derivative term."""
        if len(self.route_waypoints) == 0:
            raise ValueError("Route waypoints are not set. Cannot check route completion. Call generate_route() first.")

        # Get vehicle state
        vehicle_location, vehicle_yaw = self._get_vehicle_location()

        # Get vehicle speed
        velocity = self.vehicle.get_velocity()
        speed = max(velocity.length(), 0.1)  # Avoid division by zero

        # Find closest waypoints and path information
        closest_waypoint, closest_waypoint_road_option, path_direction_unit, path_yaw = self._find_closest_waypoints(vehicle_location)

        # Calculate heading error
        heading_error = self._calculate_heading_error(vehicle_yaw, path_yaw)

        # Calculate cross-track error
        cross_track_error = self._calculate_cross_track_error(vehicle_location, closest_waypoint, path_direction_unit)

        # Stanley control law
        cross_track_term = math.atan2(self.k_e * cross_track_error, speed + self.k_s)
        steering_angle = heading_error + cross_track_term

        # Convert to degrees and clamp
        steering_angle_deg = math.degrees(steering_angle)
        steering_angle_deg = np.clip(steering_angle_deg, -self.max_steer_angle, self.max_steer_angle)

        #print(f"Max steer angle: {self.max_steer_angle}, Steering angle: {steering_angle_deg:.2f}, ")

        # Apply derivative term for smoothing
        # D-term based on change in steering angle from previous step
        kd_term = self.k_d * (self.last_steering_angle - steering_angle_deg)
        steering_angle_deg += kd_term
        # Update last steering angle
        self.last_steering_angle = steering_angle_deg

        # Normalize steering angle
        steering_angle_deg_norm = steering_angle_deg / self.max_steer_angle

        # Extract the road option for logging
        closest_waypoint_road_option = str(closest_waypoint_road_option).rsplit(".")[-1]

        return steering_angle_deg_norm, closest_waypoint, closest_waypoint_road_option, cross_track_error

    def is_route_completed(self):
        """Checks if the vehicle has reached the end of the route."""
        if len(self.route_waypoints) == 0:
            raise ValueError("Route waypoints are not set. Cannot check route completion. Call generate_route() first.")
        return self.closest_idx == len(self.route_waypoints) - 1
