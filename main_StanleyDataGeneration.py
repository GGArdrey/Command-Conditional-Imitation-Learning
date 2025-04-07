import pygame
import carla
from CarlaEnvironment import CarlaEnvironment
from StanleyController import StanleyController
from data_recorder import SessionManager
from routes import *

'''
Main file to generate data using stanley controller. 
'''
def main():
    env = None
    try:
        # Initialize environment
        env = CarlaEnvironment(map_name="Town02_Opt")

        # Initialize Session Manager
        session_manager = SessionManager()
        session_manager.start_new_sequence("sequence")

        # Define waypoints (list of carla.Location)
        map = env.world.get_map()
        spawn_points = map.get_spawn_points()
        spawn_points = env.sort_spawn_points_by_location(spawn_points)
        print("Number of Spawn Points: ", len(spawn_points))
        route = town2_opt4km() # probide a list of waypoints to traverse
        locations = [spawn_points[i].location for i in route]
        vehicle = env.spawn_vehicle(vehicle_model='model3', spawn_point=spawn_points[route[0]])
        env.setup_sensors()
        env.set_spectator_top_down()

        # Initialize the Stanley Controller with multiple waypoints
        stanley_controller = StanleyController(vehicle)
        anomaly_waypoints = stanley_controller.generate_route(locations)
        #print("Anomaly Waypoints: ", anomaly_waypoints)

        # For debugging
        #env.draw_spawn_points()
        #env.draw_waypoints(stanley_controller.route_waypoints, life_time=9999999)

        # Run the simulation for a few steps to settle vehicle physics
        for i in range(300):
            env.world.tick()
            image = env.get_image()

        while True:
            env.world.tick()  # Get the current simulation tick
            delta_seconds = 0.1

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # Get the latest image and process it
            image = env.get_image()

            if image:
                # Compute steering using the Stanley controller
                stanley_steering_angle, current_waypoint, current_road_option, cross_track_error = stanley_controller.compute_steering()
                isCarAtJunction = current_waypoint.is_junction

                # Save image and log data
                timestamp = image.timestamp
                session_manager.save_image(image, stanley_steering_angle, current_road_option)
                session_manager.log_sequence_data(image.timestamp, cross_track_error, stanley_steering_angle, isCarAtJunction,
                                                  current_road_option, current_road_option, vehicle)
                session_manager.log_session_data(timestamp, cross_track_error, stanley_steering_angle, isCarAtJunction, current_road_option, current_road_option, vehicle)


                # Render the camera feed in Pygame window
                camera_surface = env.process_image(image)
                env.screen.blit(camera_surface, (0, 0))
                pygame.display.flip()

                # Apply control to the vehicle
                control = carla.VehicleControl()
                control.throttle = 0.3  # Adjust throttle as needed
                control.steer = stanley_steering_angle
                vehicle.apply_control(control)

                # Check if route is completed
                if stanley_controller.is_route_completed():
                    print("Route completed.")
                    # Start a new sequence if needed
                    session_manager.close_sequence()
                    session_manager.close()
                    break

    finally:
        # Clean up
        print("Cleaning up...")
        if env:
            env.clean_up()

if __name__ == '__main__':
    main()
