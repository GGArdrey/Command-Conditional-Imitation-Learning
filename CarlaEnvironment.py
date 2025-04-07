# environment.py
import carla
import pygame
import numpy as np
import queue
from carla import VehicleLightState as vls


class CarlaEnvironment:
    def __init__(self, client_address='localhost', client_port=2000, map_name='Town07_Opt', display_size=(800, 600)):
        # Initialize Pygame
        pygame.init()
        self.display_width, self.display_height = display_size
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("CARLA Simulation")

        # Connect to CARLA simulator
        self.client = carla.Client(client_address, client_port)
        self.client.set_timeout(30.0)
        #self.client.start_recorder("test_recording.log", True)
        self.world = self.client.load_world(map_name)

        # Possibility to remove layers from maps
        #self.world.unload_map_layer(carla.MapLayer.Foliage)
        #self.world.unload_map_layer(carla.MapLayer.Buildings)

        self.map = self.world.get_map()

        # Set synchronous mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enable synchronous mode
        self.settings.fixed_delta_seconds = 0.1  # 10 FPS
        self.world.apply_settings(self.settings)

        # Initialize variables
        self.vehicle = None
        self.camera = None
        self.imu_sensor = None
        self.lane_sensor = None
        self.image_queue = queue.Queue(maxsize=1)
        self.imu_queue = queue.Queue(maxsize=1)

    #https://carla.readthedocs.io/en/latest/python_api/#carla.WeatherParameters
    def set_weather(self, custom_params=None):
        """
        Creates a custom weather preset by taking specified parameters from `custom_params`
        and filling in the rest with the current weather values.

        Parameters:
        - world: The CARLA world instance.
        - custom_params: A dictionary with custom weather parameters to set.

        Returns:
        - None
        """
        # Get the current weather
        current_weather = self.world.get_weather()

        # Set the custom weather parameters, defaulting to the current weather for missing values
        weather = carla.WeatherParameters(
            cloudiness=custom_params.get('cloudiness', current_weather.cloudiness),
            precipitation=custom_params.get('precipitation', current_weather.precipitation),
            precipitation_deposits=custom_params.get('precipitation_deposits',
                                                     current_weather.precipitation_deposits),
            wind_intensity=custom_params.get('wind_intensity', current_weather.wind_intensity),
            sun_azimuth_angle=custom_params.get('sun_azimuth_angle', current_weather.sun_azimuth_angle),
            sun_altitude_angle=custom_params.get('sun_altitude_angle', current_weather.sun_altitude_angle),
            fog_density=custom_params.get('fog_density', current_weather.fog_density),
            fog_distance=custom_params.get('fog_distance', current_weather.fog_distance),
            wetness=custom_params.get('wetness', current_weather.wetness),
            fog_falloff=custom_params.get('fog_falloff', current_weather.fog_falloff),
            scattering_intensity=custom_params.get('scattering_intensity', current_weather.scattering_intensity),
            mie_scattering_scale=custom_params.get('mie_scattering_scale', current_weather.mie_scattering_scale),
            rayleigh_scattering_scale=custom_params.get('rayleigh_scattering_scale',
                                                        current_weather.rayleigh_scattering_scale),
            dust_storm=custom_params.get('dust_storm', current_weather.dust_storm)
        )

        # Apply the custom weather preset
        self.world.set_weather(weather)
        print(f"Custom weather preset applied: {custom_params}")

    def set_spectator_top_down(self):
        # Retrieve the spectator object
        spectator = self.world.get_spectator()

        # Get the location and rotation of the spectator through its transform
        transform = spectator.get_transform()

        transform.location.z += 150
        #transform.rotation.pitch -= 90

        # Set the spectator with an empty transform
        spectator.set_transform(transform)

    def draw_waypoints(self, waypoints, color=carla.Color(255, 0, 0), life_time=60.0, size=0.1):
        """Draws waypoints in the world for visualization purposes."""
        for waypoint in waypoints:
            self.world.debug.draw_point(
                waypoint.transform.location,
                size=size,
                color=color,
                life_time=life_time
            )

    def draw_locations(self, locations, color=carla.Color(0, 0, 255), life_time=60.0, size=0.1):
        """Draws waypoints in the world for visualization purposes."""
        for l in locations:
            self.world.debug.draw_point(
                l,
                size=size,
                color=color,
                life_time=life_time
            )

    def draw_spawn_points(self):
        """Draws spawn points in the world for visualization purposes."""
        spawn_points = self.map.get_spawn_points()
        spawn_points = self.sort_spawn_points_by_location(spawn_points)
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=10000000)
            self.world.debug.draw_arrow(
                spawn_point.location,
                spawn_point.location + spawn_point.get_forward_vector(),
                life_time=1000000
            )

    def sort_spawn_points_by_location(self, spawn_points):
        """
        Sorts a list of CARLA spawn points based on their x and y coordinates.

        Parameters:
        spawn_points (list of carla.Transform): List of CARLA spawn points to be sorted.

        Returns:
        list of carla.Transform: Sorted list of spawn points.
        """
        # Sort spawn points by x coordinate first, then by y coordinate
        return sorted(spawn_points, key=lambda sp: (sp.location.x, sp.location.y))

    def spawn_vehicle(self, spawn_point, vehicle_model='model3'):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_model)[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        return self.vehicle

    def activate_vehicle_high_beam(self):
        # Turn on the front lights (low beams)
        self.vehicle.set_light_state(vls(vls.HighBeam))

    def setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{self.display_width}')
        camera_bp.set_attribute('image_size_y', f'{self.display_height}')
        camera_bp.set_attribute('fov', '120')  # Adjust FOV as needed
        camera_transform = carla.Transform(carla.Location(x=1.8, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self._camera_callback)

    def setup_imu(self):
        blueprint_library = self.world.get_blueprint_library()
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=1.5))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        self.imu_sensor.listen(self._imu_callback)

    def setup_lane_invasion_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_transform = carla.Transform(carla.Location(x=0, z=0))
        self.lane_sensor = self.world.spawn_actor(lane_invasion_bp, lane_transform, attach_to=self.vehicle)
        self.lane_sensor.listen(self._lane_invasion_callback)

    def setup_sensors(self):
        self.setup_camera()
        self.setup_imu()
        self.setup_lane_invasion_sensor()

    def _camera_callback(self, image):
        if not self.image_queue.full():
            self.image_queue.put(image)

    def _imu_callback(self, imu_data):
        if not self.imu_queue.full():
            self.imu_queue.put(imu_data)

    def _lane_invasion_callback(self, event):
        pass
        #print(f"Detected lane invasion: {event.crossed_lane_markings}")

    def get_image(self):
        if not self.image_queue.empty():
            return self.image_queue.get()
        else:
            return None

    def get_imu_data(self):
        if not self.imu_queue.empty():
            return self.imu_queue.get()
        else:
            return None

    def process_image(self, image):
        array = self.convert_CarlaImage_to_RGB(image)
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        return surface

    def convert_CarlaImage_to_RGB(self, image):
        """
        Converts a CARLA image (BGRA) to a NumPy array in RGB format.
        """
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4))
        image_data = image_data[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
        return image_data

    def clean_up(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.imu_sensor:
            self.imu_sensor.stop()
            self.imu_sensor.destroy()
        if self.lane_sensor:
            self.lane_sensor.stop()
            self.lane_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        pygame.quit()

        # Reset to asynchronous mode
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
