# main_PilotNetJunction.py
import os
import time
import logging
import collections
import pygame
import carla
import tensorflow as tf
import numpy as np
import cv2
import traceback
from typing import Dict, Optional, Callable, List

# Custom modules
from CarlaEnvironment import CarlaEnvironment
from StanleyController import StanleyController
from data_recorder import SessionManager
from PilotNetJunctionBase import BasePilotNet, preprocess_image
from routes import *
from utils.uncertainty import compute_entropy, monte_carlo_dropout
import analyze_logs
from CommandInjector import CommandInjector


# Configuration
tf.keras.config.enable_unsafe_deserialization()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_INPUT_SIZE = (200, 66)  # (width, height)
VELOCITY_CHECK_INITIAL_FRAMES = 50
RECOVERY_EXTENSION_MULTIPLIER = 2
FONT_SETTINGS = ('Arial', 24)



class ExperimentConfig:
    """Container for experiment parameters"""

    def __init__(self,
                 model_path: str,
                 map_name: str,
                 route_func: Callable,
                 use_rendering: bool = True,
                 use_high_beam: bool = False,
                 weather_preset: Optional[Dict] = None,
                 avg_speed_threshold: float = 2.0,
                 cte_threshold: float = 1.0,
                 cte_threshold_junction: float = 2.0,
                 reset_cte_threshold: float = 0.05,
                 use_recovery_extension: bool = False,
                 use_weighted_average: bool = False,
                 fine_tuning_dir: Optional[str] = None,
                 fine_tuning_ext: str = "",
                 num_command_injection: int = 0):
        self.model_path = model_path
        self.map_name = map_name
        self.route_func = route_func
        self.use_rendering = use_rendering
        self.use_high_beam = use_high_beam
        self.weather_preset = weather_preset or {}
        self.avg_speed_threshold = avg_speed_threshold
        self.cte_threshold = cte_threshold
        self.cte_threshold_junction = cte_threshold_junction
        self.reset_cte_threshold = reset_cte_threshold
        self.use_recovery_extension = use_recovery_extension
        self.use_weighted_average = use_weighted_average
        self.fine_tuning_dir = fine_tuning_dir
        self.fine_tuning_ext = fine_tuning_ext
        self.num_command_injection = num_command_injection


class ExperimentRunner:
    """
    Main class for running CARLA experiments
    Can run normal driving benchmarks and command injections benchmarks
    Further, manual navigation commands can also be provided with arrow keys, vehicle is resetted with spacebar
    If manual navigation is prefered, than CTE thresholds should be set to very high values,
    because the stanley controller is always active in the background
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env = None
        self.model = None
        self.session_manager = None
        self.stanley_controller = None

        self.pilotnet_helper = None
        self.current_waypoint = None
        self.is_in_recovery_mode = False
        self.timesteps_in_recovery = 0
        self.velocity_log = collections.deque()
        self.velocity_check_counter = 0
        self.font = None
        self.injected_command = None
        self.command_injector = None


    def initialize_environment(self):
        """Initialize CARLA environment and vehicle"""
        self.env = CarlaEnvironment(map_name=self.config.map_name)
        time.sleep(5) # Wait for environment to load
        self._setup_vehicle_and_sensors()
        self._setup_weather_and_lighting()
        pygame.font.init()
        pygame.key.set_repeat(10)
        self.font = pygame.font.SysFont('Arial', 24)

    def _setup_vehicle_and_sensors(self):
        """Spawn vehicle and configure sensors"""
        route_indices = self.config.route_func()
        spawn_points = self.env.sort_spawn_points_by_location(
            self.env.world.get_map().get_spawn_points()
        )
        self.env.spawn_vehicle(vehicle_model='model3', spawn_point=spawn_points[route_indices[0]])
        self.env.setup_sensors()
        self.env.set_spectator_top_down()

    def _setup_weather_and_lighting(self):
        """Configure environmental conditions"""
        if self.config.weather_preset:
            self.env.set_weather(self.config.weather_preset)
        if self.config.use_high_beam:
            self.env.activate_vehicle_high_beam()

    def load_model(self):
        """Load and validate the trained model"""
        self.model = tf.keras.models.load_model(self.config.model_path, compile=False)
        self._validate_model_input_shape()
        self._initialize_pilotnet_helper()

    def _validate_model_input_shape(self):
        """Ensure model input matches expected dimensions"""
        input_shape = self.model.input_shape[1]
        if (input_shape[1], input_shape[2]) != DEFAULT_MODEL_INPUT_SIZE[::-1]:
            raise ValueError(f"Model input shape mismatch. Expected {DEFAULT_MODEL_INPUT_SIZE}, got {input_shape}")

    def _initialize_pilotnet_helper(self):
        """Initialize PilotNet helper class"""
        self.pilotnet_helper = BasePilotNet(params={
            "data_dirs": {"": 0},
            "save_dir": "",
            "target_width": DEFAULT_MODEL_INPUT_SIZE[0],
            "target_height": DEFAULT_MODEL_INPUT_SIZE[1],
            "batch_size": 128,
            "epochs": 100,
            "initial_learning_rate": 1e-3,
            "domain_loss_weight": 0.0
        }, create_directories=False)

    def initialize_controllers(self):
        """Initialize Stanley controller and route"""
        self.stanley_controller = StanleyController(self.env.vehicle)
        route_indices = self.config.route_func()
        spawn_points = self.env.sort_spawn_points_by_location(
            self.env.world.get_map().get_spawn_points()
        )
        locations = [spawn_points[i].location for i in route_indices]
        self.stanley_controller.generate_route(locations)

        # Initialize after route is generated
        self.command_injector = CommandInjector(self.stanley_controller, num_frames_to_inject=self.config.num_command_injection)

    def setup_logging(self):
        """Initialize data recording system"""
        if self.config.fine_tuning_dir is None:
            self._create_auto_logging_directory()

        self.session_manager = SessionManager(
            base_folder=self.config.fine_tuning_dir,
            timestamp_in_folder_name=False,
            num_classes=len(self.pilotnet_helper.boundaries)
        )

    def _create_auto_logging_directory(self):
        """Generate automatic logging directory structure"""
        route_name = self.config.route_func.__name__
        model_name = os.path.split(self.config.model_path)[-1].replace(".keras", "")
        self.config.fine_tuning_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.config.model_path)),
            f"Fine_Tuning_{route_name}{self.config.fine_tuning_ext}_{model_name}"
        )
        os.makedirs(self.config.fine_tuning_dir, exist_ok=False)

    def run(self):
        """Main experiment execution loop"""
        try:
            self._warmup_simulation()
            while not self.stanley_controller.is_route_completed():
                self._process_frame()
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            traceback.print_exc()
        finally:
            self._cleanup()

    def _warmup_simulation(self):
        """Initial simulation warmup period"""
        for _ in range(300):
            self.env.world.tick()
            _ = self.env.get_image()

    def _process_frame(self):
        """Process a single simulation frame"""
        self.env.world.tick()
        image = self.env.get_image()

        if image:
            self._handle_pygame_events()
            self._process_image(image)
            self._update_velocity_log(image.timestamp)
            self._check_low_speed_reset(image)

    def _handle_pygame_events(self):
        """Process user input events"""
        self.injected_command = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                self._handle_keyboard_event(event)

    def _handle_keyboard_event(self, event):
        """Handle keyboard input events"""
        if event.key == pygame.K_SPACE:
            self._handle_manual_reset()
        elif event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
            self._handle_manual_command(event.key)

    def _handle_manual_reset(self):
        """Reset vehicle position manually"""
        if self.current_waypoint:
            try:
                self._log_intervention(self.last_timestamp, self.last_cross_track_error)
                self.env.vehicle.set_transform(self.current_waypoint.transform)
                logger.info("Vehicle manually reset to lane center")
            except Exception as e:
                logger.error(f"Failed to reset vehicle: {e}")
        else:
            logger.warning("No current waypoint available for manual reset")

    def _handle_manual_command(self, key):
        """Convert keyboard input to manual command"""
        key_to_command = {
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_UP: "STRAIGHT",
            pygame.K_DOWN: "LANEFOLLOW"
        }
        self.injected_command = key_to_command.get(key)

    def _process_image(self, image):
        """Process camera image and compute control actions"""
        rgb_frame = self.env.convert_CarlaImage_to_RGB(image)
        stanley_data = self._calculate_stanley_control()
        pilotnet_data = self._calculate_pilotnet_control(rgb_frame, stanley_data['road_option'])

        self._update_recovery_state(stanley_data['cross_track_error'], stanley_data['is_junction'])
        self._execute_control(stanley_data, pilotnet_data)
        self._log_frame_data(image, image.timestamp, stanley_data, pilotnet_data)

        if self.config.use_rendering:
            self._render_frame(image, stanley_data, pilotnet_data)

    def _calculate_stanley_control(self):
        """Compute Stanley controller steering and state"""
        steering, waypoint, road_option, cte = self.stanley_controller.compute_steering()
        self.current_waypoint = waypoint

        # Get injected command using current waypoint index
        if self.injected_command is None:
            self.injected_command = self.command_injector.get_injected_command(
                road_option,
                self.stanley_controller.closest_idx,
                self.env.vehicle.get_location(),  # Pass vehicle location
                self.env.vehicle.get_velocity()  # Pass vehicle speed
            )


        return {
            'steering': steering,
            'waypoint': waypoint,
            'road_option': self.injected_command,
            'cross_track_error': cte,
            'is_junction': waypoint.is_junction if waypoint else False,
            'true_command': road_option
        }

    def _calculate_pilotnet_control(self, rgb_frame, road_option):
        """Compute PilotNet steering prediction"""
        image_tensor = self._preprocess_image_for_model(rgb_frame)
        command_tensor = self._preprocess_command(road_option)

        predictions = self.model.predict(
            {'image_input': image_tensor, 'command_input': command_tensor},
            verbose=0
        )

        steering_angle = self._compute_steering_angle(predictions["class_output"])


        return {
            'steering': steering_angle,
            'probabilities': predictions["class_output"][0]
        }

    def _update_recovery_state(self, cross_track_error, is_junction):
        """Update recovery mode state based on cross-track error"""
        cte_threshold = self.config.cte_threshold_junction if is_junction else self.config.cte_threshold

        if not self.is_in_recovery_mode and abs(cross_track_error) > cte_threshold:
            self._enter_recovery_mode()
        elif self.is_in_recovery_mode:
            self._check_recovery_exit(cross_track_error)

    def _enter_recovery_mode(self):
        """Initialize recovery mode"""
        self.session_manager.start_new_sequence()
        self.is_in_recovery_mode = True
        self.timesteps_in_recovery = 0
        logger.info("Entering recovery mode...")

    def _check_recovery_exit(self, cross_track_error):
        """Check if we should exit recovery mode"""
        if abs(cross_track_error) < self.config.reset_cte_threshold:
            if self.config.use_recovery_extension:
                if self.timesteps_in_recovery > 0:
                    self.timesteps_in_recovery -= 1
                else:
                    self._exit_recovery_mode()
            else:
                self._exit_recovery_mode()
        else:
            self.timesteps_in_recovery += 1

    def _exit_recovery_mode(self):
        """Clean up and exit recovery mode"""
        self.is_in_recovery_mode = False
        self.session_manager.close_sequence()
        logger.info("Leaving recovery mode...")

    def _execute_control(self, stanley_data, pilotnet_data):
        """Apply computed control to vehicle"""
        steering = stanley_data['steering'] if self.is_in_recovery_mode else pilotnet_data['steering']
        control = carla.VehicleControl(
            throttle=0.3,
            steer=steering
        )
        self.env.vehicle.apply_control(control)

    def _update_velocity_log(self, timestamp):
        """Update velocity tracking log"""
        if self.velocity_check_counter < VELOCITY_CHECK_INITIAL_FRAMES:
            self.velocity_check_counter += 1
            return

        velocity = self.env.vehicle.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        self.velocity_log.append((timestamp, speed))

        # Remove entries older than 2s
        while self.velocity_log and (timestamp - self.velocity_log[0][0] > 2.0):
            self.velocity_log.popleft()

    def _check_low_speed_reset(self, image):
        """Check if vehicle needs automatic reset due to low speed"""
        if not self.velocity_log:
            return

        avg_speed = np.mean([v[1] for v in self.velocity_log])
        if avg_speed < self.config.avg_speed_threshold and self.current_waypoint:
            self._perform_auto_reset(image.timestamp)
            self.velocity_log.clear()

    def _perform_auto_reset(self, timestamp):
        """Perform automatic reset of vehicle position"""
        try:
            self._log_intervention(timestamp, self.last_cross_track_error)
            self.env.vehicle.set_transform(self.current_waypoint.transform)
            logger.info("Vehicle auto-reset due to low speed")
        except Exception as e:
            logger.error(f"Failed to auto-reset vehicle: {e}")

    def _cleanup(self):
        """Clean up resources and generate final report"""
        try:
            if self.session_manager:
                self.session_manager.close()

            if self.env:
                self.env.clean_up()

            # Generate analysis report
            report_path = os.path.join(self.config.fine_tuning_dir, "session_data.csv")
            if os.path.exists(report_path):
                self._generate_report(report_path)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _generate_report(self, data_path):
        """Generate final analysis report"""
        try:
            report_dir = analyze_logs.analyze(data_path)
            with open(report_dir, "a") as f:
                f.write(f"Model: {self.config.model_path}\n")
                f.write(f"Map: {self.config.map_name}\n")
                f.write(f"Route: {self.config.route_func.__name__}\n")
                f.write(f"High Beam: {self.config.use_high_beam}\n")
                f.write(f"Weather: {self.config.weather_preset}\n")
                f.write(f"Speed Threshold: {self.config.avg_speed_threshold}\n")
                f.write(f"CTE Threshold: {self.config.cte_threshold}\n")
                f.write(f"Reset CTE Threshold: {self.config.reset_cte_threshold}\n")
                f.write(f"Recovery Extension: {self.config.use_recovery_extension}\n")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

    # Visualization methods
    def _render_frame(self, image, stanley_data, pilotnet_data):
        """Render current frame with debug information"""
        camera_surface = self.env.process_image(image)
        self.env.screen.blit(camera_surface, (0, 0))

        executed_steering = stanley_data['steering'] if self.is_in_recovery_mode else pilotnet_data['steering']
        mode = "Stanley" if self.is_in_recovery_mode else "PilotNet"

        texts = [
            f'Steering: {executed_steering:.2f}',
            f'CTE: {stanley_data["cross_track_error"]:.2f}',
            f'Mode: {mode}',
            f'Command: {stanley_data["road_option"]}'
        ]

        for i, text in enumerate(texts):
            surface = self.font.render(text, True, (255, 255, 255))
            self.env.screen.blit(surface, (10, 10 + i * 30))

        pygame.display.flip()


    # Helper methods
    def _preprocess_image_for_model(self, rgb_frame):
        """Preprocess image for model input"""
        image_tensor = preprocess_image(
            rgb_frame,
            self.pilotnet_helper.target_width,
            self.pilotnet_helper.target_height
        )
        return tf.expand_dims(image_tensor, axis=0)

    def _preprocess_command(self, road_option):
        """Preprocess command for model input"""
        cmd_idx = self.pilotnet_helper.control_command_to_index.get(
            road_option,
            self.pilotnet_helper.control_command_to_index['LANEFOLLOW']
        )
        cmd_one_hot = tf.keras.utils.to_categorical(
            cmd_idx,
            num_classes=self.pilotnet_helper.total_control_commands
        )
        return tf.convert_to_tensor(np.expand_dims(cmd_one_hot, axis=0), dtype=tf.float32)

    def _compute_steering_angle(self, class_probs):
        """Compute steering angle from model predictions"""
        if self.config.use_weighted_average:
            return np.dot(class_probs[0], self.pilotnet_helper.boundaries)
        else:
            steering_class = np.argmax(class_probs[0])
            return self.pilotnet_helper.boundaries[steering_class]

    def _log_frame_data(self, image, timestamp, stanley_data, pilotnet_data):
        """Log frame data and save images if in recovery mode"""
        executed_steering = stanley_data['steering'] if self.is_in_recovery_mode else pilotnet_data['steering']
        executed_by = "Stanley" if self.is_in_recovery_mode else "PilotNet"

        self.session_manager.log_session_data(
            timestamp,
            stanley_data['cross_track_error'],
            stanley_data['steering'],
            stanley_data['is_junction'],
            stanley_data['true_command'],
            stanley_data['road_option'],  # Injected command
            self.env.vehicle,
            pilotnet_steering_angle=pilotnet_data['steering'],
            probabilities=pilotnet_data['probabilities'],
            executed_steering_angle=executed_steering,
            executed_by=executed_by
        )

        if self.is_in_recovery_mode:
            self.session_manager.save_image(image, executed_steering, stanley_data['road_option'])

        # Track for potential interventions
        self.last_timestamp = timestamp
        self.last_cross_track_error = stanley_data['cross_track_error']

    def _log_intervention(self, timestamp, cross_track_error):
        """Log intervention event"""
        self.session_manager.log_session_data(
            timestamp,
            cross_track_error,
            0.0,
            False,
            None,
            None,
            self.env.vehicle,
            pilotnet_steering_angle=0.0,
            probabilities=None,
            executed_steering_angle=0.0,
            executed_by="Intervention"
        )



if __name__ == '__main__':
    '''
    How to run a set of experiments...
    '''

    weather_town02 = {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "sun_azimuth_angle": 250.0,
        "sun_altitude_angle": 30.0,
        "fog_density": 10.0,
        "fog_distance": 60.0,
        "fog_falloff": 0.9,
        "wetness": 0.0,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.03,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0
    }

    # Configuration mappings
    town_configs = {
        "Town07_Opt": {
            "route": town7_opt4km,
            "weather": None
        },
        # "Town01_Opt": {
        #     "route": town1_opt8km,
        #     "weather": weather_town02
        # },
        # "Town02_Opt": {
        #     "route": town2_opt4km,
        #     "weather": None
        # }
    }

    # List of models to test
    model_paths = [
        "/home/luca/carla/source/training/12-02-2025_14-44/checkpoints/cp-0050.keras",
    ]

    injections = [0, 50] # will test every model with no injection and with 5sec. long command injection

    # Track progress
    total_experiments = len(model_paths) * len(town_configs) * len(injections)
    completed_experiments = 0

    # Run all combinations
    for town_name, town_config in town_configs.items():
        for injection_count in injections:
            for model_path in model_paths:
                model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
                experiment_id = f"{model_name}_{town_name}_inj{injection_count}"
                print(f"\nStarting experiment {completed_experiments + 1}/{total_experiments}: {experiment_id}")

                try:
                    experiment_config = ExperimentConfig(
                        model_path=model_path,
                        map_name=town_name,
                        route_func=town_config["route"],
                        use_rendering=True,
                        weather_preset=town_config["weather"],
                        avg_speed_threshold=2.0,
                        cte_threshold=1.0,
                        cte_threshold_junction=2.0,
                        reset_cte_threshold=0.05,
                        fine_tuning_ext=f"_inj-{injection_count}",
                        num_command_injection=injection_count
                    )

                    # Run experiment with timeout
                    runner = ExperimentRunner(experiment_config)
                    runner.initialize_environment()
                    runner.load_model()
                    runner.initialize_controllers()
                    runner.setup_logging()
                    runner.run()
                    completed_experiments += 1

                except Exception as e:
                    print(f"Experiment failed: {str(e)}")
                    traceback.print_exc()



