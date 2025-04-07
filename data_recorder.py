# data_recorder.py
import io
import os
import csv
from datetime import datetime
import numpy as np
from PIL import Image
import carla


class SessionManager:
    """
    Manages the recording of data for a single session, which may contain
    multiple (recovery) sequences.
    """
    def __init__(self, base_folder='recordings', timestamp_in_folder_name=True, num_classes=None):
        if timestamp_in_folder_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_folder = os.path.join(base_folder, timestamp)
        else:
            self.session_folder = base_folder

        self.sequence_folder = None
        os.makedirs(self.session_folder, exist_ok=True)

        # Loggers for session and recovery sequences
        self.sequence_counter = 0
        self.recovery_logger = None
        self.session_logger = None
        self.image_counter = 0

        self.num_classes = num_classes  # needed to extend CSV if probabilities are provided

        # Initialize the session-wide logger
        session_csv_path = os.path.join(self.session_folder, 'session_data.csv')
        self.session_logger = DataLogger(session_csv_path, num_classes=self.num_classes)

    def start_new_sequence(self, sequence_name="recovery"):
        """
        Start a new recovery sequence.
        """
        self.sequence_counter += 1
        self.image_counter = 0  # Reset image counter for the new sequence
        self.sequence_folder = os.path.join(self.session_folder, f'{sequence_name}_{self.sequence_counter}')
        os.makedirs(self.sequence_folder, exist_ok=True)
        recovery_csv_path = os.path.join(self.sequence_folder, f'{sequence_name}_data.csv')
        self.recovery_logger = DataLogger(recovery_csv_path, num_classes=self.num_classes)

    def save_image(self, image, steering_angle, current_road_option):
        """
        Save an image for the current recovery sequence.
        """
        if self.recovery_logger:
            self.image_counter += 1
            filename = f"{self.image_counter}_{current_road_option}_{steering_angle:.4f}.jpg"
            filepath = os.path.join(self.sequence_folder, filename)

            # Convert CARLA image to NumPy array
            image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
            image_data = image_data.reshape((image.height, image.width, 4))
            image_data = image_data[:, :, :3][:, :, ::-1]  # Convert from BGRA to RGB

            # Save the image
            pil_image = Image.fromarray(image_data)
            pil_image.save(filepath, format="JPEG", quality=100)

    def log_sequence_data(self, *args, **kwargs):
        """
        Log data only for the current recovery sequence.
        """
        if self.recovery_logger:
            self.recovery_logger.log(*args, **kwargs)

    def log_session_data(self, *args, **kwargs):
        """
        Log data for the entire session (session-wide logger).
        """
        if self.session_logger:
            self.session_logger.log(*args, **kwargs)

    def close_sequence(self):
        """
        Close the current sequence logger.
        """
        if self.recovery_logger:
            self.recovery_logger.close()
            self.recovery_logger = None

    def close(self):
        """
        Close all loggers.
        """
        if self.session_logger:
            self.session_logger.close()
        if self.recovery_logger:
            self.recovery_logger.close()


class DataLogger:
    """
    Handles writing logs to CSV.
    """
    def __init__(self, filename, num_classes=None):
        self.file = open(filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.num_classes = num_classes

        columns = [
            'Timestamp', 'CrossTrackError', 'StanleySteeringAngle', 'PilotNetSteeringAngle',
            'ExecutedSteeringAngle', 'ExecutedBy', 'IsCarAtJunction', 'CurrentRoadOption', 'InjectedRoadOption',
            'X', 'Y', 'Z', 'VelX', 'VelY', 'VelZ', 'AccelX', 'AccelY', 'AccelZ',
            'AngularVelX', 'AngularVelY', 'AngularVelZ'
        ]
        if self.num_classes is not None:
            for i in range(self.num_classes):
                columns.append(f'Probability_Class_{i}')
        self.writer.writerow(columns)

    def log(
        self,
        timestamp,
        cte,
        stanley_steering_angle,
        isCarAtJunction,
        current_road_option,
        injected_road_option,
        vehicle,
        pilotnet_steering_angle=None,
        probabilities=None,
        executed_steering_angle=None,
        executed_by=None
    ):
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()

        row = [
            timestamp,
            cte,
            stanley_steering_angle,
            pilotnet_steering_angle,
            executed_steering_angle,
            executed_by,
            isCarAtJunction,
            current_road_option,
            injected_road_option,
            location.x,
            location.y,
            location.z,
            velocity.x,
            velocity.y,
            velocity.z,
            acceleration.x,
            acceleration.y,
            acceleration.z,
            angular_velocity.x,
            angular_velocity.y,
            angular_velocity.z
        ]

        if self.num_classes is not None and probabilities is not None:
            row.extend(probabilities)

        self.writer.writerow(row)

    def close(self):
        self.file.close()
