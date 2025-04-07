from PilotNetZoo import PilotNetMultiHeadSmall, \
    PilotNetMultiHeadDANNSmall, \
    PilotNetMultiHeadDANNSmallAuxiliary, \
    PilotNetSingleHeadGatedSmall, PilotNetMultiHeadSmallRaspiCar, PilotNetSingleHeadGatedSmallRaspiCar


if __name__ == "__main__":
    '''
    How to start a training...
    '''

    params = {
        "data_dirs": {'/home/luca/carla/source/raspberrypi_car/data/Home_Data': 0,
                      '/home/luca/carla/source/raspberrypi_car/data/Home_Recovery_Data': 0},
        "save_dir": "/home/luca/carla/source/training",
        "target_width": 200,
        "target_height": 66,
        "batch_size": 128,
        "epochs": 100,
        "initial_learning_rate": 1e-3,
        "model_path": None,  # Load pre-trained model,
        "use_command_weights": False,
        "boundaries": [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # Command augmentation parameters
        "random_command_rate": 0.1,  # N% of samples in each sequence get random commands
        "sequence_length_range": (1, 1),  # When replacing, do N-M consecutive samples
        "command_timing_shift_rate": 0.0,  # N% of samples get timing shifts
        "command_timing_shift_range": (0, 0)  # Shift by up to N samples earlier/later
    }
    print(params)
    pilotnet = PilotNetSingleHeadGatedSmallRaspiCar(params)
    pilotnet.train()
