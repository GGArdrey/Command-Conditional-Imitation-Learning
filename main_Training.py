from PilotNetZoo import PilotNetMultiHeadSmall, \
    PilotNetMultiHeadDANNSmall, \
    PilotNetMultiHeadDANNSmallAuxiliary, \
    PilotNetSingleHeadGatedSmall, PilotNetSingleHeadNotGatedSmall




if __name__ == "__main__":
    params = {
        "data_dirs": {'/home/luca/carla/source/recordings/town07_big2': 0,
                      '/home/luca/carla/source/recordings/town02_big': 1,
                      },
        "save_dir": "/home/luca/carla/source/training",
        "target_width": 200,
        "target_height": 66,
        "batch_size": 128,
        "epochs": 50,
        "initial_learning_rate": 1e-3,
        "domain_loss_weight": 0.0,
        "model_path": None,  # Load pre-trained model,
        "use_domain_weights": False,
        "use_command_weights": False,
        "boundaries": [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.02, 0,
             0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        "use_adapters": False,
        "adapter_training_phase": False,
        # Command augmentation parameters
        "random_command_rate": 0.5,  # N% of samples in each sequence get random commands
        "sequence_length_range": (1, 1),  # When replacing, do N-M consecutive samples
        "command_timing_shift_rate": 0.0,  # N% of samples get timing shifts
        "command_timing_shift_range": (0, 0)  # Shift by up to N samples earlier/later
    }
    print(params)
    pilotnet = PilotNetSingleHeadGatedSmall(params)
    pilotnet.train()

