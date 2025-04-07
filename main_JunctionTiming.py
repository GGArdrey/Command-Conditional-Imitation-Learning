import os
import sys
import time
import logging
import argparse
from typing import Dict, List

# Import your existing code
from main_PilotNetJunction import ExperimentRunner, ExperimentConfig
from routes import *
from JunctionTimingExperimentRunner import JunctionTimingExperiment

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Available route functions by name
ROUTE_FUNCTIONS = {
    "town7_left_turn": town07_left,
    "town7_right_turn": town07_right,
    "town7_straight": town07_straight,
}

def main():

    # Specify a list of models to test
    model_paths = [
        "/home/luca/carla/source/training/24-02-2025_13-59/checkpoints/cp-0050.keras",
    ]
    for model_path in model_paths:
        """Main execution function."""
        # Create experiment config
        exp_config = ExperimentConfig(
            model_path=model_path,
            map_name="Town07_Opt",
            route_func=town07_left, # specify which turn to test
            use_gradcam=False,
            use_visualize_fmaps=False,
            use_rendering=True,
            # Set thresholds very high to prevent auto-reset during experiment
            cte_threshold=9999.0,
            cte_threshold_junction=9999.0,
            # Custom output directory if provided
            fine_tuning_ext=f"_junction_timing" # for naming folders
        )


        try:
            # Initialize base experiment runner
            base_runner = ExperimentRunner(exp_config)
            base_runner.initialize_environment()
            base_runner.load_model()
            base_runner.initialize_controllers()
            base_runner.setup_logging()

            junction_experiment = JunctionTimingExperiment(base_runner)
            junction_experiment.setup()

            # Run the junction timing experiments
            logger.info(f"Starting junction timing experiment offsets")
            junction_experiment.run_experiments(
                test_commands=["LEFT"], #again, specify the turn to test
                tick_delays=range(0, 16,1),  # Test early and late commands
                command_durations=range(0, 31,1),  # 10, 15, 20, 25, 30 Test infinite duration and early release
                runs_per_config=1
            )

            logger.info("Junction timing experiment completed successfully")


        except KeyboardInterrupt:
            logger.info("Experiments interrupted by user")
        except Exception as e:
            logger.error(f"Error during experiments: {e}", exc_info=True)
        finally:
            # Clean up
            if 'base_runner' in locals() and base_runner:
                base_runner._cleanup()

            logger.info("Cleanup completed")

if __name__ == "__main__":
    main()