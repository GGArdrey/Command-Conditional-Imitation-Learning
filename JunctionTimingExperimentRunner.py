import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class JunctionCommandInjector:
    """Command injector with support for negative delays (early commands) and command duration testing."""

    def __init__(self, stanley_controller, num_frames_to_inject=0):
        self.stanley_controller = stanley_controller
        self.num_frames_to_inject = num_frames_to_inject

        self.max_cte = 0
        self.min_endpoint_distance = float('inf')

        # Junction indices
        self.junction_entry_idx = None
        self.junction_exit_idx = None

        # State tracking
        self.approaching_junction = False
        self.in_junction = False
        self.passed_junction = False

        # Command tracking
        self.tick_delay = 0
        self.tick_counter = 0
        self.current_tick = 0
        self.command_before_junction = "LANEFOLLOW"
        self.junction_command = None
        self.test_command = None
        self.junction_command_detected = False
        self.junction_command_applied = False

        self.pre_junction_distance = 100  # Number of waypoints before junction to start tracking
        self.early_command_applied = False
        self.command_duration = 0  # How long to apply command before releasing
        self.duration_counter = 0  # Counter for duration
        self.command_released = False  # Whether command has been released

        # Success tracking
        self.junction_success = False
        self.waypoint_deviation = None

        # Command log
        self.command_log = []

        # Find junction on initialization
        self._find_junction()

    def initialize(self, tick_delay, test_command=None, command_duration=0):
        """
        Set up the experiment with specified parameters.

        Args:
            tick_delay: Number of ticks to delay command (negative for early command)
            test_command: Command to inject (e.g., LEFT, RIGHT, STRAIGHT)
            command_duration: How long to apply command before releasing (0 = don't release)
        """
        print(f"\n=== INITIALIZING JUNCTION COMMAND INJECTOR ===")
        print(f"Delay: {tick_delay} ticks " + ("(EARLY)" if tick_delay < 0 else ""))
        print(f"Test command: {test_command if test_command else 'Using true command'}")
        print(f"Command duration: {command_duration} ticks")

        self.tick_delay = tick_delay
        self.test_command = test_command
        self.command_duration = command_duration

        # Reset all tracking variables
        self.approaching_junction = False
        self.in_junction = False
        self.passed_junction = False
        self.command_before_junction = "LANEFOLLOW"
        self.junction_command = None
        self.junction_command_detected = False
        self.junction_command_applied = False
        self.early_command_applied = False
        self.duration_counter = 0
        self.command_released = False
        self.tick_counter = 0
        self.current_tick = 0
        self.max_cte = 0
        self.min_endpoint_distance = float('inf')
        self.junction_success = False
        self.waypoint_deviation = None
        self.command_log = []

    def _find_junction(self):
        """Find the first junction in the route."""
        route = self.stanley_controller.route_waypoints

        # Find first junction waypoint
        for i, waypoint in enumerate(route):
            if waypoint.is_junction:
                # Find junction entry (first junction waypoint)
                entry_idx = i
                while entry_idx > 0 and route[entry_idx - 1].is_junction:
                    entry_idx -= 1

                # Find junction exit (last junction waypoint)
                exit_idx = i
                while exit_idx < len(route) - 1 and route[exit_idx + 1].is_junction:
                    exit_idx += 1

                self.junction_entry_idx = entry_idx
                self.junction_exit_idx = exit_idx
                print(f"Junction identified: Entry at waypoint {entry_idx}, Exit at waypoint {exit_idx}")
                return

        print("WARNING: No junction found in route!")

    def get_injected_command(self, true_command, current_waypoint_idx, vehicle_location=None, vehicle_speed=None):
        """Get the command to inject based on timing parameters."""
        # Increment tick counter for this simulation frame
        self.current_tick += 1

        # Update junction status
        self._update_junction_status(current_waypoint_idx)

        # Calculate distance to junction (for logging)
        distance_to_junction = None
        if vehicle_location and self.junction_entry_idx is not None:
            route = self.stanley_controller.route_waypoints
            if self.junction_entry_idx < len(route):
                junction_entry_loc = route[self.junction_entry_idx].transform.location
                distance_to_junction = np.sqrt((vehicle_location.x - junction_entry_loc.x) ** 2 +
                                               (vehicle_location.y - junction_entry_loc.y) ** 2)

        # Default inject LANEFOLLOW as the base command
        injected_command = self.command_before_junction

        # Store junction command when approaching or entering junction
        if (self.approaching_junction or self.in_junction) and not self.junction_command_detected:
            self.junction_command_detected = True
            # Use test command if specified, otherwise use true command
            self.junction_command = self.test_command if self.test_command else true_command

            # Reset delay counter
            self.tick_counter = 0

            if self.tick_delay < 0:
                print(f"Command will be applied {abs(self.tick_delay)} ticks BEFORE junction")
            else:
                print(f"Command will be applied {self.tick_delay} ticks AFTER junction entry")

        # Handle command application logic
        if self.junction_command_detected:
            # CASE 1: NEGATIVE DELAY (Early Command)
            if self.tick_delay < 0 and self.approaching_junction and not self.in_junction:
                # Calculate how far we are from junction in waypoints
                waypoints_to_junction = self.junction_entry_idx - current_waypoint_idx

                # Apply early command when we reach the specified negative delay
                if waypoints_to_junction <= abs(self.tick_delay) and not self.early_command_applied:
                    self.early_command_applied = True
                    self.junction_command_applied = True
                    print(
                        f"EARLY COMMAND APPLIED: {self.junction_command} at {abs(self.tick_delay)} WPs before junction")
                    injected_command = self.junction_command

                    # Start duration counter for command release
                    self.duration_counter = 0

                # Continue applying command if already started (unless duration expired)
                elif self.early_command_applied and not self.command_released:
                    # Apply command for the specified duration
                    if self.command_duration > 0:
                        if self.duration_counter < self.command_duration:
                            injected_command = self.junction_command
                            self.duration_counter += 1
                        else:
                            self.command_released = True
                            print(f"COMMAND RELEASED after {self.duration_counter} ticks")
                            injected_command = self.command_before_junction
                    else:
                        # Command duration of 0 means keep it active until end of junction
                        injected_command = self.junction_command

            # CASE 2: POSITIVE DELAY (Late Command)
            elif self.in_junction:
                # If still in delay period after junction entry, continue using LANEFOLLOW
                if self.tick_counter < self.tick_delay and not self.junction_command_applied:
                    injected_command = self.command_before_junction  # Default LANEFOLLOW
                    self.tick_counter += 1
                elif not self.junction_command_applied:
                    # Delay complete, apply junction command
                    self.junction_command_applied = True
                    print(
                        f"JUNCTION COMMAND APPLIED: {self.junction_command} after {self.tick_counter} ticks in junction")
                    injected_command = self.junction_command
                    self.duration_counter = 0
                elif self.junction_command_applied and not self.command_released:
                    # Command has been applied, check duration
                    if self.command_duration > 0:
                        if self.duration_counter < self.command_duration:
                            injected_command = self.junction_command
                            self.duration_counter += 1
                        else:
                            self.command_released = True
                            print(f"COMMAND RELEASED after {self.duration_counter} ticks")
                            injected_command = self.command_before_junction
                    else:
                        # Infinite duration (0) means keep it active until end of junction
                        injected_command = self.junction_command

        # Once we've passed the junction, always use LANEFOLLOW
        if self.passed_junction:
            injected_command = self.command_before_junction

        _, closest_waypoint, _, current_cte = self.stanley_controller.compute_steering()
        # Log this command
        self.command_log.append({
            'tick': self.current_tick,
            'cte': current_cte,
            'waypoint_idx': current_waypoint_idx,
            'distance_to_junction': distance_to_junction,
            'speed': vehicle_speed.length() if vehicle_speed else None,
            'true_command': true_command,
            'injected_command': injected_command,
            'approaching_junction': self.approaching_junction,
            'in_junction': self.in_junction,
            'passed_junction': self.passed_junction,
            'junction_command_detected': self.junction_command_detected,
            'junction_command_applied': self.junction_command_applied,
            'early_command_applied': self.early_command_applied,
            'command_released': self.command_released,
            'tick_counter': self.tick_counter,
            'duration_counter': self.duration_counter,
            'vehicle_x': vehicle_location.x if vehicle_location else None,
            'vehicle_y': vehicle_location.y if vehicle_location else None,
            'waypoint_x': closest_waypoint.transform.location.x,
            'waypoint_y': closest_waypoint.transform.location.y
        })

        return injected_command

    def _update_junction_status(self, current_waypoint_idx):
        """Update junction approach/exit status based on waypoint."""
        if self.junction_entry_idx is None:
            return

        # Check if approaching junction (within pre_junction_distance waypoints)
        if (not self.approaching_junction and
                current_waypoint_idx >= self.junction_entry_idx - self.pre_junction_distance and
                current_waypoint_idx < self.junction_entry_idx):
            self.approaching_junction = True
            print(f"APPROACHING JUNCTION at waypoint {current_waypoint_idx}")

        # Check if in junction
        if (not self.in_junction and
                current_waypoint_idx >= self.junction_entry_idx and
                current_waypoint_idx <= self.junction_exit_idx):
            self.in_junction = True
            print(f"ENTERED JUNCTION at waypoint {current_waypoint_idx}")

        # Check if passed junction
        if (not self.passed_junction and
                current_waypoint_idx > self.junction_exit_idx):
            self.passed_junction = True
            self.in_junction = False
            print(f"PASSED JUNCTION at waypoint {current_waypoint_idx}")

    def _evaluate_success(self):
        """Evaluate if the junction was successfully navigated."""
        last_wp = self.stanley_controller.route_waypoints[-1].transform.location #get last waypoint
        _, _, _, cte = self.stanley_controller.compute_steering()
        vehicle_location = self.stanley_controller.vehicle.get_location()
        self.max_cte = max(self.max_cte, abs(cte))

        current_end_point_distance = vehicle_location.distance(last_wp)
        self.min_endpoint_distance = min(current_end_point_distance, self.min_endpoint_distance) #save shortest distance to endpoint

        if self.min_endpoint_distance < 1 and self.max_cte < 2: #if the vehicle is close to the endpoint and the max cte is low, it was success
            self.junction_success = True

        if current_end_point_distance > self.min_endpoint_distance and self.junction_success: #if vehicle is moving away from the endpoint but we got success before, it is "passing" the endpoint, then we quit
            self.junction_success = True
            print(f"Junction navigation successful! (CTE: {cte})")
            return True

        return False

    def get_results(self):
        """Get the experiment results."""
        return {
            'tick_delay': self.tick_delay,
            'command_duration': self.command_duration,
            'test_command': self.test_command,
            'junction_entry_idx': self.junction_entry_idx,
            'junction_exit_idx': self.junction_exit_idx,
            'command_before_junction': self.command_before_junction,
            'junction_command': self.junction_command,
            'junction_command_applied': self.junction_command_applied,
            'early_command_applied': self.early_command_applied,
            'command_released': self.command_released,
            'success': self.junction_success,
            'max_cte': self.max_cte,
            'waypoint_deviation': self.waypoint_deviation,
            'command_log': self.command_log
        }


class JunctionTimingExperiment:
    """
    Runner for junction timing experiments.
    """

    def __init__(self, base_runner):
        """Initialize with reference to base experiment runner."""
        self.base_runner = base_runner

        # Save original thresholds (to restore after experiment)
        self.original_cte_threshold = base_runner.config.cte_threshold
        self.original_cte_threshold_junction = base_runner.config.cte_threshold_junction
        self.original_avg_speed_threshold = base_runner.config.avg_speed_threshold

        # Results storage
        self.results = []

    def setup(self):
        """Prepare for experiment by disabling auto-reset."""
        # Disable auto-reset to observe full junction behavior
        self.base_runner.config.cte_threshold = float('inf')
        self.base_runner.config.cte_threshold_junction = float('inf')
        self.base_runner.config.avg_speed_threshold = 0.0

        # Initialize command injector
        self.base_runner.command_injector = JunctionCommandInjector(self.base_runner.stanley_controller)

    def teardown(self):
        """Restore original settings."""
        self.base_runner.config.cte_threshold = self.original_cte_threshold
        self.base_runner.config.cte_threshold_junction = self.original_cte_threshold_junction
        self.base_runner.config.avg_speed_threshold = self.original_avg_speed_threshold

    def run_experiments(self, test_commands, tick_delays, command_durations=None, runs_per_config=1):
        """
        Run timing experiments for all commands, delays, and durations.

        Args:
            test_commands: List of commands to test (e.g., ["LEFT", "RIGHT", "STRAIGHT"])
            tick_delays: List of tick delays to test (can include negative values)
            command_durations: List of command durations to test (None = only test infinite duration)
            runs_per_config: Number of runs per configuration
        """
        # Create results directory
        results_dir = os.path.join(self.base_runner.config.fine_tuning_dir, "junction_timing_results")
        os.makedirs(results_dir, exist_ok=True)

        # If command_durations is None, only test with infinite duration (0)
        if command_durations is None:
            command_durations = [0]  # 0 = infinite duration

        total_configs = len(test_commands) * len(tick_delays) * len(command_durations)
        total_runs = total_configs * runs_per_config
        current_run = 0

        try:
            for command in test_commands:
                # Create command directory
                command_dir = os.path.join(results_dir, f"command_{command}")
                os.makedirs(command_dir, exist_ok=True)

                for delay in tick_delays:
                    # Create delay directory
                    delay_dir = os.path.join(command_dir, f"delay_{delay}")
                    os.makedirs(delay_dir, exist_ok=True)

                    for duration in command_durations:
                        # Create duration directory if testing multiple durations
                        if len(command_durations) > 1:
                            test_dir = os.path.join(delay_dir, f"duration_{duration}")
                        else:
                            test_dir = delay_dir
                        os.makedirs(test_dir, exist_ok=True)

                        for run in range(runs_per_config):
                            current_run += 1
                            direction = "EARLY" if delay < 0 else "LATE"
                            duration_info = f", Duration={duration}" if duration > 0 else ""
                            print(
                                f"\n=== Run {current_run}/{total_runs}: Command={command}, {direction}={abs(delay)} ticks{duration_info}, Run={run + 1} ===")

                            # Initialize command injector for this test
                            self.base_runner.command_injector.initialize(delay, command, duration)

                            # Run until junction is passed
                            self._run_single_test()

                            # Collect results
                            result = self.base_runner.command_injector.get_results()
                            result['run_id'] = current_run
                            result['command_tested'] = command
                            self.results.append(result)

                            # Save individual run data
                            self._save_run_data(result, test_dir, run)

                            # Reset vehicle for next run
                            self._reset_vehicle()

            # Save and analyze all results
            self._save_all_results(results_dir)

        finally:
            # Restore original settings
            self.teardown()


    def _run_single_test(self):
        """Run a single test until junction is passed."""
        max_steps = 120  # Safety limit
        steps = 0

        # Run until junction is passed and success is evaluated
        while steps < max_steps:
            self.base_runner._process_frame()
            steps += 1

            # Check if test is complete
            if self.base_runner.command_injector._evaluate_success():
                break

        # If timed out without success evaluation, mark as failure
        if steps >= max_steps:
            self.base_runner.command_injector.junction_success = False

    def _reset_vehicle(self):
        """Reset vehicle to start of route."""
        route = self.base_runner.stanley_controller.route_waypoints
        if route and len(route) > 0:
            start_wp = route[0]
            self.base_runner.env.vehicle.set_transform(start_wp.transform)
            self.base_runner.stanley_controller.reset()

    def _save_run_data(self, result, save_dir, run_num):
        """Save data for a single run."""
        # Extract command log
        command_log = result.get('command_log', [])
        if not command_log:
            return

        # Convert to DataFrame
        df = pd.DataFrame(command_log)

        # Add experiment metadata
        df['tick_delay'] = result['tick_delay']
        df['test_command'] = result['test_command']
        df['success'] = result['success']
        df['max_cte_during_run'] = result['max_cte']
        df['waypoint_x'] = df['waypoint_x']
        df['waypoint_y'] = df['waypoint_y']

        # Save to CSV
        csv_path = os.path.join(save_dir, f"run_{run_num + 1}.csv")
        df.to_csv(csv_path, index=False)

    def _save_all_results(self, results_dir):
        """Save summarized results of all runs."""
        # Prepare summary data for all runs
        summary_data = []

        for result in self.results:
            # Extract key metrics
            summary_data.append({
                'run_id': result['run_id'],
                'command_tested': result['command_tested'],
                'tick_delay': result['tick_delay'],
                'command_duration': result.get('command_duration', 0),
                'junction_command_applied': result['junction_command_applied'],
                'early_command_applied': result.get('early_command_applied', False),
                'command_released': result.get('command_released', False),
                'success': result['success'],
                'max_cte': result.get('max_cte', 0),
                'waypoint_deviation': result['waypoint_deviation']
            })

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save summary
        csv_path = os.path.join(results_dir, "results_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Results summary saved to {csv_path}")

        # Create success rate summary by command and delay
        if 'command_duration' in summary_df.columns and len(summary_df['command_duration'].unique()) > 1:
            # If testing multiple durations, group by command, delay and duration
            success_rates = summary_df.groupby(['command_tested', 'tick_delay', 'command_duration'])[
                'success'].mean().reset_index()
        else:
            # Otherwise just group by command and delay
            success_rates = summary_df.groupby(['command_tested', 'tick_delay'])['success'].mean().reset_index()

        success_csv_path = os.path.join(results_dir, "success_rates.csv")
        success_rates.to_csv(success_csv_path, index=False)


