class CommandInjector:
    '''
    This class is responsible for injecting commands into the closed loop testing.
    It will inject commands at a fixed rate, and will also handle junctions
    by checking if the vehicle is in a junction segment.
    The injected command will be different from the true command, and will be injected in a round robin fashion from
    the list of all commands. For the thesis the commands are injected for a fixed number of frames (50) per command
    before the next command is injected.
    '''
    def __init__(self, stanley_controller, num_frames_to_inject=50):
        self.stanley_controller = stanley_controller
        self.num_frames_to_inject = num_frames_to_inject
        self.all_commands = ["LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW"]
        self.current_command_idx = 0
        self.frame_counter = 0

        # Special case - if num_frames_to_inject is 0, disable injection
        self.injection_enabled = (num_frames_to_inject > 0)

        self.junction_segments = self._find_junction_segments()

    def _find_junction_segments(self):
        """
        Analyze the route to find junction segments.
        Returns list of tuples: [(start_idx, end_idx, command), ...]
        """
        junction_segments = []
        start_idx = None
        current_command = None

        for i, (wp, cmd) in enumerate(zip(self.stanley_controller.route_waypoints,
                                          self.stanley_controller.route_road_options)):
            if wp.is_junction:
                if start_idx is None:
                    start_idx = i
                    current_command = cmd
            elif start_idx is not None:
                # Junction ended
                junction_segments.append((start_idx, i - 1, current_command))
                start_idx = None
                current_command = None

        # Handle case where route ends in junction
        if start_idx is not None:
            junction_segments.append((start_idx, len(self.stanley_controller.route_waypoints) - 1, current_command))

        return junction_segments

    def get_injected_command(self, true_command, current_waypoint_idx, vehicle_location=None, vehicle_speed=None):
        # If injection is disabled, always return true command
        if not self.injection_enabled:
            return true_command

        for start_idx, end_idx, junction_command in self.junction_segments:
            if start_idx <= current_waypoint_idx <= end_idx:
                junction_middle = int((end_idx - start_idx) * 0.67) + start_idx # 67% through the junction
                if junction_middle >= current_waypoint_idx: # if we are before the middle of the junction
                    return true_command

        self.frame_counter += 1
        if self.frame_counter >= self.num_frames_to_inject:
            self.frame_counter = 0
            self.current_command_idx = (self.current_command_idx + 1) % len(self.all_commands)

        injected_command = self.all_commands[self.current_command_idx]
        return injected_command if injected_command != true_command else self.all_commands[
            (self.current_command_idx + 1) % len(self.all_commands)]