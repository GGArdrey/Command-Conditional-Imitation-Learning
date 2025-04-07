import pandas as pd
import numpy as np

class CommandAugmentationHelper:
    '''
    Command Augmentation Helper for CARLA to alllow for navigation command augmentation.
    This class provides methods to augment command sequences by replacing commands with random ones
    and shifting command timings. It also tracks statistics about the augmentation process.
    The augmentation process is controlled by several parameters:
    - random_command_rate: The percentage of commands to be replaced with random ones.
    - sequence_length_range: The range of lengths for the command sequences to be replaced.
    - command_timing_shift_rate: The percentage of commands to be shifted in time.
    - command_timing_shift_range: The range of shifts for the command timings.
    For the Thesis, the following parameters were used: random_command_rate= 0.0 to 0.3
    sequence_length_range= (1, 1) command_timing_shift_rate=0.0, command_timing_shift_range=(0, 0)
    '''
    def __init__(self,
                 num_commands,
                 random_command_rate=0.5,
                 sequence_length_range=(1, 1),
                 command_timing_shift_rate=0.0,
                 command_timing_shift_range=(0, 0)):

        self.num_commands = num_commands
        self.random_command_rate = max(0, random_command_rate)
        self.sequence_length_range = sequence_length_range
        self.command_timing_shift_rate = max(0, command_timing_shift_rate)
        self.command_timing_shift_range = command_timing_shift_range

        # Early check for no-op conditions
        self.do_random_replacement = (self.random_command_rate > 0 and
                                    self.sequence_length_range[1] > 0)
        self.do_timing_shift = (self.command_timing_shift_rate > 0 and
                               self.command_timing_shift_range[0] != 0 or
                               self.command_timing_shift_range[1] != 0)

        self.control_commands_list = ["LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW"]

        # Statistics tracking
        self.stats = {
            'total_sequences': 0,
            'total_samples': 0,
            'modified_samples': 0,
            'sequence_stats': [],
            'params_used': {
                'random_command_rate': random_command_rate,
                'sequence_length_range': sequence_length_range,
                'command_timing_shift_rate': command_timing_shift_rate,
                'command_timing_shift_range': command_timing_shift_range
            }
        }
        np.random.seed(42)

    def _get_random_command(self, exclude_command_idx):
        """Returns a random command index different from exclude_command_idx"""
        available_commands = [i for i in range(self.num_commands) if i != exclude_command_idx]
        return np.random.choice(available_commands)

    def _get_command_distribution(self, sequence):
        """Returns dictionary with count of each command in sequence"""
        return {i: np.sum(sequence == i) for i in range(self.num_commands)}

    def augment_sequence(self, command_sequence, sequence_id=None):
        """
        Augments samples while tracking modified indices to avoid double-counting.
        Returns original sequence if no augmentation is needed.
        """
        # Track basic statistics regardless of augmentation
        self.stats['total_sequences'] += 1
        sequence_length = len(command_sequence)
        self.stats['total_samples'] += sequence_length

        # If no augmentation needed, return original sequence with basic stats
        if not (self.do_random_replacement or self.do_timing_shift):
            sequence_stats = {
                'sequence_id': sequence_id if sequence_id is not None else self.stats['total_sequences'],
                'length': sequence_length,
                'random_replacements': 0,
                'timing_shifts': 0,
                'original_distribution': self._get_command_distribution(command_sequence),
                'final_distribution': self._get_command_distribution(command_sequence),
                'command_transitions': len(np.where(command_sequence[1:] != command_sequence[:-1])[0]),
                'modification_locations': [],
                'command_changes': [],
                'unique_samples_modified': 0,
                'modification_density': 0,
                'avg_modification_length': 0
            }
            self.stats['sequence_stats'].append(sequence_stats)
            return command_sequence.copy()

        # Proceed with augmentation as before...
        augmented_sequence = command_sequence.copy()
        modified_indices = set()

        # Initialize sequence stats
        sequence_stats = {
            'sequence_id': sequence_id if sequence_id is not None else self.stats['total_sequences'],
            'length': sequence_length,
            'random_replacements': 0,
            'timing_shifts': 0,
            'original_distribution': self._get_command_distribution(command_sequence),
            'command_transitions': len(np.where(command_sequence[1:] != command_sequence[:-1])[0]),
            'modification_locations': [],
            'command_changes': []
        }

        # 1. Random command replacements
        if self.do_random_replacement:
            num_random_samples = int(sequence_length * self.random_command_rate)
            if num_random_samples > 0:
                possible_start_indices = np.arange(sequence_length - self.sequence_length_range[0] + 1)
                start_indices = np.random.choice(possible_start_indices,
                                               size=num_random_samples,
                                               replace=False)

                for start_idx in start_indices:
                    max_possible_length = min(self.sequence_length_range[1],
                                           sequence_length - start_idx)
                    length = np.random.randint(self.sequence_length_range[0],
                                            max_possible_length + 1)

                    original_command = augmented_sequence[start_idx]
                    new_command = self._get_random_command(original_command)

                    affected_indices = set(range(start_idx, start_idx + length))
                    modified_indices.update(affected_indices)

                    augmented_sequence[start_idx:start_idx + length] = new_command
                    sequence_stats['random_replacements'] += 1

                    sequence_stats['modification_locations'].append({
                        'type': 'random',
                        'start_idx': start_idx,
                        'length': length,
                        'relative_position': start_idx / sequence_length
                    })
                    sequence_stats['command_changes'].append({
                        'from': int(original_command),
                        'to': int(new_command),
                        'length': length
                    })

        # 2. Command timing shifts
        if self.do_timing_shift:
            num_timing_samples = int(sequence_length * self.command_timing_shift_rate)
            if num_timing_samples > 0:
                transitions = np.where(command_sequence[1:] != command_sequence[:-1])[0] + 1

                if len(transitions) > 0:
                    num_shifts = min(num_timing_samples, len(transitions))
                    shift_indices = np.random.choice(transitions, size=num_shifts, replace=False)

                    for idx in shift_indices:
                        shift_amount = np.random.randint(
                            self.command_timing_shift_range[0],
                            self.command_timing_shift_range[1] + 1
                        )

                        if shift_amount != 0:
                            affected_indices = set()
                            if shift_amount > 0:
                                start_idx = max(0, idx - shift_amount)
                                augmented_sequence[start_idx:idx] = command_sequence[idx]
                                affected_indices.update(range(start_idx, idx))
                            else:
                                end_idx = min(sequence_length, idx - shift_amount)
                                augmented_sequence[idx:end_idx] = command_sequence[idx - 1]
                                affected_indices.update(range(idx, end_idx))

                            new_modifications = affected_indices - modified_indices
                            modified_indices.update(new_modifications)

                            sequence_stats['timing_shifts'] += 1
                            sequence_stats['modification_locations'].append({
                                'type': 'timing_shift',
                                'index': idx,
                                'shift_amount': shift_amount,
                                'relative_position': idx / sequence_length
                            })

        # Final statistics
        sequence_stats['unique_samples_modified'] = len(modified_indices)
        sequence_stats['final_distribution'] = self._get_command_distribution(augmented_sequence)
        sequence_stats['modification_density'] = len(modified_indices) / sequence_length
        sequence_stats['avg_modification_length'] = np.mean(
            [c['length'] for c in sequence_stats['command_changes']]) if sequence_stats['command_changes'] else 0

        self.stats['sequence_stats'].append(sequence_stats)
        self.stats['modified_samples'] += len(modified_indices)

        return augmented_sequence

    def get_statistics(self):
        """Returns enhanced statistics about the augmentation process"""
        df = pd.DataFrame(self.stats['sequence_stats'])

        # Calculate command distribution changes
        orig_dist_total = {i: 0 for i in range(self.num_commands)}
        final_dist_total = {i: 0 for i in range(self.num_commands)}
        for stats in self.stats['sequence_stats']:
            for cmd in range(self.num_commands):
                orig_dist_total[cmd] += stats['original_distribution'].get(cmd, 0)
                final_dist_total[cmd] += stats['final_distribution'].get(cmd, 0)

        # Calculate modification location statistics
        all_relative_positions = []
        for stats in self.stats['sequence_stats']:
            all_relative_positions.extend([m['relative_position'] for m in stats['modification_locations']])

        stats_str = f"""
Command Augmentation Statistics
-----------------------------
Parameters Used:
Random command rate: {self.stats['params_used']['random_command_rate']}
Sequence length range: {self.stats['params_used']['sequence_length_range']}
Timing shift rate: {self.stats['params_used']['command_timing_shift_rate']}
Timing shift range: {self.stats['params_used']['command_timing_shift_range']}

Overall Statistics:
Total sequences: {self.stats['total_sequences']}
Total samples: {self.stats['total_samples']}
Uniquely modified samples: {self.stats['modified_samples']} ({self.stats['modified_samples']/max(1, self.stats['total_samples'])*100:.2f}%)

Command Distribution Changes:
Command | Original Count | Final Count | Difference | % Change
"""
        for cmd in range(self.num_commands):
            orig = orig_dist_total[cmd]
            final = final_dist_total[cmd]
            diff = final - orig
            pct_change = (diff/orig*100) if orig > 0 else float('inf')
            stats_str += f"  {self.control_commands_list[cmd]}  |  {orig:8d}     |  {final:8d}   |  {diff:6d}   |  {pct_change:6.1f}%\n"
        
        return stats_str
