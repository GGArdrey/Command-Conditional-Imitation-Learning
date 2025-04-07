import inspect
import os
import logging
import numpy as np
import tensorflow as tf

from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, Activation, Lambda, Concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from CommandAugmentationHelper import CommandAugmentationHelper


@tf.keras.utils.register_keras_serializable(package="custom")
def stack_branches_func(outputs):
    # Stacks each command-specific branch along axis=1
    return tf.stack(outputs, axis=1)


@tf.keras.utils.register_keras_serializable(package="custom")
def mask_outputs(args):
    # Applies command_one_hot as a mask to select the correct branch output
    branches, cmds = args
    cmds_expanded = tf.expand_dims(cmds, axis=-1)
    return tf.reduce_sum(branches * cmds_expanded, axis=1)


@tf.function(jit_compile=True)
def create_gaussian_kernel(size, sigma):
    """
    Creates and returns a 2D Gaussian kernel of shape (size, size).
    The kernel is normalized so that its values sum to 1.
    """
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd.")

    ax = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= tf.reduce_sum(kernel)
    return kernel

@tf.function(jit_compile=True)
def apply_gaussian_blur(image):
    """
    Applies a 5x5 Gaussian blur with sigma=1 to the image using depthwise convolution.
    Returns the blurred image.
    """
    kernel = create_gaussian_kernel(5, 1.0)  # 5x5 kernel, sigma=1
    kernel = tf.expand_dims(kernel, axis=-1)  # Shape: [k, k, 1]
    in_channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel, [1, 1, in_channels])  # Shape: [k, k, in_channels]
    kernel = tf.expand_dims(kernel, axis=-1)  # Shape: [k, k, in_channels, 1]

    blurred = tf.nn.depthwise_conv2d(
        image[tf.newaxis, ...],
        filter=kernel,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    # Remove the batch dimension
    blurred = tf.squeeze(blurred, axis=0)
    return blurred
@tf.function(jit_compile=True)
def resize_and_crop_image_tf(image, target_width, target_height):
    """
    Resizes the image to target_width x target_height
    """
    image = tf.image.resize(image, (target_height, target_width))
    return image

def _apply_random_black_patch(image, patch_size=12, max_patches=3):
    """
    Places a single black patch of size patch_size x patch_size
    at a random location in the image.
    """
    image = tf.cast(image, tf.float32)  # Ensure image is in float format
    height, width, channels = image.shape

    def apply_patches(image_np):
        for _ in range(np.random.randint(0, max_patches + 1)):
            top = np.random.randint(0, height - patch_size)
            left = np.random.randint(0, width - patch_size)
            image_np[top:top + patch_size, left:left + patch_size, :] = 0  # Black patch
        return image_np

    augmented_image = tf.numpy_function(apply_patches, [image], tf.float32)
    augmented_image.set_shape(image.shape)  # Retain the shape for downstream compatibility
    return augmented_image


def augment_data(image):
    """
    Applies random augmentations (brightness, contrast, hue) to an image.
    Then adds up to 3 random black patches.
    """
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)



    num_patches = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    for _ in range(num_patches):
        image = _apply_random_black_patch(image, patch_size=12)
    return image


def load_image_from_disk(image_path):
    """
    Loads and decodes an image from the given path, returning a tensor with 3 channels (RGB).
    """
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    return image


def preprocess_image(image, target_width, target_height):
    """
    Preprocesses a single image:
    1) Convert to float32
    2) Resize/crop
    3) Apply Gaussian blur
    4) Convert to YUV
    5) Scale to [0, 1]
    """
    image = tf.cast(image, tf.float32)
    image = resize_and_crop_image_tf(image, target_width, target_height)
    image = apply_gaussian_blur(image)
    image = image / 255.0
    return image


# -------------------------------------------------------------------------
class BasePilotNetRaspiCar:
    """
    Base class for PilotNet in RaspiCar usage. Very similar to PilotNetJunctionBase, but w/o DANN and adapters.
    Handling data loading, preprocessing, logging directories,
    and any shared utility functions. The actual loss function is delegated
    to a subclass via _compute_loss.
    """

    def __init__(self, params, create_directories=True):
        """
        Initializes the base class with provided parameters and sets up directories.
        """
        self.params = params
        self.data_dirs = params.get("data_dirs", None)
        self.save_dir = params.get("save_dir", None)
        self.target_width = params.get("target_width", 200)
        self.target_height = params.get("target_height", 66)
        self.batch_size = params.get("batch_size", 128)
        self.epochs = params.get("epochs", 50)
        self.initial_learning_rate = params.get("initial_learning_rate", 1e-4)
        self.model_path = params.get("model_path", None)  # if wanting to load model
        self.boundaries = np.array(params.get("boundaries", [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]))  # for steering classification
        self.use_command_weights = params.get("use_command_weights", False)
        self.use_steering_weights = params.get("use_steering_weights", False)

        self.control_commands_list = ["LEFT", "RIGHT", "STRAIGHT"]
        self.control_command_to_index = {cmd: idx for idx, cmd in enumerate(self.control_commands_list)}
        self.total_control_commands = len(self.control_commands_list)

        # By default no command augmentation
        self.command_augmentor = CommandAugmentationHelper(
            self.total_control_commands,
            random_command_rate=self.params.get('random_command_rate', 0),
            sequence_length_range=self.params.get('sequence_length_range', (0, 0)),
            command_timing_shift_rate=self.params.get('command_timing_shift_rate', 0),
            command_timing_shift_range=self.params.get('command_timing_shift_range', (0, 0))
        )

        # Prepare logging/checkpoint directories
        self.log_dir_base = None
        if create_directories:
            self._prepare_directories()

    def _prepare_directories(self):
        """
        Creates directories for logging and checkpoints using a datetime stamp.
        """
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.log_dir_base = os.path.join(self.save_dir, timestamp, "logs")
        self.checkpoint_dir = os.path.join(self.save_dir, timestamp, "checkpoints")
        os.makedirs(self.log_dir_base, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def map_label_to_class(self, label):
        """
        Maps a continuous label (e.g. steering angle) to the nearest class index
        based on self.boundaries.
        """
        differences = np.abs(label - self.boundaries)
        return np.argmin(differences)

    def parse_filename(self, filename):
        """
        Parses the filename to extract control command and label.
        Expected format: {int}_{COMMAND}_{float}.jpg
        e.g. "1_LEFT_0.2.jpg" => control_command=LEFT, label=0.2
        Returns a tuple (float_label, class_label, control_command_idx)
        or None if parsing fails.
        """
        try:
            name_wo_ext = os.path.splitext(filename)[0]
            split_name = name_wo_ext.split("_")
            control_command_str = split_name[1]
            label = float(split_name[2])
            class_label = self.map_label_to_class(label)
            control_command_index = self.control_command_to_index.get(control_command_str, None)
            return label, class_label, control_command_index
        except (IndexError, ValueError, KeyError):
            print(f"Error parsing filename {filename}")
            return None

    def load_data(self, data_dirs=None):
        """
        Loads data while maintaining sequence information. Each sequence represents
        a continuous recording that should be processed as a unit for command augmentation.
        """
        if data_dirs is None:
            data_dirs = self.data_dirs

        # Lists to store all data
        all_image_paths = []
        all_class_labels = []
        all_float_labels = []
        all_control_commands = []
        sequence_lengths = []  # Track sequence lengths for logging

        # Process each domain directory
        for folder_path, domain_label in data_dirs.items():
            # Process each sequence directory
            for sequence_name in os.listdir(folder_path):
                sequence_path = os.path.join(folder_path, sequence_name)
                if os.path.isdir(sequence_path):
                    # Prepare sequence data
                    sequence_files = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
                    sequence_files.sort(key=lambda x: int(x.split('_')[0]))

                    # Lists for current sequence
                    seq_image_paths = []
                    seq_class_labels = []
                    seq_float_labels = []
                    seq_commands = []

                    # Process all files in sequence
                    for file in sequence_files:
                        result = self.parse_filename(file)
                        if result is not None:
                            label, cls_label, cmd_idx = result
                            image_path = os.path.join(sequence_path, file)
                            seq_image_paths.append(image_path)
                            seq_class_labels.append(cls_label)
                            seq_float_labels.append(label)
                            seq_commands.append(cmd_idx)

                    if seq_image_paths:  # Only process non-empty sequences
                        # Convert to numpy arrays for easier manipulation
                        seq_commands = np.array(seq_commands)

                        # Augment commands for this sequence
                        augmented_commands = self.command_augmentor.augment_sequence(seq_commands)

                        # Store sequence data
                        all_image_paths.extend(seq_image_paths)
                        all_class_labels.extend(seq_class_labels)
                        all_float_labels.extend(seq_float_labels)
                        all_control_commands.extend(augmented_commands)

                        sequence_lengths.append(len(seq_image_paths))

        # Log sequence statistics if it has self.log_dir_base
        if self.log_dir_base:
            log_file_path = os.path.join(self.log_dir_base, "sequence_stats.txt")
            with open(log_file_path, "w") as f:
                f.write(f"Total sequences: {len(sequence_lengths)}\n")
                f.write(f"Average sequence length: {np.mean(sequence_lengths):.2f}\n")
                f.write(f"Min sequence length: {min(sequence_lengths)}\n")
                f.write(f"Max sequence length: {max(sequence_lengths)}\n")
                f.write(f"\nCommand augmentation statistics:\n")
                f.write(self.command_augmentor.get_statistics())

        # Convert to numpy arrays
        return (
            all_image_paths,
            np.array(all_class_labels),
            np.array(all_float_labels, dtype=np.float32),
            np.array(all_control_commands, dtype=np.int32)
        )

    def create_tf_dataset(self, image_paths, class_labels, command_inputs, enable_augment_data=False):
        """
        Creates and returns a tf.data.Dataset object that yields
        (({"image_input": image, "command_input": one_hot_cmd})).
        """
        cmd_one_hot = to_categorical(command_inputs, num_classes=self.total_control_commands).astype("float32")
        cls_one_hot = to_categorical(class_labels, num_classes=len(self.boundaries)).astype("float32")

        ds = tf.data.Dataset.from_tensor_slices((image_paths, cmd_one_hot, cls_one_hot))

        def _map_fn(img_path, cmd, cls_label):
            image = load_image_from_disk(img_path)
            image = preprocess_image(image, self.target_width, self.target_height)

            if enable_augment_data:
                # Apply standard augmentations first
                image = augment_data(image)

                # Random horizontal flip with 50% probability
                do_flip = tf.random.uniform(shape=[], minval=0, maxval=1) < 0.5

                def flip_transformations():
                    # Flip the image horizontally
                    flipped_image = tf.image.flip_left_right(image)

                    # Swap LEFT and RIGHT commands (one-hot encoded)
                    # LEFT is index 0, RIGHT is index 1
                    flipped_cmd = tf.concat([
                        cmd[1:2],  # RIGHT becomes LEFT
                        cmd[0:1],  # LEFT becomes RIGHT
                        cmd[2:]  # Keep STRAIGHT and any other commands
                    ], axis=0)

                    # Invert steering class labels
                    # For symmetric boundaries around zero, we can reverse the one-hot vector
                    num_classes = tf.shape(cls_label)[0]
                    reverse_indices = num_classes - 1 - tf.range(num_classes)
                    flipped_cls = tf.gather(cls_label, reverse_indices)

                    return flipped_image, flipped_cmd, flipped_cls

                def identity_transformations():
                    return image, cmd, cls_label

                # Apply flip transformations if do_flip is true
                image, cmd, cls_label = tf.cond(
                    do_flip,
                    flip_transformations,
                    identity_transformations
                )

            x_dict = {
                "image_input": image,  # shape [H, W, 3]
                "command_input": cmd  # shape [num_commands]
            }
            y_dict = {
                "class_output": cls_label,  # shape [num_classes]
                "command_output": cmd  # shape [num_commands], repeat for loss calculation
            }
            return x_dict, y_dict

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE).cache()
        ds = ds.shuffle(5000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _write_log_pre_training(self, model, total_imgs, train_count, val_count, test_count, command_weights):
        """
        Logs a summary of the model and dataset setup to info.txt in self.log_dir_base.
        Also logs all self.params and boundaries, etc.
        """
        log_file_path = os.path.join(self.log_dir_base, "info.txt")
        with open(log_file_path, "a") as f:
            f.write("MODEL SUMMARY:\n")
            model.summary(print_fn=lambda x: f.write(x + "\n"))
            f.write("\nDATASET INFO:\n")
            f.write(f"Total images: {total_imgs}\n")
            f.write(f"Train count: {train_count}, Val count: {val_count}, Test count: {test_count}\n")
            f.write(f"Boundaries: {self.boundaries.tolist()}\n")
            f.write(f"Control Commands: {self.control_commands_list}\n")
            f.write(f"Command Weights: {command_weights}\n")
            f.write("\nPARAMETERS:\n")
            for k, v in self.params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n\n\n")
            f.write(inspect.getsource(self.build_model))
            f.write("\n\n\n")
            f.write(inspect.getsource(self._compute_loss))

    def calculate_command_weights(self, train_command_inputs):
        """
        Calculates and returns per-command weights to rebalance
        potential class imbalance among commands.
        """
        unique, counts = np.unique(train_command_inputs, return_counts=True)
        total = len(train_command_inputs)
        num_cmds = self.total_control_commands
        weights = np.zeros(num_cmds, dtype=np.float32)
        for cmd_idx, cnt in zip(unique, counts):
            weights[cmd_idx] = total / (cnt * num_cmds)
        # For any command not present, default to 1.0
        for i in range(num_cmds):
            if weights[i] == 0:
                weights[i] = 1.0
        return weights

    #  calculate steering class weights
    def calculate_steering_weights(self, class_labels):
        class_counts = np.bincount(class_labels)
        total = len(class_labels)
        return total / (len(class_counts) * (class_counts + 1e-8))  # Avoid division by zero



    def build_model(self, use_adapters=False):
        """
        Must be overridden by subclasses to return a compiled Keras model.
        The model's inputs must be named: "image_input", "command_input"
        The outputs must be named: "class_output"
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def _compute_loss(self, y_true, y_pred, command_weights=None, ordinal_weight=1.0):
        """
        Returns (total_loss, steering_loss).
        Implements ordinal classification loss that considers the distance between predicted
        and true class indices, as well as command weighting.

        Args:
            y_true: Dictionary containing ground truth values
            y_pred: Dictionary containing predictions
            command_weights: Optional weights for different commands
            ordinal_weight: Weight for the ordinal component of the loss (default=1.0)
        """
        # Get steering classification labels and predictions
        steering_label = y_true["class_output"]
        steering_pred = y_pred["class_output"]

        # Standard categorical cross-entropy loss
        steering_ce = tf.keras.losses.categorical_crossentropy(steering_label, steering_pred)

        # # --- Add ordinal distance-weighted component ---
        # # Get the true and predicted class indices
        # true_class_idx = tf.argmax(steering_label, axis=1, output_type=tf.int32)
        # pred_class_idx = tf.argmax(steering_pred, axis=1, output_type=tf.int32)
        #
        # # Calculate the absolute distance between predicted and true indices
        # # This represents how "far" the prediction is from the truth in the ordinal space
        # class_distance = tf.cast(tf.abs(pred_class_idx - true_class_idx), tf.float32)
        #
        # # Apply a scaling function to the distance
        # # You could use different functions here, such as:
        # # - Linear: distance
        # # - Quadratic: distance^2
        # # - Square root: sqrt(distance)
        # # Quadratic penalizes larger errors more severely
        # ordinal_penalty = tf.square(class_distance)
        #
        # # Combine standard cross-entropy with ordinal penalty
        # combined_loss = steering_ce + ordinal_weight * ordinal_penalty
        #
        # # Weight loss by command if command_weights is provided
        # if command_weights is not None:
        #     cmd_one_hot = y_true["command_output"]
        #     per_sample_cmd_weight = tf.reduce_sum(command_weights * cmd_one_hot, axis=1)
        #     combined_loss *= per_sample_cmd_weight

        # Calculate final loss
        steering_loss = tf.reduce_mean(steering_ce) #TODO change back to combined_loss if used
        total_loss = steering_loss

        return total_loss, steering_loss

    def _create_callbacks(self):
        """
        Creates standard callbacks for logging, checkpoints, and LR scheduling.
        """
        tensorboard_cb = TensorBoard(log_dir=self.log_dir_base, histogram_freq=1)
        checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.keras"),
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        )
        reduce_lr_cb = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1
        )
        return [tensorboard_cb, checkpoint_cb, reduce_lr_cb]

    @tf.function
    def _train_step(self, x, y_true, model, optimizer,
                    steering_acc_metric,
                    steering_loss_metric,
                    total_loss_metric,
                    command_weights):
        """
        Single training step updating model weights via gradient descent.
        Also logs separate steering, and total losses for the step.
        """
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            total_loss, steering_loss_val = self._compute_loss(
                y_true,
                y_pred,
                command_weights
            )

        gradients = tape.gradient(total_loss, model.trainable_variables)


        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update metrics
        steering_acc_metric.update_state(y_true["class_output"], y_pred["class_output"])
        steering_loss_metric.update_state(steering_loss_val)
        total_loss_metric.update_state(total_loss)

    @tf.function
    def _val_step(self, x, y_true, model,
                  steering_acc_metric,
                  steering_loss_metric,
                  total_loss_metric,
                  command_weights):
        """
        Single validation step without weight updates.
        Logs separate steering, domain, and total losses for analysis.
        """
        y_pred = model(x, training=False)
        total_loss, steering_loss_val = self._compute_loss(
            y_true,
            y_pred,
            command_weights
        )
        steering_acc_metric.update_state(y_true["class_output"], y_pred["class_output"])
        steering_loss_metric.update_state(steering_loss_val)
        total_loss_metric.update_state(total_loss)

    def _train_loop(self, model, train_ds, val_ds, command_weights):
        """
        Custom training loop with metrics and Keras callbacks,
        including separate steering, domain, and total losses in logs.
        """
        callbacks = self._create_callbacks()
        for cb in callbacks:
            cb.set_model(model)
            cb_params = {"epochs": self.epochs, "verbose": 1, "steps": None}
            cb.set_params(cb_params)
            cb.on_train_begin()

        # Metrics for training
        train_steeering_acc = tf.keras.metrics.CategoricalAccuracy(name="train_steering_acc")
        train_steering_loss = tf.keras.metrics.Mean(name="train_steering_loss")
        train_total_loss = tf.keras.metrics.Mean(name="train_loss")

        # Metrics for validation
        val_steering_acc = tf.keras.metrics.CategoricalAccuracy(name="val_steering_acc")
        val_steering_loss = tf.keras.metrics.Mean(name="val_steering_loss")
        val_total_loss = tf.keras.metrics.Mean(name="val_loss")

        steps_per_epoch = train_ds.cardinality().numpy()
        val_steps = val_ds.cardinality().numpy()

        # Calculate steps per epoch
        steps_per_epoch = len(train_ds)


        for epoch in range(self.epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            optimizer = model.optimizer

            # Reset training metrics
            train_steeering_acc.reset_state()
            train_steering_loss.reset_state()
            train_total_loss.reset_state()

            # Training loop
            pbar_train = tqdm(train_ds, total=steps_per_epoch, desc=f"Epoch {epoch + 1} [Train]")
            step_train = 0
            for x_batch, y_batch in pbar_train:
                self._train_step(
                    x_batch,
                    y_batch,
                    model,
                    optimizer,
                    train_steeering_acc,
                    train_steering_loss,
                    train_total_loss,
                    command_weights,
                )
                logs = {
                    "loss": train_total_loss.result().numpy(),
                    "steering_loss": train_steering_loss.result().numpy(),
                    "steering_accuracy": train_steeering_acc.result().numpy(),
                }
                for cb in callbacks:
                    cb.on_train_batch_end(step_train, logs=logs)

                pbar_train.set_postfix({
                    "loss": f"{train_total_loss.result():.4f}",
                    "steering_loss": f"{train_steering_loss.result():.4f}",
                    "steering_accuracy": f"{train_steeering_acc.result():.4f}",
                })
                step_train += 1
                if step_train >= steps_per_epoch:
                    break

            # Reset validation metrics
            val_steering_acc.reset_state()
            val_steering_loss.reset_state()
            val_total_loss.reset_state()

            # Validation loop
            pbar_val = tqdm(val_ds, total=val_steps, desc=f"Epoch {epoch + 1} [Val]")
            step_val = 0
            for x_val, y_val in pbar_val:
                self._val_step(
                    x_val,
                    y_val,
                    model,
                    val_steering_acc,
                    val_steering_loss,
                    val_total_loss,
                    command_weights,
                )
                pbar_val.set_postfix({
                    "val_loss": f"{val_total_loss.result():.4f}",
                    "val_steering_loss": f"{val_steering_loss.result():.4f}",
                    "val_steering_acc": f"{val_steering_acc.result():.4f}",
                })
                step_val += 1
                if step_val >= val_steps:
                    break

            # Logs to pass to callbacks
            epoch_logs = {
                "loss": train_total_loss.result().numpy(),
                "steering_loss": train_steering_loss.result().numpy(),
                "steering accuracy": train_steeering_acc.result().numpy(),
                "val_loss": val_total_loss.result().numpy(),
                "val_steering_loss": val_steering_loss.result().numpy(),
                "val_accuracy": val_steering_acc.result().numpy(),
            }
            for cb in callbacks:
                cb.on_epoch_end(epoch, epoch_logs)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"total_loss={train_total_loss.result():.4f}, "
                f"steering_loss={train_steering_loss.result():.4f}, "
                f"acc={train_steeering_acc.result():.4f} | "
                f"val_loss={val_total_loss.result():.4f}, "
                f"val_steering_loss={val_steering_loss.result():.4f}, "
                f"val_acc={val_steering_acc.result():.4f}"
            )

        # End training
        for cb in callbacks:
            cb.on_train_end()

    def train(self):
        """
        Main training procedure:
        1) Load data
        2) Split into train/val/test
        3) Calculate command & domain weights
        4) Build model
        5) Log info
        6) Custom loop training
        7) Evaluate on test set
        """
        # 1) Load
        image_paths, class_labels, float_labels, control_commands = self.load_data()
        total_imgs = len(image_paths)


        x_train, x_val, cls_train, cls_val, cmd_train, cmd_val = train_test_split(
            image_paths, class_labels, control_commands, test_size=0.2, random_state=42, shuffle=True, stratify=control_commands
        )

        # Create tf.data.Datasets
        train_ds = self.create_tf_dataset(x_train, cls_train, cmd_train, enable_augment_data=True)
        val_ds = self.create_tf_dataset(x_val, cls_val, cmd_val, enable_augment_data=False)

        # 3) Calculate command weights from TRAIN command labels
        command_weights = self.calculate_command_weights(cmd_train) if self.use_command_weights else None
        steering_weights = self.calculate_steering_weights(cls_train) if self.use_steering_weights else None
        print("Command Weights:", command_weights)
        print("Steering Weights:", steering_weights)

        steps_per_epoch = len(train_ds)

        # Use standard Adam with the schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)


        model = self.build_model(use_adapters=False)
        model.optimizer = optimizer
        self._write_log_pre_training(model, total_imgs, len(x_train), len(x_val),
                                     0, command_weights)
        self._train_loop(model, train_ds, val_ds, command_weights)



        # Evaluate on test set (with separate steering logging for final metrics)
        test_steering_loss_metric = tf.keras.metrics.Mean()
        test_total_loss_metric = tf.keras.metrics.Mean()
        test_steering_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        for x_test_batch, y_test_batch in val_ds:
            y_pred_test = model(x_test_batch, training=False)
            total_loss, steering_loss_val = self._compute_loss(
                y_test_batch,
                y_pred_test,
                command_weights
            )
            test_total_loss_metric.update_state(total_loss)
            test_steering_loss_metric.update_state(steering_loss_val)
            test_steering_acc_metric.update_state(y_test_batch["class_output"], y_pred_test["class_output"])

        print(f"Test Results:\n"
              f"    Test Total Loss: {test_total_loss_metric.result().numpy():.4f}\n"
              f"    Test Steering Loss: {test_steering_loss_metric.result().numpy():.4f}\n"
              f"    Test Steering Accuracy: {test_steering_acc_metric.result().numpy():.4f}\n")

        # Write overall TP, FP, FN, TN to info.txt
        log_file_path = os.path.join(self.log_dir_base, "info.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Test Results:\n"
                           f"    Test Total Loss: {test_total_loss_metric.result().numpy():.4f}\n"
                           f"    Test Steering Loss: {test_steering_loss_metric.result().numpy():.4f}\n"
                           f"    Test Steering Accuracy: {test_steering_acc_metric.result().numpy():.4f}\n")

