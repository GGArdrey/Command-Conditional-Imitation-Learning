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
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns

import pandas as pd
import math

from CommandAugmentationHelper import CommandAugmentationHelper



# Utility Functions

# Custom gradient reversal (for the domain-adversarial part)
# Not used in the Thesis
@tf.custom_gradient
def reverse_gradient(x, hp_lambda=1.0):
    """
    Forward pass: identity
    Backward pass: multiply gradient by -hp_lambda
    """
    def grad(dy):
        return -hp_lambda * dy, 0.0

    return x, grad


class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, hp_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = tf.Variable(hp_lambda, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        return reverse_gradient(inputs, self.hp_lambda)


# Helper Lambdas for multi-command branching
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
def resize_and_crop_image_tf(image, target_width, target_height, bottom_offset=10):
    """
    Resizes the image to 'target_width' (keeping aspect ratio) so that the full original width
    is preserved (scaled down to target_width). Then, it crops the bottom part of the resized image
    (with an optional bottom_offset) to obtain 'target_height'.

    Modified to match the behavior of the first implementation:
    - Calculates y_start as (new_height - target_height)
    - Applies bottom_offset by subtracting from y_start
    - Removes zero clamping to ensure consistent offset
    """
    shape = tf.shape(image)
    orig_height = tf.cast(shape[0], tf.float32)
    orig_width = tf.cast(shape[1], tf.float32)

    # Compute scaling factor based on the full original width
    scaling_factor = target_width / orig_width
    new_height = tf.cast(orig_height * scaling_factor, tf.int32)

    # Resize the image; this ensures the full width is used, scaled to target_width
    image = tf.image.resize(image, [new_height, target_width])

    # Ensure new_height is large enough for the crop
    assert_op = tf.debugging.assert_greater_equal(
        new_height,
        target_height,
        message=f"Resized image height ({new_height}) is smaller than the target height ({target_height})."
    )

    with tf.control_dependencies([assert_op]):
        # Match first implementation's cropping behavior:
        # 1. Calculate base y_start from bottom
        # 2. Apply bottom_offset by subtracting
        y_start = new_height - target_height
        y_start = y_start - bottom_offset

        image = tf.image.crop_to_bounding_box(
            image,
            y_start,
            0,
            target_height,
            target_width
        )
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

    # TODO ENABLE/DISABLE
    num_patches = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
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


def preprocess_image(image, target_width, target_height, bottom_offset=10):
    """
    Preprocesses a single image:
    1) Convert to float32
    2) Resize/crop
    3) Apply Gaussian blur
    4) Convert to YUV
    5) Scale to [0, 1]
    """
    image = tf.cast(image, tf.float32)
    image = resize_and_crop_image_tf(image, target_width, target_height, bottom_offset)
    image = apply_gaussian_blur(image)
    image = tf.image.rgb_to_yuv(image)
    # Note: scales YUV channels by /255, but U/V ranges are typically [-128, 127]
    # Normalize Y to [0,1] and U/V to [-0.5, 0.5].
    # image = (image - [0, 128, 128]) / [255.0, 255.0, 255.0]
    image = image / 255.0
    return image


# -------------------------------------------------------------------------
class BasePilotNet:
    """
    Base class for handling data loading, preprocessing, logging directories,
    and any shared utility functions. The actual loss function is delegated
    to a subclass via _compute_loss for flexibility (e.g. DANN or standard).

    This implementation also contains Domain Adversarial Neural Network (DANN) support and ability to train "adapters"
    for domain adaptation. The model is expected to be built in subclasses. The thesis does not include DANN or adapters.
    However this implementation details are kept for future use.

    If you want to have a cleaner implementation, please refer to PilotNetJunctionRaspiCarBase.py
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
        self.domain_loss_weight = params.get("domain_loss_weight", 1.0)  # for DANN if needed
        self.model_path = params.get("model_path", None)  # if wanting to load model
        self.boundaries = np.array(params.get("boundaries", [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, -0.02, 0,
             0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]))  # for steering classification
        self.use_domain_weights = params.get("use_domain_weights", True)
        self.use_command_weights = params.get("use_command_weights", True)
        self.use_steering_weights = params.get("use_steering_weights", False)
        self.use_adapters = params.get("use_adapters", False)
        self.adapter_training_phase = params.get("adapter_training_phase", False)
        self.control_commands_list = ["LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW"]
        self.control_command_to_index = {cmd: idx for idx, cmd in enumerate(self.control_commands_list)}
        self.total_control_commands = len(self.control_commands_list)
        if self.data_dirs:
            self.num_domains = len(set(self.data_dirs.values()))

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
        all_domain_labels = []
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
                        all_domain_labels.extend([domain_label] * len(seq_image_paths))
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
            np.array(all_domain_labels, dtype=np.int32),
            np.array(all_control_commands, dtype=np.int32)
        )

    def create_tf_dataset(self, image_paths, class_labels, domain_labels, command_inputs, enable_augment_data=False):
        """
        Creates and returns a tf.data.Dataset object that yields
        (({"image_input": image, "command_input": one_hot_cmd}),
         {"class_output": one_hot_cls, "domain_output": domain_label}).
        """
        cmd_one_hot = to_categorical(command_inputs, num_classes=self.total_control_commands).astype("float32")
        cls_one_hot = to_categorical(class_labels, num_classes=len(self.boundaries)).astype("float32")
        domain_one_hot = to_categorical(domain_labels, num_classes=self.num_domains).astype("float32")

        ds = tf.data.Dataset.from_tensor_slices((image_paths, cmd_one_hot, cls_one_hot, domain_one_hot))

        def _map_fn(img_path, cmd, cls_label, dom_label):
            image = load_image_from_disk(img_path)
            image = preprocess_image(image, self.target_width, self.target_height)
            if enable_augment_data:
                image = augment_data(image)

            x_dict = {
                "image_input": image,  # shape [H, W, 3]
                "command_input": cmd  # shape [num_commands]
            }
            y_dict = {
                "class_output": cls_label,  # shape [num_classes]
                "domain_output": dom_label,
                "command_output": cmd  # shape [num_commands], repeat for loss calculation
            }
            return x_dict, y_dict

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE).cache()
        ds = ds.shuffle(5000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _write_log_pre_training(self, model, total_imgs, train_count, val_count, test_count, command_weights,
                                domain_weights):
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
            f.write(f"Domain Weights: {domain_weights}\n")
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

    def calculate_domain_weights(self, train_domain_labels):
        """
        Calculates ratio-based domain weights. If domain 0 is more frequent
        than domain 1, domain 1 gets an upweight, etc.
        """
        unique, counts = np.unique(train_domain_labels, return_counts=True)
        total = len(train_domain_labels)
        num_cmds = self.num_domains
        weights = np.zeros(num_cmds, dtype=np.float32)
        for cmd_idx, cnt in zip(unique, counts):
            weights[cmd_idx] = total / (cnt * num_cmds)

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
        The outputs must be named: "class_output" (and "domain_output" if DANN is used).
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def _compute_loss(self, y_true, y_pred, command_weights=None, domain_weights=None):
        """
        Returns (total_loss, steering_loss, domain_loss).
        DANN approach => domain_loss_weight used.
        Override this method in subclasses for custom loss functions.
        """
        # Steering classification
        steering_label = y_true["class_output"]
        steering_pred = y_pred["class_output"]
        steering_ce = tf.keras.losses.categorical_crossentropy(steering_label, steering_pred)

        # Weight steering loss by command
        if command_weights is not None:
            cmd_one_hot = y_true["command_output"]
            per_sample_cmd_weight = tf.reduce_sum(command_weights * cmd_one_hot, axis=1)
            steering_ce *= per_sample_cmd_weight
        steering_loss = tf.reduce_mean(steering_ce)

        # Domain loss
        domain_loss = 0.0
        if ("domain_output" in y_true) and ("domain_output" in y_pred):
            domain_label = y_true["domain_output"]  # shape [batch_size, 1]
            domain_pred = y_pred["domain_output"]  # shape [batch_size, 1]
            domain_loss = tf.keras.losses.binary_crossentropy(domain_label, domain_pred)

            if domain_weights: #apply domain weights if provided
                domain_label_int = tf.cast(tf.squeeze(domain_label, axis=-1), tf.int32)
                sample_domain_w = tf.gather(domain_weights, domain_label_int)
                domain_loss *= sample_domain_w

            domain_loss = tf.reduce_mean(domain_loss) * self.domain_loss_weight

        total_loss = steering_loss + domain_loss
        return total_loss, steering_loss, domain_loss

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
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
        return [tensorboard_cb, checkpoint_cb, reduce_lr_cb]

    @tf.function
    def _train_step(self, x, y_true, model, optimizer,
                    steering_acc_metric,
                    steering_loss_metric,
                    domain_acc_metric,
                    domain_loss_metric,
                    total_loss_metric,
                    command_weights,
                    domain_weights):
        """
        Single training step updating model weights via gradient descent.
        Also logs separate steering, domain, and total losses for the step.
        """
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            total_loss, steering_loss_val, domain_loss_val = self._compute_loss(
                y_true,
                y_pred,
                command_weights,
                domain_weights
            )

        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Print trainable variables and their gradients
        # for var, grad in zip(model.trainable_variables, gradients):
        #     tf.print(
        #         "Variable:", var.name,
        #         "Shape:", var.shape,
        #         "Grad min/max:",
        #         tf.reduce_min(grad) if grad is not None else "None",
        #         tf.reduce_max(grad) if grad is not None else "None"
        #     )

        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update metrics
        steering_acc_metric.update_state(y_true["class_output"], y_pred["class_output"])
        steering_loss_metric.update_state(steering_loss_val)
        if ("domain_output" in y_pred):
            domain_acc_metric.update_state(y_true["domain_output"], y_pred["domain_output"])
            domain_loss_metric.update_state(domain_loss_val)
        total_loss_metric.update_state(total_loss)

    @tf.function
    def _val_step(self, x, y_true, model,
                  steering_acc_metric,
                  steering_loss_metric,
                  domain_acc_metric,
                  domain_loss_metric,
                  total_loss_metric,
                  command_weights,
                  domain_weights):
        """
        Single validation step without weight updates.
        Logs separate steering, domain, and total losses for analysis.
        """
        y_pred = model(x, training=False)
        total_loss, steering_loss_val, domain_loss_val = self._compute_loss(
            y_true,
            y_pred,
            command_weights,
            domain_weights
        )
        steering_acc_metric.update_state(y_true["class_output"], y_pred["class_output"])
        steering_loss_metric.update_state(steering_loss_val)
        if ("domain_output" in y_pred):
            domain_acc_metric.update_state(y_true["domain_output"], y_pred["domain_output"])
            domain_loss_metric.update_state(domain_loss_val)
        total_loss_metric.update_state(total_loss)

    def _train_loop(self, model, train_ds, val_ds, command_weights, domain_weights):
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
        train_domain_acc = tf.keras.metrics.BinaryAccuracy(name="train_domain_acc")
        train_domain_loss = tf.keras.metrics.Mean(name="train_domain_loss")
        train_total_loss = tf.keras.metrics.Mean(name="train_loss")

        # Metrics for validation
        val_steering_acc = tf.keras.metrics.CategoricalAccuracy(name="val_steering_acc")
        val_steering_loss = tf.keras.metrics.Mean(name="val_steering_loss")
        val_domain_acc = tf.keras.metrics.BinaryAccuracy(name="val_domain_acc")
        val_domain_loss = tf.keras.metrics.Mean(name="val_domain_loss")
        val_total_loss = tf.keras.metrics.Mean(name="val_loss")

        steps_per_epoch = train_ds.cardinality().numpy()
        val_steps = val_ds.cardinality().numpy()

        # Calculate steps per epoch
        steps_per_epoch = len(train_ds)

        is_gradient_reversal_used = False
        try:
            model.get_layer("gradient_reversal")
            is_gradient_reversal_used = True
        except:
            print("Gradient reversal layer not found in model. No Lambda update used.")

        for epoch in range(self.epochs):

            # try to set lambda in GradientReversal layer if it exists (for DANN)
            if is_gradient_reversal_used:
                model.get_layer("gradient_reversal").hp_lambda.assign(2 / (1 + np.exp(-10 * (epoch / self.epochs))) - 1)

            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            optimizer = model.optimizer

            # Reset training metrics
            train_steeering_acc.reset_state()
            train_steering_loss.reset_state()
            train_domain_acc.reset_state()
            train_domain_loss.reset_state()
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
                    train_domain_acc,
                    train_domain_loss,
                    train_total_loss,
                    command_weights,
                    domain_weights
                )
                logs = {
                    "loss": train_total_loss.result().numpy(),
                    "steering_loss": train_steering_loss.result().numpy(),
                    "domain_loss": train_domain_loss.result().numpy(),
                    "steering_accuracy": train_steeering_acc.result().numpy(),
                    "domain_accuracy": train_domain_acc.result().numpy()
                }
                for cb in callbacks:
                    cb.on_train_batch_end(step_train, logs=logs)

                pbar_train.set_postfix({
                    "loss": f"{train_total_loss.result():.4f}",
                    "steering_loss": f"{train_steering_loss.result():.4f}",
                    "domain_loss": f"{train_domain_loss.result():.4f}",
                    "steering_accuracy": f"{train_steeering_acc.result():.4f}",
                    "domain_accuracy": f"{train_domain_acc.result():.4f}"
                })
                step_train += 1
                if step_train >= steps_per_epoch:
                    break

            # Reset validation metrics
            val_steering_acc.reset_state()
            val_steering_loss.reset_state()
            val_domain_acc.reset_state()
            val_domain_loss.reset_state()
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
                    val_domain_acc,
                    val_domain_loss,
                    val_total_loss,
                    command_weights,
                    domain_weights
                )
                pbar_val.set_postfix({
                    "val_loss": f"{val_total_loss.result():.4f}",
                    "val_steering_loss": f"{val_steering_loss.result():.4f}",
                    "val_domain_loss": f"{val_domain_loss.result():.4f}",
                    "val_steering_acc": f"{val_steering_acc.result():.4f}",
                    "val_domain_acc": f"{val_domain_acc.result():.4f}"
                })
                step_val += 1
                if step_val >= val_steps:
                    break

            # Logs to pass to callbacks
            epoch_logs = {
                "loss": train_total_loss.result().numpy(),
                "steering_loss": train_steering_loss.result().numpy(),
                "domain_loss": train_domain_loss.result().numpy(),
                "steering accuracy": train_steeering_acc.result().numpy(),
                "domain accuracy": train_domain_acc.result().numpy(),
                "val_loss": val_total_loss.result().numpy(),
                "val_steering_loss": val_steering_loss.result().numpy(),
                "val_domain_loss": val_domain_loss.result().numpy(),
                "val_accuracy": val_steering_acc.result().numpy(),
                "val_domain_accuracy": val_domain_acc.result().numpy()
            }
            for cb in callbacks:
                cb.on_epoch_end(epoch, epoch_logs)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"total_loss={train_total_loss.result():.4f}, "
                f"steering_loss={train_steering_loss.result():.4f}, "
                f"domain_loss={train_domain_loss.result():.4f}, "
                f"acc={train_steeering_acc.result():.4f} | "
                f"val_loss={val_total_loss.result():.4f}, "
                f"val_steering_loss={val_steering_loss.result():.4f}, "
                f"val_domain_loss={val_domain_loss.result():.4f}, "
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
        image_paths, class_labels, float_labels, domain_labels, control_commands = self.load_data()
        total_imgs = len(image_paths)

        # 2) Split
        x_tv, x_test, cls_tv, cls_test, d_tv, d_test, cmd_tv, cmd_test = train_test_split(
            image_paths, class_labels, domain_labels, control_commands,
            test_size=0.1, random_state=42, shuffle=True, stratify=control_commands
        )
        x_train, x_val, cls_train, cls_val, d_train, d_val, cmd_train, cmd_val = train_test_split(
            x_tv, cls_tv, d_tv, cmd_tv, test_size=0.2, random_state=42, shuffle=True, stratify=cmd_tv
        )

        # Create tf.data.Datasets
        train_ds = self.create_tf_dataset(x_train, cls_train, d_train, cmd_train, enable_augment_data=True)
        val_ds = self.create_tf_dataset(x_val, cls_val, d_val, cmd_val, enable_augment_data=False)
        test_ds = self.create_tf_dataset(x_test, cls_test, d_test, cmd_test, enable_augment_data=False)

        # 3) Calculate command & domain weights from TRAIN domain/command labels
        command_weights = self.calculate_command_weights(cmd_train) if self.use_command_weights else None
        domain_weights = self.calculate_domain_weights(d_train) if self.use_domain_weights else None
        steering_weights = self.calculate_steering_weights(cls_train) if self.use_steering_weights else None
        print("Command Weights:", command_weights)
        print("Domain Weights:", domain_weights)
        print("Steering Weights:", steering_weights)

        steps_per_epoch = len(train_ds)

        # Use standard Adam with the schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)

        # Phase 1: Train base network if not in adapter phase
        if not self.adapter_training_phase:
            print("Phase 1: Training base network...")
            model = self.build_model(use_adapters=False)
            model.optimizer = optimizer
            self._write_log_pre_training(model, total_imgs, len(x_train), len(x_val),
                                         len(x_test), command_weights, domain_weights)
            self._train_loop(model, train_ds, val_ds, command_weights, domain_weights)

        # Phase 2: Train adapters if in adapter phase
        else:
            print("Phase 2: Training adapters...")
            # Load base model if path provided
            if self.model_path is None:
                raise ValueError("Model path must be provided for adapter training phase")

            custom_objects = {
                'GradientReversal': GradientReversal,
                'stack_branches_func': stack_branches_func,
                'mask_outputs': mask_outputs
            }
            base_model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)

            # Create new model with adapters
            model = self.build_model(use_adapters=True)

            # Copy weights from base model to non-adapter layers
            for layer in base_model.layers:
                if not any(x in layer.name for x in ['adapter', 'command']):
                    try:
                        model.get_layer(layer.name).set_weights(layer.get_weights())
                        model.get_layer(layer.name).trainable = False
                        print(f"Transferred weights for layer: {layer.name}")
                    except:
                        print(f"Could not transfer weights for layer: {layer.name}")

            model.optimizer = optimizer
            self._write_log_pre_training(model, total_imgs, len(x_train), len(x_val),
                                         len(x_test), command_weights, domain_weights)
            self._train_loop(model, train_ds, val_ds, command_weights, domain_weights)


        # Evaluate on test set (with separate steering/domain loss logging for final metrics)
        test_steering_loss_metric = tf.keras.metrics.Mean()
        test_domain_loss_metric = tf.keras.metrics.Mean()
        test_total_loss_metric = tf.keras.metrics.Mean()
        test_steering_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        test_domain_acc_metric = tf.keras.metrics.BinaryAccuracy()

        for x_test_batch, y_test_batch in test_ds:
            y_pred_test = model(x_test_batch, training=False)
            total_loss, steering_loss_val, domain_loss_val = self._compute_loss(
                y_test_batch,
                y_pred_test,
                command_weights,
                domain_weights
            )
            test_total_loss_metric.update_state(total_loss)
            test_steering_loss_metric.update_state(steering_loss_val)
            test_domain_loss_metric.update_state(domain_loss_val)
            test_steering_acc_metric.update_state(y_test_batch["class_output"], y_pred_test["class_output"])
            if "domain_output" in y_pred_test:
                test_domain_acc_metric.update_state(y_test_batch["domain_output"], y_pred_test["domain_output"])

        print(f"Test Results:\n"
              f"    Test Total Loss: {test_total_loss_metric.result().numpy():.4f}\n"
              f"    Test Steering Loss: {test_steering_loss_metric.result().numpy():.4f}\n"
              f"    Test Domain Loss: {test_domain_loss_metric.result().numpy():.4f}\n"
              f"    Test Steering Accuracy: {test_steering_acc_metric.result().numpy():.4f}\n"
              f"    Test Domain Accuracy: {test_domain_acc_metric.result().numpy():.4f}")

        # Write overall TP, FP, FN, TN to info.txt
        log_file_path = os.path.join(self.log_dir_base, "info.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Test Results:\n"
                           f"    Test Total Loss: {test_total_loss_metric.result().numpy():.4f}\n"
                           f"    Test Steering Loss: {test_steering_loss_metric.result().numpy():.4f}\n"
                           f"    Test Domain Loss: {test_domain_loss_metric.result().numpy():.4f}\n"
                           f"    Test Steering Accuracy: {test_steering_acc_metric.result().numpy():.4f}\n"
                           f"    Test Domain Accuracy: {test_domain_acc_metric.result().numpy():.4f}")

        print("\nPerforming distribution shift analysis...")
        self.analyze_distribution_shift(model, train_ds, test_ds)

    def create_embedding_model(self, original_model, layer_name="flatten"):
        """Creates a model that outputs embeddings from the flatten layer"""
        return Model(
            inputs=original_model.inputs,
            outputs=original_model.get_layer(layer_name).output,
            name="embedding_model"
        )

    def extract_embeddings(self, model, dataset, num_samples=1000):
        """Extracts embeddings from a dataset using the trained model"""
        embeddings = []
        labels = []

        for idx, (x_batch, y_batch) in enumerate(dataset):
            if idx * self.batch_size >= num_samples:
                break
            batch_embeddings = model.predict(x_batch, verbose=0)
            embeddings.append(batch_embeddings)
            labels.append(y_batch["domain_output"].numpy())

        return np.vstack(embeddings), np.concatenate(labels)

    def analyze_distribution_shift(self, model, train_ds, test_ds, layer_name="flatten"):
        """Performs dimensionality reduction and KL analysis on embeddings"""
        # Create embedding model
        embedding_model = self.create_embedding_model(model, layer_name)

        # Extract embeddings
        train_embeddings, train_labels = self.extract_embeddings(embedding_model, train_ds)
        test_embeddings, test_labels = self.extract_embeddings(embedding_model, test_ds)

        # Combine datasets
        all_embeddings = np.concatenate([train_embeddings, test_embeddings])
        all_labels = np.concatenate([train_labels, test_labels])

        # Dimensionality reduction
        self._visualize_embeddings(all_embeddings, all_labels, layer_name)

        # KL-divergence analysis
        self._calculate_kl_divergence(train_embeddings, test_embeddings)

    def _visualize_embeddings(self, embeddings, labels, layer_name="flatten"):
        """Visualizes embeddings using PCA and t-SNE"""
        # Convert labels to 1D array of strings
        labels = np.argmax(labels, axis=1)
        labels = labels.squeeze().astype(str)  # Add this line to fix shape issues

        # PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(embeddings)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
        tsne_results = tsne.fit_transform(embeddings)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x_pca': pca_results[:, 0],
            'y_pca': pca_results[:, 1],
            'x_tsne': tsne_results[:, 0],
            'y_tsne': tsne_results[:, 1],
            'domain': labels
        })

        # Plotting
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x='x_pca',
            y='y_pca',
            hue='domain',
            data=plot_df,
            palette="viridis",
            alpha=0.6
        )
        plt.title(f"PCA Visualization of Embeddings: {layer_name}")

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            x='x_tsne',
            y='y_tsne',
            hue='domain',
            data=plot_df,
            palette="viridis",
            alpha=0.6
        )
        plt.title(f"t-SNE Visualization of Embeddings: {layer_name}")

        plt.savefig(os.path.join(self.log_dir_base, f"{layer_name}_embedding_visualization.png"))
        plt.close()

    def _calculate_kl_divergence(self, source_emb, target_emb, bins=50):
        """Calculates KL divergence between source and target embeddings"""
        # Flatten and normalize
        source_flat = source_emb.flatten()
        target_flat = target_emb.flatten()

        # Create histograms
        min_val = min(np.min(source_flat), np.min(target_flat))
        max_val = max(np.max(source_flat), np.max(target_flat))

        source_hist = np.histogram(source_flat, bins=bins, range=(min_val, max_val))[0] + 1e-10
        target_hist = np.histogram(target_flat, bins=bins, range=(min_val, max_val))[0] + 1e-10

        # Normalize
        source_hist /= np.sum(source_hist)
        target_hist /= np.sum(target_hist)

        # Calculate KL divergence
        kl_source_target = entropy(source_hist, target_hist)
        kl_target_source = entropy(target_hist, source_hist)

        # Log results
        log_file_path = os.path.join(self.log_dir_base, "distribution_shift_analysis.txt")
        with open(log_file_path, "w") as f:
            f.write(f"KL Divergence (Source || Target): {kl_source_target:.4f}\n")
            f.write(f"KL Divergence (Target || Source): {kl_target_source:.4f}\n")
            f.write("\nInterpretation:\n")
            f.write("Values > 1.0 indicate significant distribution shift\n")
            f.write("Values > 2.0 indicate very large distribution differences\n")
            # now print all general model params:
            f.write("\nPARAMETERS:\n")
            for k, v in self.params.items():
                f.write(f"{k}: {v}\n")

        print(f"KL Divergence Analysis saved to {log_file_path}")
