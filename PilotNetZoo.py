import keras
import tensorflow as tf

from keras.src.layers import LayerNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, Activation, Lambda, Concatenate, Add, Multiply, GlobalAveragePooling2D, Reshape
from PilotNetJunctionBase import BasePilotNet
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.optimizers import Adam
from not_used.PilotNetJunctionModelFit import PilotNetModelFit
from PilotNetJunctionRaspiCarBase import BasePilotNetRaspiCar


# --- Custom gradient reversal ---
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


class PilotNetMultiHeadSmall(BasePilotNet):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)

        branch_outputs = []
        for cmd_name in self.control_commands_list:
            b = Dense(100, activation='elu', name=f"{cmd_name}_fc1")(x)
            b = LayerNormalization(name=f"{cmd_name}_norm1")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d1")(b)
            b = Dense(50, activation='elu', name=f"{cmd_name}_fc2")(b)
            b = LayerNormalization(name=f"{cmd_name}_norm2")(b)
            b = Dense(30, activation='elu', name=f"{cmd_name}_fc3")(b)
            b = LayerNormalization(name=f"{cmd_name}_norm3")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d2")(b)
            out = Dense(len(self.boundaries), activation='softmax',
                        name=f"{cmd_name}_output")(b)
            branch_outputs.append(out)

        # Stack branches, then select correct branch
        stacked_branches = Lambda(stack_branches_func, name="stack_branches")(branch_outputs)
        final_output = Lambda(mask_outputs, name="class_output")([stacked_branches, command_input])


        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": final_output},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetSingleHeadGatedSmall(BasePilotNet):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)

        # Command embedding layer
        command_embedding = Dense(4, activation="relu", name="command_embedding")(command_input)

        # Gating mechanism
        gating_weights = Dense(units=x.shape[-1], activation="sigmoid", name="gating_weights")(command_embedding)
        gated_features = tf.keras.layers.Multiply(name="gated_features")([x, gating_weights])

        # Concatenate gated features and command embedding
        combined = Concatenate(name="combined_features")([gated_features, command_embedding])

        b = Dense(100, activation='elu', name=f"fc1")(combined)
        b = LayerNormalization(name=f"classifier_norm1")(b)
        b = Dropout(0.2, name=f"classifier_d1")(b)
        b = Dense(50, activation='elu', name=f"classifier_fc2")(b)
        b = LayerNormalization(name=f"classifier_norm2")(b)
        b = Dense(30, activation='elu', name=f"classifier_fc3")(b)
        b = LayerNormalization(name=f"classifier_norm3")(b)
        b = Dropout(0.2, name=f"classifier_d2")(b)
        out = Dense(len(self.boundaries), activation='softmax',
                    name=f"class_output")(b)



        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": out},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetMultiHeadSmallRaspiCar(BasePilotNetRaspiCar):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)

        branch_outputs = []
        for cmd_name in self.control_commands_list:
            b = Dense(100, activation='elu', name=f"{cmd_name}_fc1")(x)
            b = LayerNormalization(name=f"{cmd_name}_norm1")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d1")(b)
            b = Dense(50, activation='elu', name=f"{cmd_name}_fc2")(b)
            b = LayerNormalization(name=f"{cmd_name}_norm2")(b)
            b = Dense(30, activation='elu', name=f"{cmd_name}_fc3")(b)
            b = LayerNormalization(name=f"{cmd_name}_norm3")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d2")(b)
            out = Dense(len(self.boundaries), activation='softmax',
                        name=f"{cmd_name}_output")(b)
            branch_outputs.append(out)

        # Stack branches, then select correct branch
        stacked_branches = Lambda(stack_branches_func, name="stack_branches")(branch_outputs)
        final_output = Lambda(mask_outputs, name="class_output")([stacked_branches, command_input])


        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": final_output},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetSingleHeadGatedSmallRaspiCar(BasePilotNetRaspiCar):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)

        # Command embedding layer
        command_embedding = Dense(4, activation="relu", name="command_embedding")(command_input)

        # Gating mechanism
        gating_weights = Dense(units=x.shape[-1], activation="sigmoid", name="gating_weights")(command_embedding)
        gated_features = tf.keras.layers.Multiply(name="gated_features")([x, gating_weights])

        # Concatenate gated features and command embedding
        combined = Concatenate(name="combined_features")([gated_features, command_embedding])

        b = Dense(100, activation='elu', name=f"fc1")(combined)
        b = LayerNormalization(name=f"classifier_norm1")(b)
        b = Dropout(0.2, name=f"classifier_d1")(b)
        b = Dense(50, activation='elu', name=f"classifier_fc2")(b)
        b = LayerNormalization(name=f"classifier_norm2")(b)
        b = Dense(30, activation='elu', name=f"classifier_fc3")(b)
        b = LayerNormalization(name=f"classifier_norm3")(b)
        b = Dropout(0.2, name=f"classifier_d2")(b)
        out = Dense(len(self.boundaries), activation='softmax',
                    name=f"class_output")(b)



        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": out},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetSingleHeadNotGatedSmall(BasePilotNet):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)


        combined = Concatenate(name="combined_features")([x, command_input])

        b = Dense(100, activation='elu', name=f"fc1")(combined)
        b = LayerNormalization(name=f"classifier_norm1")(b)
        b = Dropout(0.2, name=f"classifier_d1")(b)
        b = Dense(50, activation='elu', name=f"classifier_fc2")(b)
        b = LayerNormalization(name=f"classifier_norm2")(b)
        b = Dense(30, activation='elu', name=f"classifier_fc3")(b)
        b = LayerNormalization(name=f"classifier_norm3")(b)
        b = Dropout(0.2, name=f"classifier_d2")(b)
        out = Dense(len(self.boundaries), activation='softmax',
                    name=f"class_output")(b)



        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": out},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetSingleHeadNotGatedSmallRaspiCar(BasePilotNetRaspiCar):
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = LayerNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = LayerNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = LayerNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = LayerNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = LayerNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        x = Dropout(0.2, name="d3")(x)


        # Concatenate gated features and command embedding
        combined = Concatenate(name="combined_features")([x, command_input])

        b = Dense(100, activation='elu', name=f"fc1")(combined)
        b = LayerNormalization(name=f"classifier_norm1")(b)
        b = Dropout(0.2, name=f"classifier_d1")(b)
        b = Dense(50, activation='elu', name=f"classifier_fc2")(b)
        b = LayerNormalization(name=f"classifier_norm2")(b)
        b = Dense(30, activation='elu', name=f"classifier_fc3")(b)
        b = LayerNormalization(name=f"classifier_norm3")(b)
        b = Dropout(0.2, name=f"classifier_d2")(b)
        out = Dense(len(self.boundaries), activation='softmax',
                    name=f"class_output")(b)



        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": out},
            name="PilotNetMultiHeadSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={"class_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"]},
        )

        return model

class PilotNetMultiHeadDANNSmall(BasePilotNet):
    '''
    # Not used in Thesis
    '''
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv2")(image_input)
        x = BatchNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv1")(x)
        x = BatchNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = BatchNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = BatchNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = BatchNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        flatten = Dropout(0.2, name="d3")(x)

        branch_outputs = []
        for cmd_name in self.control_commands_list:
            b = Dense(100, activation='elu', name=f"{cmd_name}_fc1")(flatten)
            b = BatchNormalization(name=f"{cmd_name}_norm1")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d1")(b)

            b = Dense(50, activation='elu', name=f"{cmd_name}_fc2")(b)
            b = BatchNormalization(name=f"{cmd_name}_norm2")(b)

            b = Dense(30, activation='elu', name=f"{cmd_name}_fc3")(b)
            b = BatchNormalization(name=f"{cmd_name}_norm3")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d2")(b)

            out = Dense(len(self.boundaries), activation='softmax',
                        name=f"{cmd_name}_output")(b)
            branch_outputs.append(out)

        # Stack branches, then select correct branch
        stacked_branches = Lambda(stack_branches_func, name="stack_branches")(branch_outputs)
        final_output = Lambda(mask_outputs, name="class_output")([stacked_branches, command_input])

        # --- Domain Classifier ---
        grl = GradientReversal(hp_lambda=1.0, name="gradient_reversal")(flatten)
        d = Dense(100, activation='elu', name="domain_fc1")(grl)
        d = BatchNormalization(name=f"domain_norm1")(d)
        d = Dropout(0.2, name="domain_d1")(d)

        d = Dense(50, activation='elu', name="domain_fc2")(d)
        d = BatchNormalization(name=f"domain_norm2")(d)

        d = Dense(30, activation='elu', name="domain_fc3")(d)
        d = BatchNormalization(name=f"domain_norm3")(d)
        d = Dropout(0.2, name="domain_d2")(d)

        domain_output = Dense(1, activation='sigmoid', name="domain_output")(d)

        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": final_output, "domain_output": domain_output},
            name="PilotNetMultiHeadDANNSmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"class_output": "categorical_crossentropy",
                  "domain_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"],
                     "domain_output": ["accuracy"]},
        )

        return model

class PilotNetMultiHeadDANNSmallAuxiliary(BasePilotNet):
    '''
    # Not used in Thesis
    '''
    def build_model(self, use_adapters=False):
        image_input = Input(shape=(self.target_height, self.target_width, 3),
                            name="image_input")
        command_input = Input(shape=(self.total_control_commands,),
                              name="command_input")

        # First Conv Block
        x = Conv2D(24, (5, 5), strides=(2, 2), activation=None, name="conv1")(image_input)
        x = BatchNormalization(name="norm1")(x)
        x = Activation("elu", name="act1")(x)
        x = Dropout(0.2, name="d1")(x)

        # Second Conv Block
        x = Conv2D(36, (5, 5), strides=(2, 2), activation=None, name="conv2")(x)
        x = BatchNormalization(name="norm2")(x)
        x = Activation("elu", name="act2")(x)

        # Third Conv Block
        x = Conv2D(48, (5, 5), strides=(2, 2), activation=None, name="conv3")(x)
        x = BatchNormalization(name="norm3")(x)
        x = Activation("elu", name="act3")(x)

        # Fourth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv4")(x)
        x = BatchNormalization(name="norm4")(x)
        x = Activation("elu", name="act4")(x)
        x = Dropout(0.2, name="d2")(x)

        # Fifth Conv Block
        x = Conv2D(64, (3, 3), activation=None, name="conv5")(x)
        x = BatchNormalization(name="norm5")(x)
        x = Activation("elu", name="act5")(x)

        # Flatten the feature maps
        x = Flatten(name="flatten")(x)
        flatten = Dropout(0.2, name="d3")(x)

        branch_outputs = []
        for cmd_name in self.control_commands_list:
            b = Dense(100, activation='elu', name=f"{cmd_name}_fc1")(flatten)
            b = BatchNormalization(name=f"{cmd_name}_norm1")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d1")(b)

            b = Dense(50, activation='elu', name=f"{cmd_name}_fc2")(b)
            b = BatchNormalization(name=f"{cmd_name}_norm2")(b)

            b = Dense(30, activation='elu', name=f"{cmd_name}_fc3")(b)
            b = BatchNormalization(name=f"{cmd_name}_norm3")(b)
            b = Dropout(0.2, name=f"{cmd_name}_d2")(b)

            out = Dense(len(self.boundaries), activation='softmax',
                        name=f"{cmd_name}_output")(b)
            branch_outputs.append(out)

        # Stack branches, then select correct branch
        stacked_branches = Lambda(stack_branches_func, name="stack_branches")(branch_outputs)
        final_output = Lambda(mask_outputs, name="class_output")([stacked_branches, command_input])

        # --- Domain Classifier ---
        grl = GradientReversal(hp_lambda=1.0, name="gradient_reversal")(flatten)
        d = Dense(100, activation='elu', name="domain_fc1")(grl)
        d = BatchNormalization(name=f"domain_norm1")(d)
        d = Dropout(0.2, name="domain_d1")(d)

        d = Dense(50, activation='elu', name="domain_fc2")(d)
        d = BatchNormalization(name=f"domain_norm2")(d)

        d = Dense(30, activation='elu', name="domain_fc3")(d)
        d = BatchNormalization(name=f"domain_norm3")(d)
        d = Dropout(0.2, name="domain_d2")(d)

        domain_output = Dense(self.num_domains, activation='sigmoid', name="domain_output")(d)

        # --- Command Classifier ---
        c = Dense(100, activation='elu', name="command_fc1")(flatten)
        c = BatchNormalization(name=f"command_norm1")(c)
        c = Dropout(0.2, name="command_d1")(c)

        c = Dense(50, activation='elu', name="command_fc2")(c)
        c = BatchNormalization(name=f"command_norm2")(c)

        c = Dense(30, activation='elu', name="command_fc3")(c)
        c = BatchNormalization(name=f"command_norm3")(c)
        c = Dropout(0.2, name="command_d2")(c)

        command_output = Dense(self.total_control_commands, activation='sigmoid', name="command_output")(c)

        model = Model(
            inputs={"image_input": image_input, "command_input": command_input},
            outputs={"class_output": final_output, "domain_output": domain_output, "command_output": command_output},
            name="PilotNetMultiHeadDANNAuxiliarySmall"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"class_output": "categorical_crossentropy",
                  "domain_output": "categorical_crossentropy",
                  "command_output": "categorical_crossentropy"},
            metrics={"class_output": ["accuracy"],
                     "domain_output": ["accuracy"],
                     "command_output": ["accuracy"]},
        )

        return model

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

        # command classification
        command_label = y_true["command_output"]
        command_pred = y_pred["command_output"]
        command_ce = tf.keras.losses.categorical_crossentropy(command_label, command_pred)
        command_ce = tf.reduce_mean(command_ce)

        # Weight steering loss by command
        if command_weights:
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

        total_loss = steering_loss + domain_loss + command_ce
        return total_loss, steering_loss, domain_loss


