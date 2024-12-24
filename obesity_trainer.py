import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

LABEL_KEY = 'ObesityCategory'

def _transformed_name(key):
    """Generate transformed feature name."""
    return key.replace(' ', '_').lower() + '_xf'

def gzip_reader_fn(filenames):
    """Loads compressed data."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern,
             tf_transform_output,
             num_epochs=None,
             batch_size=64) -> tf.data.Dataset:
    """Prepare input data for training."""
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_transformed_name(LABEL_KEY)
    )
    return dataset

def model_builder():
    """Build a machine learning model for obesity prediction."""
    input_features = [
        'Age', 'Bmi', 'Gender', 'PhysicalActivityLevel'
    ]
    
    input_layers = [
        tf.keras.Input(shape=(1,), name=_transformed_name(f), dtype=tf.float32)
        for f in input_features
    ]
    
    concatenated_features = tf.keras.layers.concatenate(input_layers)
    x = layers.Dense(128, activation='relu')(concatenated_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(5, activation='softmax')(x)  # Assuming 5 categories for obesity

    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=[SparseCategoricalAccuracy()]
    )
    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Create a serving signature for the model."""
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    """Train and export the model."""
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy', 
        patience=10
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(fn_args.serving_model_dir, 'best_model.keras'),
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True
    )

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Prepare datasets
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=20)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    # Build and train the model
    model = model_builder()

    model.fit(
        train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        steps_per_epoch=100,
        validation_steps=20,
        epochs=20
    )

    # Save the model with serving signature
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }
    # Use tf.saved_model.save to save with signatures
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
