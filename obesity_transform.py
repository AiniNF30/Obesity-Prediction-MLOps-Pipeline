import tensorflow as tf
import tensorflow_transform as tft

def _transformed_name(key):
    return key.replace(' ', '_').lower() + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input features."""
    outputs = {}

    # Normalize numerical features
    for feature_name in ['Age', 'BMI']:
        outputs[_transformed_name(feature_name)] = tft.scale_to_z_score(inputs[feature_name])

    # Pass categorical features as is
    for feature_name in ['Gender', 'ObesityCategory', 'PhysicalActivityLevel']:
        outputs[_transformed_name(feature_name)] = inputs[feature_name]

    return outputs
