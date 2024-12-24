import tensorflow as tf
import tensorflow_transform as tft

def _transformed_name(key):
    """Generate a transformed feature name."""
    return key.lower() + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input features for the dataset."""
    outputs = {}

    # Normalize numerical features
    numerical_features = ['Age', 'BMI']
    for feature_name in numerical_features:
        outputs[_transformed_name(feature_name)] = tft.scale_to_z_score(inputs[feature_name])

    # Encode categorical features
    categorical_features = ['Gender', 'ObesityCategory', 'PhysicalActivityLevel']
    for feature_name in categorical_features:
        outputs[_transformed_name(feature_name)] = tft.compute_and_apply_vocabulary(inputs[feature_name])

    return outputs
