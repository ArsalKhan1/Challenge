# sagemaker example

import sagemaker
from sagemaker.pytorch import PyTorch

# Define your S3 paths
train_data_s3_path = './jigsaw-unintended-bias-train-modified.csv'
validation_data_s3_path = './validation.csv'

# Define the estimator
estimator = PyTorch(
    entry_point='profanity_classifier_mps_2_with_validation_torchsize_fix.py',  # Your script
    role=sagemaker.get_execution_role(),
    framework_version='1.9.0',
    py_version='py38',
    instance_count=1,
    instance_type='ml.g4dn.2xlarge',  # Choose your instance type
    hyperparameters={
        'epochs': 3,
        'batch-size': 16,
        'learning-rate': 5e-5
    },
    output_path='./results',
)

# Set the input data paths for training and validation
estimator.fit({
    'train': train_data_s3_path,
    'validation': validation_data_s3_path
})
