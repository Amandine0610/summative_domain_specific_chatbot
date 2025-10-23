#!/usr/bin/env python3
"""
Demonstration script for Human Activity Recognition using Hidden Markov Models.
This script shows the complete pipeline from data generation to model evaluation.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_collection import ActivityDataGenerator
from src.feature_extraction import FeatureExtractor
from src.hmm_model import ActivityHMM, create_evaluation_table


def main():
    """Run the complete HMM activity recognition demonstration."""
    
    print("=" * 60)
    print("Human Activity Recognition using Hidden Markov Models")
    print("=" * 60)
    
    # Configuration
    sampling_rate = 50
    window_size = 2.0
    overlap = 0.5
    samples_per_activity = 12
    duration_per_sample = 8.0
    
    print(f"\nConfiguration:")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Window size: {window_size} seconds")
    print(f"  Window overlap: {overlap*100}%")
    print(f"  Samples per activity: {samples_per_activity}")
    print(f"  Duration per sample: {duration_per_sample} seconds")
    
    # Step 1: Generate synthetic sensor data
    print(f"\n{'-'*40}")
    print("Step 1: Data Collection")
    print(f"{'-'*40}")
    
    generator = ActivityDataGenerator(sampling_rate=sampling_rate)
    dataset = generator.generate_dataset(
        samples_per_activity=samples_per_activity,
        duration_per_sample=duration_per_sample
    )
    
    print(f"Activities: {generator.activities}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Activity distribution:\n{dataset['activity'].value_counts()}")
    
    # Save dataset
    os.makedirs('data', exist_ok=True)
    dataset_path = 'data/synthetic_activity_data.csv'
    generator.save_dataset(dataset, dataset_path)
    
    # Step 2: Feature extraction
    print(f"\n{'-'*40}")
    print("Step 2: Feature Extraction")
    print(f"{'-'*40}")
    
    extractor = FeatureExtractor(
        window_size=window_size,
        overlap=overlap,
        sampling_rate=sampling_rate
    )
    
    features_df = extractor.extract_features_from_dataset(dataset, include_labels=True)
    
    print(f"Extracted {features_df.shape[1]-3} features from {features_df.shape[0]} windows")
    print(f"Feature windows per activity:\n{features_df['activity'].value_counts()}")
    
    # Save features
    features_path = 'data/extracted_features.csv'
    features_df.to_csv(features_path, index=False)
    
    # Step 3: Data splitting
    print(f"\n{'-'*40}")
    print("Step 3: Data Splitting")
    print(f"{'-'*40}")
    
    # Split by sample_id to avoid data leakage
    unique_samples = features_df['sample_id'].unique()
    train_samples = []
    test_samples = []
    
    for activity in generator.activities:
        activity_samples = [s for s in unique_samples if s.startswith(activity)]
        n_train = int(0.8 * len(activity_samples))
        train_samples.extend(activity_samples[:n_train])
        test_samples.extend(activity_samples[n_train:])
    
    train_df = features_df[features_df['sample_id'].isin(train_samples)]
    test_df = features_df[features_df['sample_id'].isin(test_samples)]
    
    print(f"Training samples: {len(train_samples)} ({len(train_df)} windows)")
    print(f"Test samples: {len(test_samples)} ({len(test_df)} windows)")
    
    # Step 4: HMM training
    print(f"\n{'-'*40}")
    print("Step 4: HMM Training")
    print(f"{'-'*40}")
    
    hmm_model = ActivityHMM(n_states=4, random_state=42)
    hmm_model.fit(train_df)
    
    # Step 5: Model evaluation
    print(f"\n{'-'*40}")
    print("Step 5: Model Evaluation")
    print(f"{'-'*40}")
    
    metrics = hmm_model.evaluate(test_df)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Mean Log Probability: {metrics['mean_log_probability']:.3f}")
    
    # Create evaluation table
    eval_table = create_evaluation_table(metrics, hmm_model.activities)
    print(f"\nDetailed Evaluation Results:")
    print(eval_table.round(3).to_string(index=False))
    
    # Step 6: Visualizations
    print(f"\n{'-'*40}")
    print("Step 6: Visualizations")
    print(f"{'-'*40}")
    
    print("Generating visualizations...")
    
    # Transition matrix
    hmm_model.visualize_transition_matrix()
    
    # Sample predictions
    test_sample_ids = test_df['sample_id'].unique()[:2]
    for sample_id in test_sample_ids:
        hmm_model.visualize_predictions(test_df, sample_id=sample_id)
    
    # Step 7: Save results
    print(f"\n{'-'*40}")
    print("Step 7: Saving Results")
    print(f"{'-'*40}")
    
    os.makedirs('results', exist_ok=True)
    
    # Save model
    model_path = 'results/trained_hmm_model.pkl'
    hmm_model.save_model(model_path)
    
    # Save evaluation results
    results_path = 'results/evaluation_results.csv'
    eval_table.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")
    
    # Save detailed metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = 'results/detailed_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Detailed metrics saved to {metrics_path}")
    
    print(f"\n{'='*60}")
    print("Demonstration completed successfully!")
    print(f"{'='*60}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  - Generated {len(dataset)} sensor data points")
    print(f"  - Extracted {len(features_df)} feature windows")
    print(f"  - Trained HMM with {hmm_model.n_states} states")
    print(f"  - Achieved {metrics['overall_accuracy']:.1%} accuracy on test data")
    
    print(f"\nFiles created:")
    print(f"  - {dataset_path}")
    print(f"  - {features_path}")
    print(f"  - {model_path}")
    print(f"  - {results_path}")
    print(f"  - {metrics_path}")
    
    print(f"\nNext steps:")
    print(f"  1. Open notebooks/human_activity_hmm.ipynb for detailed analysis")
    print(f"  2. Collect real sensor data using Sensor Logger app")
    print(f"  3. Replace synthetic data with real data for improved results")


if __name__ == "__main__":
    main()