"""
Hidden Markov Model Implementation for Human Activity Recognition
Implements HMM with Viterbi algorithm for activity state decoding.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')


class ActivityHMM:
    """
    Hidden Markov Model for Human Activity Recognition.
    """
    
    def __init__(self, n_states: int = 4, n_components: int = 1, 
                 covariance_type: str = "full", random_state: int = 42):
        """
        Initialize the HMM model.
        
        Args:
            n_states: Number of hidden states (activities)
            n_components: Number of Gaussian components per state
            covariance_type: Type of covariance matrix ("full", "diag", "tied", "spherical")
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=100
        )
        
        # State and activity mappings
        self.activities = ['standing', 'walking', 'jumping', 'still']
        self.state_to_activity = {i: activity for i, activity in enumerate(self.activities)}
        self.activity_to_state = {activity: i for i, activity in enumerate(self.activities)}
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, features_df: pd.DataFrame, 
                    feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare feature data for HMM training.
        
        Args:
            features_df: DataFrame with extracted features
            feature_columns: List of feature columns to use (if None, use all numeric columns)
            
        Returns:
            Tuple of (features, labels, sequence_lengths)
        """
        if feature_columns is None:
            # Use all numeric columns except metadata
            exclude_cols = ['sample_id', 'window_idx', 'activity']
            feature_columns = [col for col in features_df.columns 
                             if col not in exclude_cols and 
                             features_df[col].dtype in ['float64', 'int64']]
        
        # Extract features
        X = features_df[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract labels if available
        if 'activity' in features_df.columns:
            y = features_df['activity'].map(self.activity_to_state).values
        else:
            y = None
        
        # Calculate sequence lengths (number of windows per sample)
        if 'sample_id' in features_df.columns:
            sequence_lengths = features_df.groupby('sample_id').size().values
        else:
            sequence_lengths = [len(features_df)]
        
        return X, y, sequence_lengths
    
    def fit(self, features_df: pd.DataFrame, feature_columns: List[str] = None):
        """
        Train the HMM model on feature data.
        
        Args:
            features_df: DataFrame with extracted features and activity labels
            feature_columns: List of feature columns to use
        """
        print("Preparing training data...")
        X, y, sequence_lengths = self.prepare_data(features_df, feature_columns)
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HMM model
        print(f"Training HMM with {self.n_states} states...")
        print(f"Feature dimensions: {X_scaled.shape[1]}")
        print(f"Number of sequences: {len(sequence_lengths)}")
        print(f"Total observations: {X_scaled.shape[0]}")
        
        self.model.fit(X_scaled, sequence_lengths)
        self.is_fitted = True
        
        # Store feature columns for later use
        self.feature_columns = feature_columns if feature_columns else [
            col for col in features_df.columns 
            if col not in ['sample_id', 'window_idx', 'activity'] and 
            features_df[col].dtype in ['float64', 'int64']
        ]
        
        print("HMM training completed!")
        
        # Print model parameters
        self._print_model_summary()
    
    def _print_model_summary(self):
        """Print summary of trained model parameters."""
        print("\n=== HMM Model Summary ===")
        print(f"Number of states: {self.model.n_components}")
        print(f"Feature dimensions: {len(self.feature_columns)}")
        
        # Transition matrix
        print(f"\nTransition Matrix:")
        transition_df = pd.DataFrame(
            self.model.transmat_,
            index=[f"State_{i} ({self.state_to_activity[i]})" for i in range(self.n_states)],
            columns=[f"State_{i}" for i in range(self.n_states)]
        )
        print(transition_df.round(3))
        
        # Initial state probabilities
        print(f"\nInitial State Probabilities:")
        for i, prob in enumerate(self.model.startprob_):
            print(f"State_{i} ({self.state_to_activity[i]}): {prob:.3f}")
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict activity states using the trained HMM.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Tuple of (predicted_states, log_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _, sequence_lengths = self.prepare_data(features_df, self.feature_columns)
        X_scaled = self.scaler.transform(X)
        
        # Predict states for each sequence
        all_predictions = []
        all_log_probs = []
        
        start_idx = 0
        for seq_len in sequence_lengths:
            end_idx = start_idx + seq_len
            sequence = X_scaled[start_idx:end_idx]
            
            # Use Viterbi algorithm to find most likely state sequence
            log_prob, states = self.model.decode(sequence, algorithm="viterbi")
            
            all_predictions.extend(states)
            all_log_probs.append(log_prob)
            
            start_idx = end_idx
        
        return np.array(all_predictions), np.array(all_log_probs)
    
    def evaluate(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            features_df: DataFrame with features and true activity labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if 'activity' not in features_df.columns:
            raise ValueError("Activity labels required for evaluation")
        
        # Get predictions
        predicted_states, log_probs = self.predict(features_df)
        
        # Get true labels
        true_states = features_df['activity'].map(self.activity_to_state).values
        
        # Convert states back to activity names
        predicted_activities = [self.state_to_activity[state] for state in predicted_states]
        true_activities = [self.state_to_activity[state] for state in true_states]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_activities, predicted_activities)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_activities, predicted_activities, average=None, labels=self.activities
        )
        
        # Calculate sensitivity (recall) and specificity for each class
        cm = confusion_matrix(true_activities, predicted_activities, labels=self.activities)
        
        metrics = {
            'overall_accuracy': accuracy,
            'mean_log_probability': np.mean(log_probs)
        }
        
        # Per-class metrics
        for i, activity in enumerate(self.activities):
            if i < len(precision):
                metrics[f'{activity}_precision'] = precision[i]
                metrics[f'{activity}_recall'] = recall[i]  # Same as sensitivity
                metrics[f'{activity}_f1'] = f1[i]
                metrics[f'{activity}_support'] = support[i]
                
                # Calculate specificity
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics[f'{activity}_specificity'] = specificity
        
        return metrics
    
    def visualize_transition_matrix(self):
        """Visualize the transition matrix as a heatmap."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")
        
        plt.figure(figsize=(8, 6))
        
        # Create labeled transition matrix
        transition_df = pd.DataFrame(
            self.model.transmat_,
            index=[f"{activity}" for activity in self.activities],
            columns=[f"{activity}" for activity in self.activities]
        )
        
        sns.heatmap(transition_df, annot=True, cmap='Blues', fmt='.3f',
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('HMM Transition Matrix')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, features_df: pd.DataFrame, sample_id: str = None):
        """
        Visualize predicted vs true activity sequences.
        
        Args:
            features_df: DataFrame with features and true labels
            sample_id: Specific sample to visualize (if None, use first sample)
        """
        if 'activity' not in features_df.columns:
            print("True activity labels not available for comparison")
            return
        
        # Filter to specific sample if requested
        if sample_id:
            sample_data = features_df[features_df['sample_id'] == sample_id]
        else:
            # Use first sample
            first_sample = features_df['sample_id'].iloc[0]
            sample_data = features_df[features_df['sample_id'] == first_sample]
            sample_id = first_sample
        
        if sample_data.empty:
            print(f"No data found for sample: {sample_id}")
            return
        
        # Get predictions for this sample
        predicted_states, _ = self.predict(sample_data)
        true_states = sample_data['activity'].map(self.activity_to_state).values
        
        # Create time axis
        time_points = np.arange(len(predicted_states))
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_points, true_states, 'o-', label='True States', alpha=0.7)
        plt.ylabel('Activity State')
        plt.title(f'True Activity Sequence - Sample: {sample_id}')
        plt.yticks(range(self.n_states), self.activities)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_points, predicted_states, 's-', label='Predicted States', 
                color='red', alpha=0.7)
        plt.xlabel('Time Window')
        plt.ylabel('Activity State')
        plt.title('Predicted Activity Sequence')
        plt.yticks(range(self.n_states), self.activities)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print accuracy for this sample
        accuracy = np.mean(predicted_states == true_states)
        print(f"Sample accuracy: {accuracy:.3f}")
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'activities': self.activities,
            'state_to_activity': self.state_to_activity,
            'activity_to_state': self.activity_to_state,
            'n_states': self.n_states
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.activities = model_data['activities']
        self.state_to_activity = model_data['state_to_activity']
        self.activity_to_state = model_data['activity_to_state']
        self.n_states = model_data['n_states']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")


def create_evaluation_table(metrics: Dict[str, float], activities: List[str]) -> pd.DataFrame:
    """
    Create evaluation table as specified in the project requirements.
    
    Args:
        metrics: Dictionary of evaluation metrics
        activities: List of activity names
        
    Returns:
        DataFrame with evaluation results
    """
    evaluation_data = []
    
    for activity in activities:
        row = {
            'State (Activity)': activity,
            'Number of Samples': int(metrics.get(f'{activity}_support', 0)),
            'Sensitivity': metrics.get(f'{activity}_recall', 0.0),
            'Specificity': metrics.get(f'{activity}_specificity', 0.0),
            'Overall Accuracy': metrics.get('overall_accuracy', 0.0)
        }
        evaluation_data.append(row)
    
    return pd.DataFrame(evaluation_data)


if __name__ == "__main__":
    # Example usage
    from data_collection import ActivityDataGenerator
    from feature_extraction import FeatureExtractor
    
    # Generate sample data
    print("Generating sample data...")
    generator = ActivityDataGenerator(sampling_rate=50)
    dataset = generator.generate_dataset(samples_per_activity=10, duration_per_sample=8.0)
    
    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor(window_size=2.0, overlap=0.5, sampling_rate=50)
    features_df = extractor.extract_features_from_dataset(dataset)
    
    # Split data
    train_samples = features_df['sample_id'].unique()[:8*4]  # 8 samples per activity for training
    test_samples = features_df['sample_id'].unique()[8*4:]   # 2 samples per activity for testing
    
    train_df = features_df[features_df['sample_id'].isin(train_samples)]
    test_df = features_df[features_df['sample_id'].isin(test_samples)]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Train HMM
    print("Training HMM...")
    hmm_model = ActivityHMM(n_states=4, random_state=42)
    hmm_model.fit(train_df)
    
    # Evaluate
    print("Evaluating model...")
    metrics = hmm_model.evaluate(test_df)
    
    # Create evaluation table
    eval_table = create_evaluation_table(metrics, hmm_model.activities)
    print("\nEvaluation Results:")
    print(eval_table.round(3))
    
    # Visualizations
    hmm_model.visualize_transition_matrix()
    hmm_model.visualize_predictions(test_df)