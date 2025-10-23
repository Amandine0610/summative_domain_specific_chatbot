"""
Feature Extraction Module for Human Activity Recognition
Extracts time-domain and frequency-domain features from accelerometer and gyroscope data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extracts comprehensive features from accelerometer and gyroscope sensor data.
    """
    
    def __init__(self, window_size: float = 2.0, overlap: float = 0.5, 
                 sampling_rate: int = 50):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Size of sliding window in seconds
            overlap: Overlap between windows (0.0 to 1.0)
            sampling_rate: Sampling rate in Hz
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_size * sampling_rate)
        self.step_size = int(self.window_samples * (1 - overlap))
        
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from a signal window.
        
        Args:
            signal: 1D array of sensor readings
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal)
        
        # Higher-order statistical features
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # Root Mean Square (RMS)
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        # Mean absolute deviation
        features['mad'] = np.mean(np.abs(signal - features['mean']))
        
        # Interquartile range
        features['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        
        # Energy
        features['energy'] = np.sum(signal**2)
        
        return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features from a signal window.
        
        Args:
            signal: 1D array of sensor readings
            
        Returns:
            Dictionary of frequency-domain features
        """
        features = {}
        
        # Compute FFT
        fft_values = fft(signal)
        fft_magnitude = np.abs(fft_values)
        freqs = fftfreq(len(signal), 1/self.sampling_rate)
        
        # Only use positive frequencies
        positive_freq_idx = freqs > 0
        fft_magnitude = fft_magnitude[positive_freq_idx]
        freqs = freqs[positive_freq_idx]
        
        if len(fft_magnitude) == 0:
            # Return zero features if no positive frequencies
            return {
                'dominant_frequency': 0.0,
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_energy': 0.0,
                'spectral_entropy': 0.0
            }
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]
        
        # Spectral centroid (weighted mean of frequencies)
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(fft_magnitude**2)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Spectral energy
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
        # Spectral entropy
        normalized_fft = fft_magnitude / np.sum(fft_magnitude)
        normalized_fft = normalized_fft[normalized_fft > 0]  # Remove zeros for log
        features['spectral_entropy'] = -np.sum(normalized_fft * np.log2(normalized_fft))
        
        return features
    
    def extract_cross_axis_features(self, acc_data: np.ndarray, 
                                  gyro_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features that involve multiple axes or sensor types.
        
        Args:
            acc_data: Accelerometer data [x, y, z]
            gyro_data: Gyroscope data [x, y, z]
            
        Returns:
            Dictionary of cross-axis features
        """
        features = {}
        
        # Signal Magnitude Area (SMA) for accelerometer
        features['acc_sma'] = np.mean(np.abs(acc_data).sum(axis=1))
        
        # Signal Magnitude Area (SMA) for gyroscope
        features['gyro_sma'] = np.mean(np.abs(gyro_data).sum(axis=1))
        
        # Vector magnitude for accelerometer
        acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        features['acc_magnitude_mean'] = np.mean(acc_magnitude)
        features['acc_magnitude_std'] = np.std(acc_magnitude)
        
        # Vector magnitude for gyroscope
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
        features['gyro_magnitude_std'] = np.std(gyro_magnitude)
        
        # Correlation between accelerometer axes
        if len(acc_data) > 1:
            features['acc_xy_corr'] = np.corrcoef(acc_data[:, 0], acc_data[:, 1])[0, 1]
            features['acc_xz_corr'] = np.corrcoef(acc_data[:, 0], acc_data[:, 2])[0, 1]
            features['acc_yz_corr'] = np.corrcoef(acc_data[:, 1], acc_data[:, 2])[0, 1]
        else:
            features['acc_xy_corr'] = 0.0
            features['acc_xz_corr'] = 0.0
            features['acc_yz_corr'] = 0.0
        
        # Correlation between gyroscope axes
        if len(gyro_data) > 1:
            features['gyro_xy_corr'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 1])[0, 1]
            features['gyro_xz_corr'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 2])[0, 1]
            features['gyro_yz_corr'] = np.corrcoef(gyro_data[:, 1], gyro_data[:, 2])[0, 1]
        else:
            features['gyro_xy_corr'] = 0.0
            features['gyro_xz_corr'] = 0.0
            features['gyro_yz_corr'] = 0.0
        
        # Handle NaN correlations (when std is 0)
        for key in features:
            if 'corr' in key and (np.isnan(features[key]) or np.isinf(features[key])):
                features[key] = 0.0
        
        return features
    
    def extract_window_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all features from a single window of sensor data.
        
        Args:
            window_data: DataFrame with columns [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            
        Returns:
            Dictionary of all extracted features
        """
        all_features = {}
        
        # Extract accelerometer and gyroscope data
        acc_data = window_data[['acc_x', 'acc_y', 'acc_z']].values
        gyro_data = window_data[['gyro_x', 'gyro_y', 'gyro_z']].values
        
        # Extract features for each axis
        axes = ['x', 'y', 'z']
        sensors = ['acc', 'gyro']
        
        for sensor_idx, sensor in enumerate(sensors):
            sensor_data = acc_data if sensor == 'acc' else gyro_data
            
            for axis_idx, axis in enumerate(axes):
                signal = sensor_data[:, axis_idx]
                
                # Time-domain features
                time_features = self.extract_time_domain_features(signal)
                for feature_name, value in time_features.items():
                    all_features[f'{sensor}_{axis}_{feature_name}'] = value
                
                # Frequency-domain features
                freq_features = self.extract_frequency_domain_features(signal)
                for feature_name, value in freq_features.items():
                    all_features[f'{sensor}_{axis}_{feature_name}'] = value
        
        # Cross-axis features
        cross_features = self.extract_cross_axis_features(acc_data, gyro_data)
        all_features.update(cross_features)
        
        return all_features
    
    def create_sliding_windows(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create sliding windows from continuous sensor data.
        
        Args:
            data: DataFrame with sensor data
            
        Returns:
            List of DataFrame windows
        """
        windows = []
        
        for start_idx in range(0, len(data) - self.window_samples + 1, self.step_size):
            end_idx = start_idx + self.window_samples
            window = data.iloc[start_idx:end_idx].copy()
            windows.append(window)
        
        return windows
    
    def extract_features_from_dataset(self, data: pd.DataFrame, 
                                    include_labels: bool = True) -> pd.DataFrame:
        """
        Extract features from entire dataset using sliding windows.
        
        Args:
            data: DataFrame with sensor data and optional activity labels
            include_labels: Whether to include activity labels in output
            
        Returns:
            DataFrame with extracted features
        """
        all_features = []
        
        # Group by sample_id if available, otherwise process as single sequence
        if 'sample_id' in data.columns:
            groups = data.groupby('sample_id')
        else:
            groups = [('single_sequence', data)]
        
        for sample_id, sample_data in groups:
            # Create sliding windows for this sample
            windows = self.create_sliding_windows(sample_data)
            
            for window_idx, window in enumerate(windows):
                if len(window) < self.window_samples:
                    continue  # Skip incomplete windows
                
                # Extract features from this window
                features = self.extract_window_features(window)
                
                # Add metadata
                features['sample_id'] = sample_id
                features['window_idx'] = window_idx
                
                # Add activity label if available
                if include_labels and 'activity' in window.columns:
                    # Use the most common activity in the window
                    activity_counts = window['activity'].value_counts()
                    features['activity'] = activity_counts.index[0]
                
                all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"Extracted {len(features_df)} feature windows")
        print(f"Feature dimensions: {len(features_df.columns) - (3 if include_labels else 2)}")
        
        return features_df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be extracted.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Time-domain features for each sensor and axis
        time_features = ['mean', 'std', 'var', 'min', 'max', 'range', 'median',
                        'skewness', 'kurtosis', 'rms', 'zero_crossing_rate', 
                        'mad', 'iqr', 'energy']
        
        # Frequency-domain features for each sensor and axis
        freq_features = ['dominant_frequency', 'spectral_centroid', 'spectral_rolloff',
                        'spectral_energy', 'spectral_entropy']
        
        # Add sensor-axis specific features
        for sensor in ['acc', 'gyro']:
            for axis in ['x', 'y', 'z']:
                for feature in time_features + freq_features:
                    feature_names.append(f'{sensor}_{axis}_{feature}')
        
        # Add cross-axis features
        cross_features = ['acc_sma', 'gyro_sma', 'acc_magnitude_mean', 'acc_magnitude_std',
                         'gyro_magnitude_mean', 'gyro_magnitude_std',
                         'acc_xy_corr', 'acc_xz_corr', 'acc_yz_corr',
                         'gyro_xy_corr', 'gyro_xz_corr', 'gyro_yz_corr']
        
        feature_names.extend(cross_features)
        
        return feature_names
    
    def visualize_features(self, features_df: pd.DataFrame, activity_col: str = 'activity'):
        """
        Visualize feature distributions by activity.
        
        Args:
            features_df: DataFrame with extracted features
            activity_col: Column name containing activity labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if activity_col not in features_df.columns:
            print(f"Activity column '{activity_col}' not found in features DataFrame")
            return
        
        # Select a subset of important features for visualization
        important_features = [
            'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
            'acc_x_std', 'acc_y_std', 'acc_z_std',
            'gyro_x_std', 'gyro_y_std', 'gyro_z_std',
            'acc_sma', 'gyro_sma',
            'acc_magnitude_mean', 'gyro_magnitude_mean'
        ]
        
        # Filter features that exist in the DataFrame
        available_features = [f for f in important_features if f in features_df.columns]
        
        if not available_features:
            print("No important features found for visualization")
            return
        
        # Create subplots
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                sns.boxplot(data=features_df, x=activity_col, y=feature, ax=axes[i])
                axes[i].set_title(f'{feature}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    from data_collection import ActivityDataGenerator
    
    # Generate sample data
    generator = ActivityDataGenerator(sampling_rate=50)
    dataset = generator.generate_dataset(samples_per_activity=5, duration_per_sample=8.0)
    
    # Extract features
    extractor = FeatureExtractor(window_size=2.0, overlap=0.5, sampling_rate=50)
    features_df = extractor.extract_features_from_dataset(dataset)
    
    print(f"Feature extraction complete!")
    print(f"Dataset shape: {features_df.shape}")
    print(f"Activities: {features_df['activity'].value_counts()}")
    
    # Visualize features
    extractor.visualize_features(features_df)