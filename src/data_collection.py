"""
Data Collection Module for Human Activity Recognition
Generates synthetic sensor data that mimics real accelerometer and gyroscope readings
for different human activities: standing, walking, jumping, and still.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class ActivityDataGenerator:
    """
    Generates synthetic accelerometer and gyroscope data for different human activities.
    This simulates data that would be collected from smartphone sensors.
    """
    
    def __init__(self, sampling_rate: int = 50):
        """
        Initialize the data generator.
        
        Args:
            sampling_rate: Samples per second (Hz)
        """
        self.sampling_rate = sampling_rate
        self.activities = ['standing', 'walking', 'jumping', 'still']
        
        # Activity-specific parameters for realistic signal generation
        self.activity_params = {
            'standing': {
                'acc_mean': [0.0, 0.0, 9.8],  # Gravity dominates Z-axis
                'acc_std': [0.5, 0.5, 0.3],   # Small variations
                'gyro_mean': [0.0, 0.0, 0.0], # Minimal rotation
                'gyro_std': [0.1, 0.1, 0.1],  # Small noise
                'frequency': 0.5  # Low frequency variations
            },
            'walking': {
                'acc_mean': [0.0, 0.0, 9.8],
                'acc_std': [2.0, 1.5, 1.0],   # Higher variations
                'gyro_mean': [0.0, 0.0, 0.0],
                'gyro_std': [0.5, 0.3, 0.2],  # More rotation
                'frequency': 2.0  # Walking frequency ~2 Hz
            },
            'jumping': {
                'acc_mean': [0.0, 0.0, 9.8],
                'acc_std': [3.0, 3.0, 5.0],   # High variations, especially Z
                'gyro_mean': [0.0, 0.0, 0.0],
                'gyro_std': [1.0, 1.0, 0.5],  # Significant rotation
                'frequency': 3.0  # Jumping frequency ~3 Hz
            },
            'still': {
                'acc_mean': [0.0, 0.0, 9.8],
                'acc_std': [0.1, 0.1, 0.1],   # Very small variations
                'gyro_mean': [0.0, 0.0, 0.0],
                'gyro_std': [0.05, 0.05, 0.05],  # Minimal noise
                'frequency': 0.1  # Almost no periodic motion
            }
        }
    
    def generate_activity_data(self, activity: str, duration: float, 
                             start_time: datetime = None) -> pd.DataFrame:
        """
        Generate synthetic sensor data for a specific activity.
        
        Args:
            activity: Activity name ('standing', 'walking', 'jumping', 'still')
            duration: Duration in seconds
            start_time: Start timestamp (defaults to current time)
            
        Returns:
            DataFrame with timestamp, accelerometer (x,y,z), and gyroscope (x,y,z) data
        """
        if activity not in self.activities:
            raise ValueError(f"Activity must be one of {self.activities}")
        
        if start_time is None:
            start_time = datetime.now()
        
        # Calculate number of samples
        n_samples = int(duration * self.sampling_rate)
        
        # Generate time array
        time_delta = timedelta(seconds=1/self.sampling_rate)
        timestamps = [start_time + i * time_delta for i in range(n_samples)]
        
        # Get activity parameters
        params = self.activity_params[activity]
        
        # Generate time array for signal generation
        t = np.linspace(0, duration, n_samples)
        
        # Generate accelerometer data with periodic components
        acc_x = (params['acc_mean'][0] + 
                np.random.normal(0, params['acc_std'][0], n_samples) +
                0.5 * np.sin(2 * np.pi * params['frequency'] * t))
        
        acc_y = (params['acc_mean'][1] + 
                np.random.normal(0, params['acc_std'][1], n_samples) +
                0.3 * np.cos(2 * np.pi * params['frequency'] * t))
        
        acc_z = (params['acc_mean'][2] + 
                np.random.normal(0, params['acc_std'][2], n_samples) +
                0.4 * np.sin(2 * np.pi * params['frequency'] * t + np.pi/4))
        
        # Generate gyroscope data
        gyro_x = (params['gyro_mean'][0] + 
                 np.random.normal(0, params['gyro_std'][0], n_samples) +
                 0.1 * np.sin(2 * np.pi * params['frequency'] * t))
        
        gyro_y = (params['gyro_mean'][1] + 
                 np.random.normal(0, params['gyro_std'][1], n_samples) +
                 0.1 * np.cos(2 * np.pi * params['frequency'] * t))
        
        gyro_z = (params['gyro_mean'][2] + 
                 np.random.normal(0, params['gyro_std'][2], n_samples) +
                 0.05 * np.sin(2 * np.pi * params['frequency'] * t))
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'activity': activity
        })
        
        return data
    
    def generate_dataset(self, samples_per_activity: int = 12, 
                        duration_per_sample: float = 8.0) -> pd.DataFrame:
        """
        Generate a complete dataset with multiple samples for each activity.
        
        Args:
            samples_per_activity: Number of samples to generate per activity
            duration_per_sample: Duration of each sample in seconds
            
        Returns:
            Complete dataset with all activities
        """
        all_data = []
        current_time = datetime.now()
        
        for activity in self.activities:
            print(f"Generating {samples_per_activity} samples for {activity}...")
            
            for sample_idx in range(samples_per_activity):
                # Add some time gap between samples
                sample_start = current_time + timedelta(seconds=sample_idx * 15)
                
                sample_data = self.generate_activity_data(
                    activity=activity,
                    duration=duration_per_sample,
                    start_time=sample_start
                )
                
                # Add sample identifier
                sample_data['sample_id'] = f"{activity}_{sample_idx:02d}"
                all_data.append(sample_data)
        
        # Combine all data
        complete_dataset = pd.concat(all_data, ignore_index=True)
        
        print(f"Generated dataset with {len(complete_dataset)} total data points")
        print(f"Activities: {complete_dataset['activity'].value_counts().to_dict()}")
        
        return complete_dataset
    
    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """Save dataset to CSV file."""
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def visualize_sample_data(self, dataset: pd.DataFrame, activity: str, 
                            sample_id: str = None):
        """
        Visualize accelerometer and gyroscope data for a specific activity sample.
        
        Args:
            dataset: Complete dataset
            activity: Activity to visualize
            sample_id: Specific sample ID (if None, uses first sample)
        """
        # Filter data
        activity_data = dataset[dataset['activity'] == activity]
        
        if sample_id:
            activity_data = activity_data[activity_data['sample_id'] == sample_id]
        else:
            # Use first sample
            first_sample = activity_data['sample_id'].iloc[0]
            activity_data = activity_data[activity_data['sample_id'] == first_sample]
        
        if activity_data.empty:
            print(f"No data found for activity: {activity}")
            return
        
        # Create time axis (relative to start)
        time_seconds = np.arange(len(activity_data)) / self.sampling_rate
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot accelerometer data
        ax1.plot(time_seconds, activity_data['acc_x'], label='Acc X', alpha=0.8)
        ax1.plot(time_seconds, activity_data['acc_y'], label='Acc Y', alpha=0.8)
        ax1.plot(time_seconds, activity_data['acc_z'], label='Acc Z', alpha=0.8)
        ax1.set_title(f'Accelerometer Data - {activity.title()}')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot gyroscope data
        ax2.plot(time_seconds, activity_data['gyro_x'], label='Gyro X', alpha=0.8)
        ax2.plot(time_seconds, activity_data['gyro_y'], label='Gyro Y', alpha=0.8)
        ax2.plot(time_seconds, activity_data['gyro_z'], label='Gyro Z', alpha=0.8)
        ax2.set_title(f'Gyroscope Data - {activity.title()}')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def load_real_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load real sensor data from CSV file (for use with actual Sensor Logger data).
    
    Args:
        filepath: Path to CSV file from Sensor Logger app
        
    Returns:
        Processed DataFrame with standardized column names
    """
    try:
        # Try to load the data
        data = pd.read_csv(filepath)
        
        # Common column name mappings for different sensor apps
        column_mappings = {
            # Sensor Logger format
            'ACCELEROMETER X (m/s²)': 'acc_x',
            'ACCELEROMETER Y (m/s²)': 'acc_y', 
            'ACCELEROMETER Z (m/s²)': 'acc_z',
            'GYROSCOPE X (rad/s)': 'gyro_x',
            'GYROSCOPE Y (rad/s)': 'gyro_y',
            'GYROSCOPE Z (rad/s)': 'gyro_z',
            'Time (s)': 'timestamp',
            
            # Physics Toolbox format
            'time': 'timestamp',
            'ax': 'acc_x',
            'ay': 'acc_y',
            'az': 'acc_z',
            'gx': 'gyro_x',
            'gy': 'gyro_y',
            'gz': 'gyro_z'
        }
        
        # Rename columns if they exist
        data = data.rename(columns=column_mappings)
        
        # Ensure we have the required columns
        required_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}")
            print(f"Available columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    generator = ActivityDataGenerator(sampling_rate=50)
    
    # Generate dataset
    dataset = generator.generate_dataset(samples_per_activity=12, duration_per_sample=8.0)
    
    # Save dataset
    generator.save_dataset(dataset, "/workspace/data/synthetic_activity_data.csv")
    
    # Visualize some samples
    for activity in generator.activities:
        generator.visualize_sample_data(dataset, activity)