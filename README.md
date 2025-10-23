# Modeling Human Activity States Using Hidden Markov Models

A comprehensive implementation of Hidden Markov Models (HMM) for human activity recognition using accelerometer and gyroscope sensor data. This project demonstrates the complete pipeline from data collection to model evaluation for recognizing activities like standing, walking, jumping, and being still.

---

## Project Overview

This project implements a complete system for human activity recognition that:
- Collects or generates realistic sensor data from accelerometer and gyroscope
- Extracts comprehensive time-domain and frequency-domain features
- Trains Hidden Markov Models to recognize activity patterns
- Uses the Viterbi algorithm for optimal state sequence decoding
- Evaluates model performance with detailed metrics
- Provides visualizations of results and model parameters

**Activities Recognized:**
- **Standing**: Stationary upright position
- **Walking**: Regular locomotion with periodic patterns
- **Jumping**: High-energy vertical movements
- **Still**: No movement (device at rest)

---

## Features

- **Realistic Data Generation**: Synthetic sensor data that mimics real smartphone sensors
- **Comprehensive Feature Extraction**: 100+ time and frequency domain features
- **HMM Implementation**: Full Hidden Markov Model with Gaussian emissions
- **Viterbi Decoding**: Optimal activity sequence prediction
- **Performance Evaluation**: Sensitivity, specificity, and accuracy metrics
- **Rich Visualizations**: Transition matrices, activity sequences, and feature distributions
- **Real Data Support**: Easy integration with actual sensor data from mobile apps

---

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `numpy`, `pandas`, `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities
- `hmmlearn` - Hidden Markov Model implementation
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `librosa` - Signal processing
- `jupyter` - Interactive notebooks

---

## Quick Start

### Option 1: Run the Demo Script
```bash
python demo_hmm_activity.py
```

### Option 2: Use the Jupyter Notebook
```bash
jupyter notebook notebooks/human_activity_hmm.ipynb
```

### Option 3: Use Individual Modules
```python
from src.data_collection import ActivityDataGenerator
from src.feature_extraction import FeatureExtractor
from src.hmm_model import ActivityHMM

# Generate data
generator = ActivityDataGenerator(sampling_rate=50)
dataset = generator.generate_dataset(samples_per_activity=12)

# Extract features
extractor = FeatureExtractor(window_size=2.0, overlap=0.5)
features = extractor.extract_features_from_dataset(dataset)

# Train HMM
hmm_model = ActivityHMM(n_states=4)
hmm_model.fit(features)

# Evaluate
metrics = hmm_model.evaluate(test_data)
```

---

## Data Collection

### Using Real Sensor Data

1. **Install a sensor logging app:**
   - **Sensor Logger** (iOS/Android) - Recommended
   - **Physics Toolbox Accelerometer** (Android)

2. **Configure settings:**
   - Sampling rate: 50-100 Hz
   - Sensors: Accelerometer (x,y,z) + Gyroscope (x,y,z)

3. **Record activities:**
   - Each activity: 5-10 seconds
   - Repeat: ~12 times per activity
   - Export as CSV files

4. **Load real data:**
```python
from src.data_collection import load_real_sensor_data
real_data = load_real_sensor_data('path/to/sensor_data.csv')
```

### Using Synthetic Data

The project includes a sophisticated synthetic data generator that creates realistic sensor patterns:

```python
generator = ActivityDataGenerator(sampling_rate=50)
dataset = generator.generate_dataset(
    samples_per_activity=12,
    duration_per_sample=8.0
)
```

---

## Feature Extraction

The system extracts comprehensive features from sliding windows of sensor data:

### Time-Domain Features (per axis)
- Statistical: mean, std, variance, min, max, range, median
- Higher-order: skewness, kurtosis
- Energy-based: RMS, energy, signal magnitude area (SMA)
- Temporal: zero-crossing rate, mean absolute deviation

### Frequency-Domain Features (per axis)
- Spectral: dominant frequency, centroid, rolloff, energy
- Information: spectral entropy
- FFT-based: frequency components and magnitudes

### Cross-Axis Features
- Vector magnitudes for accelerometer and gyroscope
- Correlations between axes
- Combined sensor features

**Total Features:** ~100+ features per window

---

## Hidden Markov Model

### Model Architecture
- **Hidden States**: 4 states corresponding to activities
- **Observations**: Feature vectors from sensor data
- **Emission Model**: Gaussian distributions with full covariance
- **Transition Model**: Learned probability matrix between activities

### Key Algorithms
- **Training**: Baum-Welch algorithm (via hmmlearn)
- **Decoding**: Viterbi algorithm for optimal state sequences
- **Evaluation**: Forward-backward algorithm for likelihood

### Model Components
```python
hmm_model = ActivityHMM(
    n_states=4,                    # One per activity
    covariance_type="full",        # Full covariance matrices
    random_state=42               # Reproducible results
)
```

---

## Evaluation Metrics

The system provides comprehensive evaluation following the project requirements:

| Metric | Description |
|--------|-------------|
| **Sensitivity** | True positive rate (recall) per activity |
| **Specificity** | True negative rate per activity |
| **Overall Accuracy** | Correct predictions / total predictions |
| **Precision** | Positive predictive value per activity |
| **F1-Score** | Harmonic mean of precision and recall |

### Sample Results Table
```
State (Activity)  | Number of Samples | Sensitivity | Specificity | Overall Accuracy
------------------|-------------------|-------------|-------------|------------------
standing          | 45                | 0.889       | 0.967       | 0.875
walking           | 48                | 0.917       | 0.956       | 0.875
jumping           | 42                | 0.952       | 0.989       | 0.875
still             | 39                | 0.821       | 0.978       | 0.875
```

---

## Visualizations

The project includes rich visualizations:

1. **Sensor Data Plots**: Raw accelerometer and gyroscope signals
2. **Feature Distributions**: Box plots showing feature separability
3. **Transition Matrix**: Heatmap of state transition probabilities
4. **Prediction Sequences**: True vs predicted activity over time
5. **Confusion Matrix**: Classification performance breakdown

---

## Project Structure

```
workspace/
├── src/                          # Source code modules
│   ├── data_collection.py        # Data generation and loading
│   ├── feature_extraction.py     # Feature extraction pipeline
│   └── hmm_model.py             # HMM implementation and evaluation
├── notebooks/                    # Jupyter notebooks
│   └── human_activity_hmm.ipynb # Complete analysis notebook
├── data/                         # Generated datasets
│   ├── synthetic_activity_data.csv
│   └── extracted_features.csv
├── results/                      # Model outputs and metrics
│   ├── trained_hmm_model.pkl
│   ├── evaluation_results.csv
│   └── detailed_metrics.csv
├── demo_hmm_activity.py         # Demonstration script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

---

## Advanced Usage

### Custom Feature Selection
```python
# Use specific features only
feature_columns = ['acc_x_std', 'acc_y_std', 'acc_z_std', 'acc_sma']
hmm_model.fit(train_data, feature_columns=feature_columns)
```

### Model Persistence
```python
# Save trained model
hmm_model.save_model('my_model.pkl')

# Load model later
new_model = ActivityHMM()
new_model.load_model('my_model.pkl')
```

### Real-time Prediction
```python
# Extract features from new data window
new_features = extractor.extract_window_features(new_window)

# Predict activity
predicted_state, log_prob = hmm_model.predict(new_features)
activity = hmm_model.state_to_activity[predicted_state[0]]
```

---

## Performance Optimization

### Tips for Better Results

1. **Data Quality**:
   - Use consistent phone orientation
   - Ensure stable sampling rate
   - Collect diverse samples per activity

2. **Feature Engineering**:
   - Experiment with window sizes (1-4 seconds)
   - Try different overlap ratios (0.25-0.75)
   - Add domain-specific features

3. **Model Tuning**:
   - Adjust covariance type (`"full"`, `"diag"`, `"tied"`)
   - Experiment with number of Gaussian components
   - Use cross-validation for hyperparameter selection

---

## Real-World Applications

- **Health Monitoring**: Track daily activity patterns and exercise
- **Fitness Apps**: Automatic workout recognition and counting
- **Smart Home**: Context-aware automation based on user activity
- **Elderly Care**: Fall detection and activity monitoring
- **Sports Analytics**: Performance analysis and technique assessment

---

## Limitations and Future Work

### Current Limitations
- Synthetic data may not capture all real-world variations
- Limited to four basic activities
- Assumes consistent phone placement and orientation
- No handling of transition activities or unknown states

### Potential Improvements
1. **More Activities**: Add running, cycling, climbing stairs
2. **Hierarchical HMMs**: Model sub-activities and transitions
3. **Deep Learning**: Compare with LSTM/CNN approaches
4. **Online Learning**: Adapt model to individual users
5. **Sensor Fusion**: Incorporate additional sensors (magnetometer, barometer)
6. **Robust Features**: Handle orientation and placement variations

---

## Contributing

Contributions are welcome! Areas for improvement:
- Additional feature extraction methods
- Support for more sensor types
- Real-time processing optimizations
- Mobile app integration
- Comparison with other ML approaches

---

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
2. Lara, O. D., & Labrador, M. A. (2013). A survey on human activity recognition using wearable sensors.
3. Bulling, A., et al. (2014). A tutorial on human activity recognition using body-worn inertial sensors.

---

## License

This project is provided for educational purposes. Please cite appropriately if used in academic work.

---

**Happy Activity Recognition!** 🏃‍♀️📱🤖 Transformers library and pre-trained models
- **Gradio**: For the web interface framework

---

**Happy Healthcare Chatbotting! **
