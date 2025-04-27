
# Real-Time Sign Language Converter (RSLC)

RSLC is a machine learning-based application designed to bridge the communication gap between mute and deaf individuals and those who do not know sign language. It uses a hybrid model combining Convolutional Neural Networks (CNN) and Deep Belief Networks (DBN) to convert sign language gestures into text in real time.

## Features

- Real-time sign language gesture recognition.
- Converts sign language into readable text.
- Built with deep learning techniques using CNN and DBN.
- Designed for accessibility, especially for mute and deaf communities.

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Machine Learning Algorithms**: Convolutional Neural Networks (CNN), Deep Belief Networks (DBN)
- **Libraries Used**: OpenCV (for capturing gestures via webcam), NumPy, Pandas

## Setup and Installation

### Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rslc.git
   cd rslc
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models (if applicable) or train your own model (instructions below).

4. Run the application:
   ```bash
   python main.py
   ```

### Training the Model

If you want to train the model from scratch, you will need a dataset containing sign language gestures. You can use publicly available datasets like the American Sign Language (ASL) dataset.

1. Prepare the dataset and organize the images into classes based on the gesture labels.
2. Modify the training script `train_model.py` to load the dataset.
3. Train the model:
   ```bash
   python train_model.py
   ```
4. The trained model will be saved as `sign_language_model.h5` (or as specified).

## Usage

1. Launch the application using `python main.py`.
2. The application will open a webcam feed and begin detecting hand gestures.
3. The recognized gestures will be converted into text and displayed on the screen.
4. The system can be paused, and the webcam can be turned off using the provided interface.

## Contributing

Feel free to contribute to the project! Fork the repository, create a branch for your changes, and submit a pull request.

### Steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add feature'`).
5. Push to the branch (`git push origin feature-name`).
6. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- TensorFlow and Keras for deep learning frameworks.
- OpenCV for real-time image capture.
- The open-source community for contributing datasets and tutorials.


