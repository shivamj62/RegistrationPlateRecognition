# Registration Plate Recognition

This project focuses on detecting and recognizing license plates in video footage. It leverages YOLO models for object detection, SORT for object tracking, and easyocr for optical character recognition (OCR).  Missing data in the tracking is handled using interpolation.

## Features and Functionality

*   **Vehicle Detection:** Uses YOLOv8n to detect vehicles in a video.
*   **License Plate Detection:** Uses a custom-trained YOLO model (`./models/license_plate_detector.pt`) to detect license plates.
*   **Object Tracking:** Implements object tracking using the SORT algorithm to track vehicles across frames.
*   **License Plate Recognition:** Employs easyocr to extract text from detected license plates.
*   **Missing Data Interpolation:** Includes `add_missing_data.py` to interpolate bounding boxes for frames where data is missing.  This improves the robustness of tracking and recognition.
*   **CSV Output:** Writes the detected car and license plate information, including bounding boxes, recognized text, and confidence scores, to a CSV file (`test.csv`).  The `add_missing_data.py` script creates `test_interpolated.csv` which has interpolated data and is used for visualization.
*   **Visualization:** Provides a `visualize.py` script to generate a video with bounding boxes around detected vehicles and license plates and the recognized license plate text displayed.

## Technology Stack

*   **Python 3.x**
*   **Ultralytics YOLO:** For object detection ([https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics))
*   **OpenCV (cv2):** For video processing and image manipulation.
*   **NumPy:** For numerical computations.
*   **SciPy:** For interpolation of missing bounding box data.
*   **easyocr:** For Optical Character Recognition.
*   **pandas:** For data manipulation and CSV file handling in the visualization script.
*   **SORT (Simple Online and Realtime Tracking):** For object tracking.

## Prerequisites

Before running the code, ensure you have the following installed:

1.  **Python 3.x:**  Verify with `python --version` or `python3 --version`.

2.  **Required Python Packages:** Install using pip:

    ```bash
    pip install opencv-python numpy ultralytics easyocr pandas scipy
    ```

3.  **YOLO Models:** Download the necessary YOLO models:
    *   `yolov8n.pt`: Download from the official Ultralytics YOLOv8 repository.  This is the general object detection model. The code automatically downloads this on first run.
    *   `license_plate_detector.pt`: This is the custom trained model for detecting license plates. Place this in the `./models/` directory. *Important: You'll need to train this model yourself, or obtain a pre-trained version.*

## Installation Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/shivamj62/RegistrationPlateRecognition.git
    cd RegistrationPlateRecognition
    ```

2.  **Create `./models/` directory:**

    ```bash
    mkdir models
    ```

3.  **Place `license_plate_detector.pt`:** Download or train your `license_plate_detector.pt` and place it inside the `models` directory.

## Usage Guide

1.  **Run `main.py`:** This script processes the video, detects vehicles and license plates, performs OCR, and saves the results to `test.csv`.

    ```bash
    python main.py
    ```

    *   This script uses `./sample.mp4` as the input video.  You may need to modify the `cap = cv2.VideoCapture('./sample.mp4')` line in `main.py` to point to your video file.  Also, the results are written to `./test.csv`.

2. **Interpolate Missing Data:**  Run `add_missing_data.py` to interpolate bounding boxes when frames are missing for a specific car. This will read `./test.csv` and create `./test_interpolated.csv`.
    ```bash
    python add_missing_data.py
    ```

3.  **Visualize the Results:**  Run `visualize.py` to generate a video (`out.mp4`) with bounding boxes and recognized license plate text.  This script reads `./test_interpolated.csv`.

    ```bash
    python visualize.py
    ```

## File Descriptions

*   **`main.py`:** The main script for vehicle and license plate detection, tracking, and OCR.
*   **`util.py`:** Contains utility functions for:
    *   Writing results to a CSV file (`write_csv`).
    *   Checking license plate format (`license_complies_format`).
    *   Formatting license plate text (`format_license`).
    *   Reading license plate text using easyocr (`read_license_plate`).
    *   Associating license plates with vehicles (`get_car`).
*   **`sort/sort.py`:**  Implementation of the SORT (Simple Online and Realtime Tracking) algorithm.
*   **`add_missing_data.py`:** Script for interpolating missing bounding box data to improve tracking accuracy. It reads from `test.csv` and writes to `test_interpolated.csv`.
*   **`visualize.py`:** Script to visualize the results by drawing bounding boxes and displaying license plate numbers on the video.  Requires the `test_interpolated.csv` file.
*   **`testincv.py`:**  A simple test script to verify that OpenCV is installed correctly.  It creates a blank white image and displays it.
*   **`./models/license_plate_detector.pt`:** The YOLOv8 model (PyTorch format) trained to detect license plates.
*   **`./sample.mp4`:** Sample video file.

## API Documentation

This project doesn't expose a traditional API. However, the core functionality can be accessed and extended through the Python scripts and their functions.

*   **`util.py` Functions:**  The functions within `util.py` provide modular access to specific tasks such as license plate formatting and reading. You can import this module into other scripts to reuse these functionalities. For example, the `read_license_plate` function can be used independently to extract text from a license plate image.

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Test your changes thoroughly.
5.  Submit a pull request with a clear description of your changes.

## License Information

No license is specified in the repository.  All rights are reserved unless otherwise stated.

## Contact/Support Information

For questions, issues, or support, please contact shivamj62 through GitHub.