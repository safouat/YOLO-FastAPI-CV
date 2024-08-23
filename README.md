# FastAPI YOLO Object Detection API

This repository contains a FastAPI application that provides endpoints for object detection and classification using the YOLO model. The application can process both images and videos, returning predictions or annotated media files.

## Features

- **Image Classification**: Classify objects in an image and return the predictions.
- **Image Detection**: Detect objects in an image, annotate bounding boxes, and return the annotated image.
- **Video Classification**: Classify objects in each frame of a video and stream the results in JSON format.
- **Video Detection**: Detect objects in each frame of a video, annotate bounding boxes, and stream the annotated video.

## Installation

### Prerequisites

- Docker

### Setup and Running with Docker

1. **Clone the repository**:

   ```bash
   git clone

   ```

2. **Build the Docker image**:

   ```bash
   docker build -t yolo-fastapi-app .
   ```

3. **Run the Docker container**:

   ```bash
   docker run -d -p 8000:8000 yolo-fastapi-app
   ```

4. **Access the API**:
   - The API will be available at `http://127.0.0.1:8000` or `http://localhost:8000`.

### Endpoints

#### 1. `/classify/v1` - Image Classification

- **Method**: `POST`
- **Description**: Classify objects in an uploaded image.
- **Input**: `picture` (image file)
- **Output**: JSON object with predictions.

#### 2. `/detect/v1` - Image Detection

- **Method**: `POST`
- **Description**: Detect objects in an uploaded image and return an annotated image.
- **Input**: `picture` (image file)
- **Output**: JPEG image with bounding boxes.

#### 3. `/classify/v2` - Video Classification

- **Method**: `POST`
- **Description**: Classify objects in each frame of an uploaded video and stream the results in JSON format.
- **Input**: `video` (video file)
- **Output**: JSON stream with frame-wise predictions.

#### 4. `/detect_video/v1` - Video Detection

- **Method**: `POST`
- **Description**: Detect objects in each frame of an uploaded video, annotate the frames with bounding boxes, and stream the annotated video.
- **Input**: `video` (video file)
- **Output**: Streamed video with bounding boxes.

### Example Usage with `curl`

- **Image Classification**:

- **Image Detection**:

- **Video Classification**:

- **Video Detection**:

### Notes

- Make sure the `best.pt` YOLO model weights file is placed in the root directory before building the Docker image.
- The Docker container exposes port `8000` by default, but you can change this by modifying the Docker run command.
