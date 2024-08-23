from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import tempfile
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
import json
import os 
import asyncio 
import supervision as sv

app = FastAPI()

model = YOLO('best.pt')

@app.post("/classify/v1")
async def post_picture(picture: UploadFile ):
    image_bytes = await picture.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    results = model.predict(source=img, iou=0.5)
    
    serialized_results = []
    
    for result in results:  
        for box in result.boxes:  
            serialized_results.append({
                'class_id': int(box.cls),  
                'label': result.names[int(box.cls)], 
                'confidence': float(box.conf),  
                'bbox': box.xyxy.tolist() 
            })
    
    return {"predictions": serialized_results}

@app.post("/detect/v1")
async def detect_boxes(picture: UploadFile):
    # Read the uploaded image file
    img_bytes = await picture.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model.predict(source=img, iou=0.5, conf=0.1)

    detection_boxes = []
    detection_confidences = []
    detection_labels = []

    for result in results:  
        for box in result.boxes:  
            class_id = int(box.cls.item())
            confidence = box.conf.numpy()
            bbox = box.xyxy.numpy().tolist()  
            
            for box_item, conf_item in zip(bbox, confidence):
                detection_boxes.append(box_item)
                detection_confidences.append(float(conf_item))
                detection_labels.append(class_id) 

    detection_boxes = np.array(detection_boxes)
    detection_confidences = np.array(detection_confidences)
    detection_labels = np.array(detection_labels)

    print(detection_boxes.shape)  
    print(detection_confidences.shape) 
    print(detection_labels.shape)  

    detections = sv.Detections(
        xyxy=detection_boxes,
        confidence=detection_confidences,
        class_id=detection_labels
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=img.copy(),
        detections=detections
    )

    success, img_encoded = cv2.imencode('.jpg', annotated_image)
    
    if not success:
        return {"error": "Failed to encode image"}

    img_buffer = io.BytesIO(img_encoded.tobytes())

    return StreamingResponse(img_buffer, media_type="image/jpeg")

@app.post("/classify/v2")
async def post_video(video: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await video.read())
            temp_file.flush()
            temp_file.seek(0)

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Could not open video file.")

            frame_index = 0

            async def generate():
                nonlocal frame_index
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        try:
                            predictions = model.predict(source=frame, iou=0.5)
                            frame_predictions = []

                            for result in predictions:
                                for box in result.boxes:
                                    frame_predictions.append({
                                        'class_id': int(box.cls),
                                        'label': result.names[int(box.cls)],
                                        'confidence': float(box.conf),
                                        'bbox': box.xyxy.tolist()
                                    })

                            # Send frame-wise results
                            result_json = json.dumps({
                                'frame_index': frame_index,
                                'predictions': frame_predictions
                            }) + "\n"
                            yield result_json
                            frame_index += 1

                            await asyncio.sleep(0) 
                        except Exception as e:
                            error_json = json.dumps({
                                'frame_index': frame_index,
                                'error': str(e)
                            }) + "\n"
                            yield error_json

                finally:
                    cap.release()
                    os.remove(temp_file.name)

            return StreamingResponse(generate(), media_type="application/json", headers={"Cache-Control": "no-cache"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/detect_video/v1")
async def detect_video(video: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await video.read())
            temp_file.flush()
            temp_file.seek(0)

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Could not open video file.")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = cv2.VideoWriter(temp_file.name + "_output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            async def generate():
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        try:
                            results = model.predict(source=frame, iou=0.5)

                            # Draw bounding boxes on the frame
                            detection_boxes = []
                            detection_confidences = []
                            detection_labels = []

                            for result in results:
                                for box in result.boxes:
                                    class_id = int(box.cls.item())
                                    confidence = box.conf.numpy()
                                    bbox = box.xyxy.numpy().tolist()
                                    
                                    for box_item, conf_item in zip(bbox, confidence):
                                        detection_boxes.append(box_item)
                                        detection_confidences.append(float(conf_item))
                                        detection_labels.append(class_id)

                            detection_boxes = np.array(detection_boxes)
                            detection_confidences = np.array(detection_confidences)
                            detection_labels = np.array(detection_labels)

                            detections = sv.Detections(
                                xyxy=detection_boxes,
                                confidence=detection_confidences,
                                class_id=detection_labels
                            )

                            bounding_box_annotator = sv.BoundingBoxAnnotator()

                            annotated_image = bounding_box_annotator.annotate(
                                scene=frame.copy(),
                                detections=detections
                            )

                            output.write(annotated_image)

                            ret, jpeg = cv2.imencode('.jpg', annotated_image)
                            if not ret:
                                continue

                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                        except Exception as frame_error:
                            print(f"Error processing frame: {frame_error}")

                except Exception as e:
                    print(f"Error in video processing: {e}")
                finally:
                    cap.release()
                    output.release()
                    os.remove(temp_file.name)


            return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))