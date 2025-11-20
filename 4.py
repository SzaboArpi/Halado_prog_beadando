import cv2
from ultralytics import YOLO

# Instantiate the YOLO model (already done in previous step)
model = YOLO("yolov8s.pt")

# Initialize video capture from a video file (already done in previous step)
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully (already done in previous step)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# 1. Initialize an empty set to store unique track IDs of detected people
unique_person_ids = set()

# Get the index for the 'person' class from the model.names dictionary
try:
    person_class_id = [ key for key, val in model.names.items() if val =='person'][0]
except ValueError:
    print("'person' class not found in the model's names.")
    exit()

# Get video properties for output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize output video writer
output_video_path = 'output_video_with_detections.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    exit()

# 2. Start a while loop that continues as long as cap.isOpened() is True
while cap.isOpened():
    # 3. Read a frame from the video capture object
    ret, frame = cap.read()

    # 4. Check if the frame was read successfully
    if not ret:
        print("End of video stream or an error occurred.")
        break

    # 5. Perform object detection and tracking on the current frame
    # Only track 'person' class with persist=True for continuous tracking
    results = model.track(frame, persist=True, classes=person_class_id, verbose=False)[0]

    # Initialize current frame's detected track IDs
    current_frame_person_ids = set()

    # 6. Iterate through the boxes attribute of the results object
    if results.boxes.id is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            track_id = int(results.boxes.id[i].item())

            # Ensure it's a 'person' detection
            if cls_id == person_class_id:
                current_frame_person_ids.add(track_id)
                unique_person_ids.add(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green color

                # Add track ID and confidence score as text
                label = f"ID: {track_id} Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add total unique persons count to the frame
    total_unique_text = f"Total Unique Persons: {len(unique_person_ids)}"
    cv2.putText(frame, total_unique_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # White text

    # Write the annotated frame to the output video file
    out.write(frame)

# 8. Release the video capture object and the video writer object
cap.release()
out.release()
print(f"Total unique persons detected throughout the video: {len(unique_person_ids)}")
print(f"Annotated video saved to: {output_video_path}")
