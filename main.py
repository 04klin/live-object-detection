import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

# where the zone is going to be set to count up the objects
ZONE_POLYGON = np.array([
  [0, 0],
  [.5, 0],
  [.5, 1],
  [0, 1]
])

# Sets the webcam resolution
def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
    "--webcam-resolution", 
    default=[1280,720],
    nargs=2, 
    type=int
  )
  args = parser.parse_args()
  return args

def main():
  args = parse_arguments()
  frame_width, frame_height = args.webcam_resolution

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

  # Set model
  model = YOLO("yolov8l.pt")

  box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness=2,
    text_scale=1
  )

  zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
  zone = sv.PolygonZone(polygon =zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
  zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, 
    color=sv.Color.red(),
    text_thickness=4,
    text_scale=2
    )

  while True:
    ret, frame = cap.read()

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    # Removes person object from detection
    detections = detections[detections.class_id != 0]
    # This line of code only allows apples
    # detections = detections[detections.class_id == 47]
    labels = [
      f"{model.model.names[class_id]} {confidence:0.2f}"
      for _, confidence, class_id, _
      in detections
    ]
    frame = box_annotator.annotate(
      scene=frame, 
      detections=detections, 
      labels=labels
      )

    zone.trigger(detections=detections)
    frame = zone_annotator.annotate(scene=frame)
    cv2.imshow("yolov8", frame)

    if (cv2.waitKey(15) == 27):
      break

if __name__ == "__main__":
  main()