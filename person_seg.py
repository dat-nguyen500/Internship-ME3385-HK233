# Đoạn mã chạy mô hình AI tiền huấn luyện của YOLOv8 thực hiện 
# phân đoạn (segmentation) trên các lớp liên quan tới người (Person)
# để trích xuất thông tin pixel của gần nhất của người với hàng rào ảo

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load segmentation model
model = YOLO("yolov8n-seg.pt")
names = model.model.names

# Find the class indices for 'person' and 'book' in the names dictionary
person_class_index = None
for index, class_name in names.items():
    if class_name == 'person':
        person_class_index = index

# Check if class indices were found
if person_class_index is None:
    raise ValueError("Class indices for 'person' and 'book' were not found in the model names.")

# Open video capture
cap = cv2.VideoCapture(2)

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Get predictions from the model
    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            # Check if the detected class is 'person' or 'book'
            if int(cls) == person_class_index:
                annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True), det_label=names[int(cls)])

    # Display the result
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
