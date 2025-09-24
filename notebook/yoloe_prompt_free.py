from ultralytics import YOLOE
import cv2

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf.pt")  # or select yoloe-11s/m-seg-pf.pt for different sizes

# Conduct model validation on the COCO128-seg example dataset
metrics = model.val(data="coco128-seg.yaml", single_cls=True)

image_path = "image_2.png"

# Run detection on the given image
results = model.predict(
    image_path,
    conf=0.9,
    iou=0.45,
    max_det=100,
    )

# Show results
results[0].show()

for r in results:
    img = r.plot()  # BGR ndarray with boxes + labels
    cv2.imshow("YOLOE prompt-free detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()