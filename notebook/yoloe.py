from ultralytics import YOLOE
import cv2

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["black chair"]
model.set_classes(names, model.get_text_pe(names))

image_path = "image_2.png"

# Run detection on the given image
results = model.predict(
    image_path,
    conf=0.01,
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