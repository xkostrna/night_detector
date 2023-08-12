import cv2
import matplotlib.pyplot as plt

BOX_COLOR = (255, 0, 0)    # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_yolo_bboxes(img,
                          bboxes: list[list[float]],
                          category_ids: list[int],
                          category_id_to_name: dict[int, str]):
    height, width, _ = img.shape

    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        x_center, y_center, w, h = bbox

        # Convert normalized YOLO coordinates to pixel coordinates
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=2)

        # Draw class name
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
    plt.show()
