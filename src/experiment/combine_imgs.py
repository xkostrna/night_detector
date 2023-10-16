from pathlib import Path
import cv2
from PIL import Image

CLASS_ID_TO_NAME = {0: "Bicycle",
                    1: "Boat",
                    2: "Bottle",
                    3: "Bus",
                    4: "Car",
                    5: "Cat",
                    6: "Chair",
                    7: "Cup",
                    8: "Dog",
                    9: "Motorbike",
                    10: "People",
                    11: "Table"}


def combine(imgs: list[Path], dest: Path):
    # Open the three existing images
    images = [Image.open(str(pth)) for pth in imgs]

    # Define the padding/margin size
    padding = 60

    # Calculate the total width and maximum height
    total_width = sum(image.width for image in images) + (len(images) - 1) * padding
    max_height = max(image.height for image in images)

    # Create a new blank image with the desired dimensions and background color
    new_image = Image.new("RGB", (total_width, max_height), color="white")

    # Paste the existing images onto the new image with padding
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, (max_height - image.height) // 2))
        x_offset += image.width + padding

    # Save the resulting image
    new_image.save(dest)


def combine_predicts(predict_paths: list[Path]):
    dest = r"F:\School\Ing\DIPLOMA\night_detector\runs\detect\500e-LRFLIP0-COPY_PASTE1\combined-predicts\combined"
    for i, pth_tuple in enumerate(zip(*(p.iterdir() for p in predict_paths))):
        combine(list(pth_tuple),
                Path(dest + f"_{i}.jpg"))


def draw_and_save_bboxes(image_path: Path, label_path: Path, output_path: Path):
    # Load the image
    image = cv2.imread(str(image_path))
    image_height, image_width, _ = image.shape

    # Read the label data from a text file in YOLO format
    with open(label_path, 'r') as file:
        for line in file:
            label = line.strip().split()
            class_id, center_x, center_y, width, height = map(float, label)

            # Convert YOLO coordinates to absolute coordinates
            x_min = int((center_x - width / 2) * image_width)
            y_min = int((center_y - height / 2) * image_height)
            x_max = int((center_x + width / 2) * image_width)
            y_max = int((center_y + height / 2) * image_height)

            # Define the bounding box color (e.g., green)
            color = (255, 255, 0)

            # Draw the bounding box on the image
            class_name = CLASS_ID_TO_NAME.get(int(class_id), 'Unknown')
            label_text = f'{class_name}'
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)
            image = cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color,
                                thickness=1)

    # Save the image with bounding boxes
    cv2.imwrite(str(output_path), image)


def main():
    # predict_paths = [
    #     Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE1-190e/predict"),
    #     Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE2-290e/predict"),
    #     Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE3-142e/predict"),
    #     Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/yolov8x-predict/predict")]
    # combine_predicts(predict_paths)
    images = Path(r"F:\School\Ing\DIPLOMA\night_detector\datasets\predict")
    labels = Path(r"F:\School\Ing\DIPLOMA\night_detector\datasets\exdark\yolo\exdark416\test\labels")
    output = Path(r"F:\School\Ing\DIPLOMA\night_detector\disp-bboxes")
    for img, label in zip(images.iterdir(), labels.iterdir()):
        draw_and_save_bboxes(img, label, output / img.name)


if __name__ == "__main__":
    main()
