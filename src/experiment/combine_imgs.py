from pathlib import Path

from PIL import Image


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


def main():
    predict_paths = [
        Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE1-190e/predict"),
        Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE2-290e/predict"),
        Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/TRENOVANIE3-142e/predict"),
        Path("F:/School/Ing/DIPLOMA/night_detector/runs/detect/500e-LRFLIP0-COPY_PASTE1/yolov8x-predict/predict")]
    combine_predicts(predict_paths)


if __name__ == "__main__":
    main()
