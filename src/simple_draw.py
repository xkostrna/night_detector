from PIL import Image, ImageDraw


def draw_coco(img: str, upper_left_corner: tuple[int, int], width: int, height: int):
    # Open an image file
    img = Image.open(img)
    draw = ImageDraw.Draw(img)

    # Calculate the lower-right corner based on the width and height
    lower_right_corner = (upper_left_corner[0] + width, upper_left_corner[1] + height)

    # Draw the rectangle
    draw.rectangle((upper_left_corner, lower_right_corner), outline="red")

    # Save the image with rectangle
    img.save(r'F:\School\Ing\DIPLOMA\doku\bicycle_original.png')


def draw_yolo(img: str, normalized_coordinates: tuple[float, float, float, float]):
    # Open an image file
    img = Image.open(img)
    draw = ImageDraw.Draw(img)

    # Convert normalized coordinates to pixel coordinates
    x_center = normalized_coordinates[0] * img.width
    y_center = normalized_coordinates[1] * img.height
    width = normalized_coordinates[2] * img.width
    height = normalized_coordinates[3] * img.height

    # Calculate the upper-left and lower-right corners of the rectangle
    upper_left_corner = (x_center - width / 2, y_center - height / 2)
    lower_right_corner = (x_center + width / 2, y_center + height / 2)

    # Draw the rectangle
    draw.rectangle((upper_left_corner, lower_right_corner), outline="red")

    # Save the image with rectangle
    img.save(r'F:\School\Ing\DIPLOMA\doku\bicycle_rescaled.png')


if __name__ == "__main__":
    draw_coco(img=r'F:\School\Ing\DIPLOMA\night_detector\datasets\exdark\original\images\Bicycle\2015_00001.png',
              upper_left_corner=(204, 28),
              width=271,
              height=193)

    draw_yolo(img=r'F:\School\Ing\DIPLOMA\night_detector\datasets\exdark\yolo\exdark640\train\images\2015_00001.png',
              normalized_coordinates=(0.67890625, 0.33203125, 0.5421875, 0.5140625))
