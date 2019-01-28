import cv2
import os

import paths


def resize_all_images_in_directory(src_dir, dst_dir, dst_size):
    image_extensions = [
        "png", "jpg", "jpeg"
    ]
    images_names = [
        name
        for name in os.listdir(src_dir)
        if name.split(".")[-1] in image_extensions
    ]

    os.makedirs(dst_dir, exist_ok=True)

    for name in images_names:
        img = cv2.imread(os.path.join(src_dir, name))
        img = cv2.resize(img, dsize=dst_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dst_dir, name), img)


def main():
    src_dir = os.path.join(paths.data_root, "KostetCentral")
    dst_dir = os.path.join(paths.data_root, "KostetCentralResized")
    # resize_all_images_in_directory(src_dir, dst_dir, (96 * 2, 128 * 2))
    resize_all_images_in_directory(src_dir, dst_dir, (96 * 3, 128 * 3))


if __name__ == '__main__':
    main()
