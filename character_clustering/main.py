from output import *
import argparse
from sklearn.cluster import DBSCAN
import cv2


def parse_input_file(input_file_path):
    with open(input_file_path) as file:
        return file.read().splitlines()


def resize_image(image, max_height, max_width):
    image = cv2.copyMakeBorder(
        image,
        (max_height - image.shape[0]) // 2,
        (max_height - image.shape[0] + 1) // 2,
        (max_width - image.shape[1]) // 2,
        (max_width - image.shape[1] + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=255
    )
    return image.flatten()


def get_images(image_paths):
    max_height = 0
    max_width = 0
    images = {}

    for image_path in image_paths:
        image = cv2.imread(image_path, 0)
        max_height = max(max_height, image.shape[0])
        max_width = max(max_width, image.shape[1])
        images[image_path] = image

    for image_path in images.keys():
        images[image_path] = resize_image(images[image_path], max_height, max_width)

    return images


def cluster_images(image_paths):
    images = get_images(image_paths)

    model = DBSCAN(
        metric="euclidean",
        eps=560,
        n_jobs=-1,
        min_samples=1
    )
    model.fit(list(images.values()))
    clusters_list = model.labels_

    clusters_dict = {}
    for index, (path, image) in zip(clusters_list, images.items()):
        clusters_dict.setdefault(index, {})
        clusters_dict[index][path] = image

    return clusters_dict
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path")
    args = parser.parse_args()

    image_paths = parse_input_file(args.input_file_path)
    clusters = cluster_images(image_paths)
    save_clusters_to_file(clusters)
    create_html(clusters)


if __name__ == '__main__':
    main()

