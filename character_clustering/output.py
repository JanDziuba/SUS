from airium import Airium
import os


def create_html(clusters):
    a = Airium()

    a('<!DOCTYPE html>')
    with a.html(lang="en"):
        with a.head():
            a.meta(charset="utf-8")
            a.title(_t="Character clusters")

        with a.body():
            with a.div(len(clusters)):
                for cluster in clusters.values():
                    for image_path in cluster.keys():
                        a.img(src=image_path)
                    a.hr()

    with open("output.html", 'w') as file:
        file.write(str(a))


def save_clusters_to_file(clusters):
    with open("output.txt", "w") as file:
        for cluster in clusters.values():
            for image_path in cluster.keys():
                file.write(f"{os.path.basename(image_path)} ")
            file.write("\n")
