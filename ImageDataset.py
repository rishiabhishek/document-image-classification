import cv2
import os
import numpy as np
# import tensorflow as tf


class ImageDataSet(object):

    # 0 letter
    # 1 form
    # 2 email
    # 6 scientific publication
    # 9 news article
    # 11 invoice
    # 14 resume
    # 15 memo

    def __init__(self, base_path):

        self.base_path = base_path

        self.label2idx = {0: 0, 1: 1, 2: 2, 6: 3, 9: 4, 11: 5, 14: 6, 15: 7}
        self.idx2label = {value: key for key, value in self.label2idx.items()}
        self.label_onehot = np.identity(len(self.label2idx))

        self.image_size = (1000, 745, 3)

    def read_image(self, image_path):
        im = cv2.imread(image_path)
        return im

    def read_label_files(self, label_file_path):
        image_dict = {}
        lines = open(label_file_path, 'r')
        for line in lines:
            path, label = line.split()
            image_path = os.path.join(self.base_path, "images", path)
            image_dict[image_path] = label
        return image_dict

    def build_dataset(self, batch_size, *argv):

        if ("test" in argv):
            label_path = os.path.join(self.base_path, "labels/test.txt")
        elif ("val" in argv):
            label_path = os.path.join(self.base_path, "labels/val.txt")
        else:
            label_path = os.path.join(self.base_path, "labels/train.txt")
        image_dict = self.read_label_files(label_path)
        images = []
        labels = []
        i = 0

        for path, label in image_dict.items():
            if int(label) in self.label2idx:
                image = self.read_image(path)
                # print(path)
                # print(str(i))
                images.append(cv2.resize(image, (1000, 754)))
                index = self.label2idx[int(label)]
                labels.append(self.label_onehot[index])
                i += 1
                if i != 0 and (i % batch_size == 0):
                    yield np.array(images), np.array(labels)


def main():
    imageDataset = ImageDataSet(
        "/Volumes/My Passport/abhishek/Datasets/Image Dataset/rvl-cdip/dataset")
    batches = imageDataset.build_dataset(20, "train")

    for batch in batches:
        images = batch[0]
        labels = batch[1]
        print(images.shape)
        print(labels.shape)
        # for image in images:
        #     print(image.shape)


if __name__ == "__main__":
    main()
