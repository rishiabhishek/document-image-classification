from ImageDataset import ImageDataSet
from ImageModel import ImageModel


def main():
    imageDataset = ImageDataSet(
        "/Volumes/My Passport/abhishek/Datasets/Image Dataset/rvl-cdip/dataset")

    model = ImageModel(8, 224, 224)
    model.train_model(imageDataset, batch_size=50)


if __name__ == "__main__":
    main()
