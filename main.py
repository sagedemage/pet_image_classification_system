"""Movie Recommendation Program"""

import sys
import torch
from torch.utils.data import DataLoader
from ml.model import PetClassifier
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from config import BATCH_SIZE, img_size

annotations_text = "dataset/oxford-iiit-pet/annotations/trainval.txt"
OXFORD_III_PET_LABELS_CSV = "labels/oxford-iiit-pet_labels.csv"


def load_image(root_path: str, image_size: tuple[int, int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data = datasets.ImageFolder(root=root_path, transform=transform)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


def main():
    if len(sys.argv) < 2:
        print("Missing the model file path!")
        exit()
    args = sys.argv
    # Use trained model
    model_path = args[1]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    picked_data_loader = load_image("picked_image/", img_size)

    picked_data_inputs, _ = next(iter(picked_data_loader))

    # Load a saved version of the model
    saved_model = PetClassifier().to(device)
    saved_model.load_state_dict(torch.load(model_path, weights_only=True))

    # Perform a transform on the data for it to be usable for the model
    picked_data_inputs = picked_data_inputs.to(device)
    picked_data_inputs = picked_data_inputs.type(torch.float32)

    logits = saved_model(picked_data_inputs)

    # Convert the model output to a NumPy ndarray in order to get
    # a list of 1 predicted label
    pred_probab = torch.max(logits.data, 1)[1].numpy(force=True)
    index = int(pred_probab[0])

    df_pet_labels_data = pd.read_csv(OXFORD_III_PET_LABELS_CSV)
    rows = df_pet_labels_data.loc[df_pet_labels_data["ID"] == index]
    row = rows.iloc[0]
    label = row["Label"]
    cat_or_dog = row["Cat_Dog"]
    images = []
    image_1 = row["Image_1"]
    image_2 = row["Image_2"]
    images.append(image_1)
    images.append(image_2)

    image_1_elements = image_1.split("/")
    image_2_elements = image_2.split("/")

    image_file_names = []
    image_1_file_name = image_1_elements[3].rstrip(".jpg")
    image_2_file_name = image_2_elements[3].rstrip(".jpg")
    image_file_names.append(image_1_file_name)
    image_file_names.append(image_2_file_name)

    if cat_or_dog == 0:
        # Cat
        print("The pet is a cat")
        print(f"The breed of the cat is: {label}")
    elif cat_or_dog == 1:
        # Dog
        print("The pet is a dog")
        print(f"The breed of the dog is: {label}")

    fig = plt.figure(
        num="Images of the Predicted Breed of the Pet", figsize=(10, 5)
    )
    columns = 2
    rows = 1

    for i in range(columns * rows):
        img = mpimg.imread(images[i])
        axes = fig.add_subplot(rows, columns, i + 1)
        axes.set_title(f"{image_file_names[i]}")
        plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    main()
