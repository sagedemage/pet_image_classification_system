"""Movie Recommendation Program"""

import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from ml.model import PetClassifier
import numpy as np
from torchvision import datasets, transforms
import pandas as pd

from config import BATCH_SIZE

annotations_text = "dataset/oxford-iiit-pet/annotations/trainval.txt"
OXFORD_III_PET_LABELS_CSV = "labels/oxford-iiit-pet_labels.csv"


def load_training(root_path: str, img_size: tuple[int, int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
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

    img_size = (500, 300)

    picked_data_loader = load_training("picked_images/", img_size)

    picked_data_inputs, _ = next(iter(picked_data_loader))

    # Load a saved version of the model
    saved_model = PetClassifier().to(device)
    saved_model.load_state_dict(torch.load(model_path, weights_only=True))

    # Perform a transform on the data for it to be usable for the model
    picked_data_inputs = picked_data_inputs.to(device)
    picked_data_inputs = picked_data_inputs.type(torch.float32)

    logits = saved_model(picked_data_inputs)

    # Apply the rectified linear unit function (ReLU)
    # to the model output to ensure the output is a
    # tensor that always contains positive numbers.
    #
    # Output of the model for each number in the
    # tensor is from [0, infinity).
    #
    # This is required to avoid an IndexError when
    # using the get_item_by_movie_id method of the
    # MovieDataset class.
    pred_probab = nn.ReLU()(logits)
    rand_nums = np.random.rand(BATCH_SIZE)
    batch_pred = rand_nums.argmax()
    pred_input = round(float(pred_probab[batch_pred][0]) * 100, 0)
    index = int(pred_input)

    df_pet_labels_data = pd.read_csv(OXFORD_III_PET_LABELS_CSV)
    rows = df_pet_labels_data.loc[df_pet_labels_data["ID"] == index]
    row = rows.iloc[0]
    label = row["Label"]

    print(f"The breed of the pet is: {label}")


if __name__ == "__main__":
    main()
