"""Pet Image Picker GUI Program"""

from tkinter import Tk, Label, Button, IntVar
from PIL import ImageTk, Image
import glob
import shutil
from pathlib import Path
from config import img_size

WIDTH = 1020
HEIGHT = 645

window = Tk()
window.title("Pet Image Picker")
window.geometry(f"{WIDTH}x{HEIGHT}")

images = []
picked_image_paths = []

image_paths = glob.glob("dataset/oxford-iiit-pet/images/*.jpg", recursive=False)

for image_path in image_paths:
    image = ImageTk.PhotoImage(Image.open(image_path).resize((600, 400)))
    images.append(image)

counter = 0


def next_image():
    global counter
    if counter < len(image_paths) - 1:
        counter += 1
    else:
        counter = 0

    imageLabel.config(image=images[counter])
    infoLabel.config(
        text="Image " + str(counter + 1) + " of " + str(len(image_paths))
    )


def previous_image():
    global counter
    if counter > 0:
        counter -= 1
    else:
        counter = 0

    imageLabel.config(image=images[counter])
    infoLabel.config(
        text="Image " + str(counter + 1) + " of " + str(len(image_paths))
    )


imageLabel = Label(window, image=images[0])
infoLabel = Label(
    window, text=f"Image 1 of {len(images)}", font="Helvetica, 20"
)
next_button = Button(
    window,
    text="Next",
    width=20,
    height=2,
    bg="gray",
    fg="white",
    command=next_image,
)
previous_button = Button(
    window,
    text="Previous",
    width=20,
    height=2,
    bg="gray",
    fg="white",
    command=previous_image,
)

var1 = IntVar()


def pick_image():
    picked_image_path = image_paths[counter]
    if picked_image_path not in picked_image_paths:
        if len(picked_image_paths) < 4:
            picked_image_paths.append(picked_image_path)
            print("Picked pet image.")
        else:
            print("Picked 4 pet images!")
    else:
        print("This pet image has already been picked!")


def remove_image():
    if len(picked_image_paths) > 0:
        picked_image_paths.pop()
        print("Removed pet image.")
    else:
        print("There are no picked images!")


def save_images():
    if len(picked_image_paths) == 4:
        dest_path_s = "picked_images/pets"
        file = open(
            "picked_images/pets/picked_pet_images.txt", "w", encoding="utf-8"
        )
        for i in range(len(picked_image_paths)):
            image_path_s = picked_image_paths[i]
            copied_file_path_s = shutil.copy(image_path_s, dest_path_s)
            copied_file_path = Path(copied_file_path_s)
            new_name = f"pet{i + 1}.jpg"
            new_path = f"{dest_path_s}/{new_name}"
            copied_file_path.rename(new_path)
            file.write(image_path_s + "\n")
        print("Saved pet images.")
        file.close()
    else:
        print("Pick 4 pet images!")


pick_button = Button(
    window,
    text="Pick this image",
    width=20,
    height=2,
    bg="blue",
    fg="white",
    command=pick_image,
)

remove_button = Button(
    window,
    text="Remove this image",
    width=20,
    height=2,
    bg="red",
    fg="white",
    command=remove_image,
)

save_button = Button(
    window,
    text="Save the images",
    width=20,
    height=2,
    bg="green",
    fg="white",
    command=save_images,
)

# display components
imageLabel.pack()
infoLabel.pack()
previous_button.pack(side="left", pady=3)
next_button.pack(side="right", pady=3)
remove_button.pack(side="bottom", pady=3)
pick_button.pack(side="bottom", pady=3)
save_button.pack(side="bottom", pady=3)
window.mainloop()
