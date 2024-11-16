"""Pet Image Picker GUI Program"""

from tkinter import Tk, Label, Button
from PIL import ImageTk, Image
import glob
import shutil
from pathlib import Path

WIDTH = 1020
HEIGHT = 645

counter = 0


def main():
    window = Tk()
    window.title("Pet Image Picker")
    window.geometry(f"{WIDTH}x{HEIGHT}")

    images = []
    picked_image_paths = []

    image_paths = glob.glob(
        "dataset/oxford-iiit-pet/images/*.jpg", recursive=False
    )

    for image_path in image_paths:
        image = ImageTk.PhotoImage(Image.open(image_path).resize((600, 400)))
        images.append(image)

    num_pets_pick = 1

    def next_image():
        global counter
        if counter < len(image_paths) - 1:
            counter += 1
        else:
            counter = 0

        image_label.config(image=images[counter])
        info_label.config(
            text="Image " + str(counter + 1) + " of " + str(len(image_paths))
        )

    def previous_image():
        global counter
        if counter > 0:
            counter -= 1
        else:
            counter = 0

        image_label.config(image=images[counter])
        info_label.config(
            text="Image " + str(counter + 1) + " of " + str(len(image_paths))
        )

    image_label = Label(window, image=images[0])
    info_label = Label(
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

    def pick_image():
        picked_image_path = image_paths[counter]
        if picked_image_path not in picked_image_paths:
            if len(picked_image_paths) < num_pets_pick:
                picked_image_paths.append(picked_image_path)
                print("Picked pet image.")
            else:
                print(f"Picked {num_pets_pick} pet image!")
        else:
            print("This pet image has already been picked!")

    def remove_image():
        if len(picked_image_paths) > 0:
            picked_image_paths.pop()
            print("Removed pet image.")
        else:
            print("There are no picked image!")

    def save_images():
        if len(picked_image_paths) == num_pets_pick:
            dest_path_s = "picked_images/pets"
            file = open(
                "picked_images/pets/picked_pet_image.txt",
                "w",
                encoding="utf-8",
            )
            image_path_s = picked_image_paths[0]
            copied_file_path_s = shutil.copy(image_path_s, dest_path_s)
            copied_file_path = Path(copied_file_path_s)
            new_name = "pet.jpg"
            new_path = f"{dest_path_s}/{new_name}"
            copied_file_path.rename(new_path)
            file.write(image_path_s + "\n")
            file.close()
            print("Saved pet image.")
        else:
            print(f"Pick {num_pets_pick} pet image!")

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

    def previous_image_binding(event):
        previous_image()

    def next_image_binding(event):
        next_image()

    window.bind("<Left>", previous_image_binding)
    window.bind("<Right>", next_image_binding)

    # display components
    image_label.pack()
    info_label.pack()
    previous_button.pack(side="left", pady=3)
    next_button.pack(side="right", pady=3)
    remove_button.pack(side="bottom", pady=3)
    pick_button.pack(side="bottom", pady=3)
    save_button.pack(side="bottom", pady=3)
    window.mainloop()


if __name__ == "__main__":
    main()
