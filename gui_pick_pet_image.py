"""Pet Image Picker GUI Program"""

from tkinter import Tk, Label, Button
from PIL import ImageTk, Image
import glob
import shutil
from pathlib import Path

WIDTH = 1020
HEIGHT = 645

counter: int = 0
picked_image_path: str = ""


def main():
    window = Tk()
    window.title("Pet Image Picker")
    window.geometry(f"{WIDTH}x{HEIGHT}")

    images = []

    image_paths = glob.glob(
        "dataset/oxford-iiit-pet/images/*.jpg", recursive=False
    )

    for image_path in image_paths:
        image = ImageTk.PhotoImage(Image.open(image_path).resize((600, 400)))
        images.append(image)

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
        global picked_image_path
        selected_image_path = image_paths[counter]
        if selected_image_path != picked_image_path:
            picked_image_path = selected_image_path
            print("Picked pet image.")
        else:
            print("This pet image has already been picked!")

    def save_images():
        if picked_image_path != "":
            dest_path_s = "picked_images/pets"
            file = open(
                "picked_images/pets/picked_pet_image.txt",
                "w",
                encoding="utf-8",
            )
            copied_file_path_s = shutil.copy(picked_image_path, dest_path_s)
            copied_file_path = Path(copied_file_path_s)
            new_name = "pet.jpg"
            new_path = f"{dest_path_s}/{new_name}"
            copied_file_path.rename(new_path)
            file.write(picked_image_path + "\n")
            file.close()
            print("Saved pet image.")
        else:
            print("Pick a pet image!")

    pick_button = Button(
        window,
        text="Pick this image",
        width=20,
        height=2,
        bg="blue",
        fg="white",
        command=pick_image,
    )

    save_button = Button(
        window,
        text="Save the image",
        width=20,
        height=2,
        bg="green",
        fg="white",
        command=save_images,
    )

    def previous_image_binding(event):  # pylint: disable=unused-argument
        previous_image()

    def next_image_binding(event):  # pylint: disable=unused-argument
        next_image()

    def save_image_binding(event):  # pylint: disable=unused-argument
        save_images()

    def pick_image_binding(event):  # pylint: disable=unused-argument
        pick_image()

    # Set the key bindings for the window
    window.bind("<Left>", previous_image_binding)
    window.bind("<Right>", next_image_binding)
    window.bind("<Control-s>", save_image_binding)
    window.bind("<Return>", pick_image_binding)

    # display the components
    image_label.pack()
    info_label.pack()
    previous_button.pack(side="left", pady=3)
    next_button.pack(side="right", pady=3)
    pick_button.pack(side="bottom", pady=3)
    save_button.pack(side="bottom", pady=3)

    # this runs the application window, and it responds to events
    # and process these events
    window.mainloop()


if __name__ == "__main__":
    main()
