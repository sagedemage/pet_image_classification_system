"""Pet Image Picker GUI Program"""

from tkinter import Tk, Label, Button
from PIL import ImageTk, Image
import glob

root = Tk()
root.title("Image Picker")
root.geometry("680x430")

images = []

image_paths = glob.glob("dataset/oxford-iiit-pet/images/*.jpg", recursive=False)

for image_path in image_paths:
    image = ImageTk.PhotoImage(Image.open(image_path).resize((600, 350)))
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


imageLabel = Label(root, image=images[0])
infoLabel = Label(root, text=f"Image 1 of {len(images)}", font="Helvetica, 20")
next_button = Button(
    root,
    text="Next",
    width=20,
    height=2,
    bg="purple",
    fg="white",
    command=next_image,
)
previous_button = Button(
    root,
    text="Previous",
    width=20,
    height=2,
    bg="purple",
    fg="white",
    command=previous_image,
)

# display components
imageLabel.pack()
infoLabel.pack()
previous_button.pack(side="left", pady=3)
next_button.pack(side="right", pady=3)
root.mainloop()
