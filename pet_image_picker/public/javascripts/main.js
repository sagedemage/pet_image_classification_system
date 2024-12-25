let index = 0
let length = 0
let imageFile = ""

window.addEventListener("keydown", previousEvent)
window.addEventListener("keydown", nextEvent)

async function getImages() {
    const url = "/api/images"
    try {
        const response = await fetch(url)
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json();
        const imageFiles = json.image_files;
        length = imageFiles.length - 1
        imageFile = "public/" + imageFiles[index]
        document.getElementById("pet_image").src = imageFiles[index]
        document.getElementById("image_pos").innerText = `${index} of ${length}`
    } catch (error) {
        console.error(error.message)
    }
}

window.onload = () => {
    getImages()
}

function previous() {
    if (index > 0) {
        index -= 1
        getImages()
    }
}

function next() {
    if (index < length) {
        index += 1
        getImages()
    }
}

async function saveImage() { // eslint-disable-line no-unused-vars
    const saveImage = confirm("Do you want to save the pet image?")
    if (saveImage) {
        const url = "/api/save_image"
        try {
            const response = await fetch(url, {
                method: "POST",
                body: JSON.stringify({ image_file: imageFile }),
                headers: {
                    "Content-Type": "application/json"
                }
            })
            if (!response.ok) {
                throw new Error(`Response status: ${response.status}`)
            }

            const json = await response.json();
            const msg = json.msg;
            alert(msg)
        } catch (error) {
            console.error(error.message)
        }
    }
}

function previousEvent(event) {
    if (event.keyCode === 37) {
        // Left arrow key
        previous()
    }
}

function nextEvent(event) {
    if (event.keyCode === 39) {
        // Right arrow key
        next()
    }
}