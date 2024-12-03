let index = 0
let length = 0

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
        let image_files = json.image_files;
        length = image_files.length-1
        document.getElementById("pet_image").src = image_files[index]
        document.getElementById("image_pos").innerText = `${index} of ${length}` 
    } catch (error) {
        console.error(error.message)
    }
}

getImages()

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

function previousEvent(event) {
    if (event.keyCode == 37) {
        // Left arrow key
        previous()
    }
}

function nextEvent(event) {
    if (event.keyCode == 39) {
        // Right arrow key
        next()
    }
}