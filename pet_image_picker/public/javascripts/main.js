let index = 0
let length = 0
let image_file = ""

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
        image_file = "public/" + image_files[index]
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

async function save_image() {
    console.log(image_file)

    const url = "/api/save_image"
    try {
        const response = await fetch(url, {
            method: "POST",
            body: JSON.stringify({image_file: image_file}),
            headers: {
                "Content-Type": "application/json"
            }
        })
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`)
        }

        const json = await response.json();
        let image_files = json.image_files;
        length = image_files.length-1
        image_file = "public/" + image_files[index]
        document.getElementById("pet_image").src = image_files[index]
        document.getElementById("image_pos").innerText = `${index} of ${length}`
    } catch (error) {
        console.error(error.message)
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