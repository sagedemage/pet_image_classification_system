const express = require('express');
const fs = require("fs");
let router = express.Router();

/* GET users listing. */
router.get('/hello', function(req, res, next) {
  res.send('Hello world!');
});

router.get("/images", function(req, res, next) {
  let image_dir = "public/images"
  let files = fs.readdirSync(image_dir);
  let image_files = []
  let row = []
  console.log(files.length)

  for (let i = 0; i < files.length; i++) {
    let file = files[i]
    let image_file = "images/" + file;
    image_files.push(image_file);
  }

  let json_content = JSON.stringify({image_files: image_files})
  res.setHeader("Content-Type", "application/json");
  res.send(json_content)
})

router.post("/save_image", function(req, res, next) {
  const dest_dir = "../picked_image/pet/"
  const image_dest = dest_dir + "pet.jpg"
  res.setHeader("Content-Type", "application/json");
  let image_file = req.body.image_file
  fs.cpSync(image_file, image_dest, { recursive: true })
  let image_items = image_file.split("/")
  let image_file_name = image_items[2]
  const data = "Image: " + image_file_name
  fs.writeFileSync(dest_dir + "picked_pet_image.txt", data, {flag: "w"})
  res.send(JSON.stringify({msg: "Saved image"}))
})

module.exports = router;
