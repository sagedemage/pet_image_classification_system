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

module.exports = router;
