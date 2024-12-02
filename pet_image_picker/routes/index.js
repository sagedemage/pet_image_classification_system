const express = require('express');
const fs = require("fs");
let router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  let image_dir = "public/images"
  let files = fs.readdirSync(image_dir);
  let image_files = []
  let row = []
  console.log(files.length)

  for (let i = 0; i < files.length; i++) {
    let file = files[i]
    let image_file = "images/" + file;
    row.push(image_file);
    if ((i+1) % 10 == 0) {
      image_files.push(row)
      row = []
    }
  }
  res.render('index', { title: 'Pet Image Picker', image_files: image_files });
});

router.get("/about", function(req, res, next) {
  res.render('about', { title: 'About' });
})

router.get("/view-image", function(req, res, next) {
  res.render('view_image', { title: 'View an Image' });
})

module.exports = router;
