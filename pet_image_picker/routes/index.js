var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Pet Image Picker' });
});

router.get("/about", function(req, res, next) {
  res.render('about', { title: 'About' });
})

router.get("/view-image", function(req, res, next) {
  res.render('view_image', { title: 'View an Image' });
})

module.exports = router;
