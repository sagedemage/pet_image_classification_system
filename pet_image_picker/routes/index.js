const express = require('express');
let router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Pet Image Picker' });
});

router.get("/about", function(req, res, next) {
  res.render('about', { title: 'About' });
})

module.exports = router;
