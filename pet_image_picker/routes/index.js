const express = require('express');
const router = express.Router();

/* GET home page. */
router.get('/', function(req, res) {
  res.render('index', { title: 'Pet Image Picker' });
});

router.get("/about", function(req, res) {
  res.render('about', { title: 'About' });
})

module.exports = router;
