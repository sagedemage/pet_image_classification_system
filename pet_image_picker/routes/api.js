const express = require('express')
const fs = require('node:fs')
const router = express.Router()

router.get('/images', function (req, res) {
  const imageDir = 'public/images'
  fs.readdir(imageDir, (err, files) => {
    if (err) {
      console.error(err)
    } else {
      const imageFiles = []

      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const imageFile = 'images/' + file
        imageFiles.push(imageFile)
      }

      const jsonContent = JSON.stringify({ image_files: imageFiles })
      res.setHeader('Content-Type', 'application/json')
      res.send(jsonContent)
    }
  })
})

router.post('/save_image', function (req, res) {
  const destDir = '../picked_image/pet/'
  const imageDest = destDir + 'pet.jpg'
  res.setHeader('Content-Type', 'application/json')
  const imageFile = req.body.image_file
  fs.cp(imageFile, imageDest, { recursive: true }, (err) => {
    if (err) {
      console.error(err)
    }
  })
  const imageItems = imageFile.split('/')
  const imageFileName = imageItems[2]
  const data = 'Image: ' + imageFileName
  fs.writeFile(destDir + 'picked_pet_image.txt', data, { flag: 'w' }, (err) => {
    if (err) {
      console.error(err)
    }
  })
  res.send(JSON.stringify({ msg: 'Saved the pet image' }))
})

module.exports = router
