const createError = require('http-errors')
const express = require('express')
const path = require('path')
const cookieParser = require('cookie-parser')
const logger = require('morgan')
const fs = require('node:fs')

const indexRouter = require('./routes/index')
const apiRouter = require('./routes/api')

const app = express()

// view engine setup
app.set('views', path.join(__dirname, 'views'))
app.set('view engine', 'jade')

app.use(logger('dev'))
app.use(express.json())
app.use(express.urlencoded({ extended: false }))
app.use(cookieParser())
app.use(express.static(path.join(__dirname, 'public')))

app.use('/', indexRouter)
app.use('/api', apiRouter)

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404))
})

// error handler
app.use(function (err, req, res) {
  // set locals, only providing error in development
  res.locals.message = err.message
  res.locals.error = req.app.get('env') === 'development' ? err : {}

  // render the error page
  res.status(err.status || 500)
  res.render('error')
})

module.exports = app

const dest = 'public/images/'

try {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest)
  }
} catch (err) {
  console.error(err)
}

try {
  let imageFiles = fs.readdirSync(dest)

  if (imageFiles.length === 0) {
    const datasetImageDir = '../dataset/oxford-iiit-pet/images'
    fs.cpSync(datasetImageDir, dest, { recursive: true })

    imageFiles = fs.readdirSync(dest)

    imageFiles.forEach((file) => {
      const fileExt = file.slice(file.length - 4, file.length)
      if (fileExt !== '.jpg') {
        fs.rmSync(dest + file)
      }
    })
  }
} catch (err) {
  console.error(err)
}

console.log('Server at http://localhost:3000')
