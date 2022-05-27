const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  pages : {
    index : {
      entry:'src/main.js',
      title : "곰파다"
    }
  },
  lintOnSave : false,
  transpileDependencies: true
})
