const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  pages : {
    index : {
      entry:'src/main.js',
      title : "곰파다"
    }
  },
  devServer : {
    proxy : {
      '/api' : {
        target : "http://localhost:8000",
        changeOrigin : true
      }
    }
  },
  lintOnSave : false,
  transpileDependencies: true,

})
