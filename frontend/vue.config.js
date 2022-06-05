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
        target : "http://27.96.130.147:30002",
        changeOrigin : true
      }
    }
  },
  lintOnSave : false,
  transpileDependencies: true,

})
