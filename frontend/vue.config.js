const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.(csv|xlsx|xls)$/,
          loader: 'file-loader',
          options: {
            name: `files/[name].[ext]`
          }
        }
      ],
     },
  },
  pages : {
    index : {
      entry:'src/main.js',
      title : "곰파다"
    }
  },
  devServer : {
    proxy : {
      '/api' : {
        target : "http://27.96.130.147:30001",
        changeOrigin : true
      }
    }
  },
  lintOnSave : false,
  transpileDependencies: true,

})
