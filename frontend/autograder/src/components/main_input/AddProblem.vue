<template>
<div class="add-problem">
  <div class="add-problem-header">
    <el-input class="input-form" placeholder="Please input" v-model="input1">
      <template slot="prepend">문제 </template>
    </el-input>
    
    <el-input class="input-form" placeholder="Please input" v-model="input2">
      <template slot="prepend">모범답안</template>
    </el-input>
    <el-upload
      class="upload-demo"
      drag
      action="https://jsonplaceholder.typicode.com/posts/"

      :file-list="fileList"
      multiple>
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
      <div class="el-upload__tip" slot="tip">only csv files allowed</div>
  </el-upload>
  </div>
</div>
</template>

<script>
export default {
    components : {

    },
    props : [],
    data(){
      return{
        fileList : [],
        input1 : "",
        input2 : "",
      }
    },
}
</script>

<style lang="scss" scoped>
.add-problem{
  .el-select .el-input {
      width: 110px;
    }
  .input-with-select .el-input-group__prepend {
    background-color: #fff;
  }
  .input-form{
    margin : 0 0 10px 0;
  }
  .upload-demo{
    text-align: center;
  }
}
</style>