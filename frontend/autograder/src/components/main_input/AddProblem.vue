<template>
<div class="add-problem">
  <div class="add-problem-header">
    <h3>문제와 모범 답안</h3>
    <el-input class="input-form" placeholder="Please input" v-model="input1">
      <template slot="prepend">문제 </template>
    </el-input>
    
    <el-input class="input-form" placeholder="Please input" v-model="input2">
      <template slot="prepend">모범답안</template>
    </el-input>
    <h3>필수 키워드</h3>
    <div class="keyword-tag-container">
      <el-tag
        :key="tag"
        v-for="tag in dynamicTags"
        closable
        :disable-transitions="false"
        @close="handleClose(tag)">
        {{tag}}
      </el-tag>
      <el-input
        class="input-new-tag"
        v-if="inputVisible"
        v-model="inputValue"
        ref="saveTagInput"
        size="mini"
        @keyup.enter.native="handleInputConfirm"
        @blur="handleInputConfirm"
      >
      </el-input>
      <el-button v-else class="button-new-tag" size="small" @click="showInput">+ New Tag</el-button>
    </div>
    <h3>파일 업로드</h3>
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
        dynamicTags: [],
        inputVisible: false,
        inputValue: ''
      }
    },
    methods : {
      handleClose(tag) {
        this.dynamicTags.splice(this.dynamicTags.indexOf(tag), 1);
      },
      showInput() {
        this.inputVisible = true;
        this.$nextTick(_ => {
          this.$refs.saveTagInput.$refs.input.focus();
        });
      },
      handleInputConfirm() {
        let inputValue = this.inputValue;
        if (inputValue) {
          this.dynamicTags.push(inputValue);
        }
        this.inputVisible = false;
        this.inputValue = '';
      }

    }
}
</script>

<style lang="scss" scoped>
.add-problem {
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

.keyword-tag-container {
  .el-tag + .el-tag {
    margin-left: 10px;
  }
  .button-new-tag {
    margin-left: 10px;
    height: 32px;
    line-height: 30px;
    padding-top: 0;
    padding-bottom: 0;
  }
  .input-new-tag {
    width: 90px;
    margin-left: 10px;
    vertical-align: bottom;
  }
}
</style>