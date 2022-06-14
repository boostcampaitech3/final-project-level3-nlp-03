<template>
<div class="add-problem">
  <div class="add-problem-header">
    <h3>문제와 모범 답안</h3>
    <el-input class="input-form" placeholder="문제를 입력해주세요" v-model="input1">
      <template slot="prepend">문제 </template>
    </el-input>
    
    <el-input class="input-form" placeholder="모범 답안을 입력해주세요" v-model="input2">
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
      <el-popover
    placement="top-start"
    title="파일은 csv형식으로 올려야 하며 형식은 다음과 같습니다.(예시)"
    width="200"
    trigger="hover"
    :content="basic_info">
          <template v-slot>
        <code>
          student_id,answer </br>
          0,나무보다 금속이 열 전달을 더 잘하기 때문에</br>
          3,금속이 나무보다 열을 더 잘 흡수를 잘하기 때문이다.</br>
        </code>
      </template>
    <i slot="reference" class="el-icon-info"></i>
    
  </el-popover>
    <!-- <el-tooltip :content="basic_info" placement="top" effect="light">
      <template v-slot:content>
        <div>파일은 csv형식으로 올려야 하며 형식은 다음과 같습니다.</div>
        <code>
          student_id,answer
0,나무보다 금속이 열 전달을 더 잘하기 때문에
3,금속이 나무보다 열을 더 잘 흡수를 잘하기 때문이다.
12,금속이 나무보다 열전도성이 좋기 때문이다.
        </code>
      </template>
        <i class="el-icon-info"></i>
    </el-tooltip> -->
    <el-upload
      class="upload-demo"
      drag
      action="https://jsonplaceholder.typicode.com/posts/"
      :file-list="fileList"
      :on-success="readCsv"
      >
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
      <div class="el-upload__tip" slot="tip">only csv files allowed</div>
  </el-upload>
  <el-button type="primary" class="save-problem" @click="addProblem(tab_name)" :disabled="button">{{tab_name}} 저장</el-button>
  </div>
</div>
</template>

<script>
/*
columnA	columnB	columnC
Susan	41	a
Mike	5	b
Jake	33	c
Jill	30	d

*/
import Papa from 'papaparse';

export default {
    components : {

    },
    props : ["tab_name"],
    data(){
      return{
        csv : null,
        basic_info : "qwer",
        fileList : [],
        file : null,
        input1 : "",
        input2 : "",
        dynamicTags: [],
        inputVisible: false,
        inputValue: '',
        content : [],
        parsed : false,
        button : false,
      }
    },
    methods : {
      readCsv(response, file, fileList){
        this.file = file.raw

        this.parseFile()
      },
//       handleFileUpload( event ){
//     this.file = event.target.files[0];
//     console.log("에전꺼")
//     console.log(event.target.files[0])
//     this.parseFile();
// },
      parseFile(){
        Papa.parse( this.file, {
            header: true,
            skipEmptyLines: true,
            complete: function( results ){
                this.content = results;
                this.parsed = true;
            }.bind(this)
        });

        
      },
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
      },
      answerPreprocess(answer){
        let answers = []
        answer.forEach((v, i)=>{
          answers.push([v.student_id, v.answer])
        })
        return answers
      },
      addProblem(tab_name){


        let answers = this.answerPreprocess(this.content.data)    
        this.$store.commit('addProblem', {
          question : this.input1,
          gold_answer : this.input2,
          keywords : this.dynamicTags,
          answers : answers
        })

        //console.log("문제추가")
        //console.log(this.$store.getters.getProblem)

        this.button = true

        this.$notify({
          title: '문제 저장 성공',
          message: `${tab_name} 저장되었습니다.`,
          type: 'success'
        });


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
.save-problem{
  float : right;
}
h3{
  display:inline-block;
  margin-right : 5px;
}
code {
    padding: 0.25rem;
    background-color: #F1F1F1;
    border-radius: 5px;
    font-family: "Consolas", "Sans Mono", "Courier", "monospace";
    font-size: 0.75rem;
}
</style>