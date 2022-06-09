
<template>
  <el-row class="footer-conainter" :gutter="10">
    <el-col class="logo-link" :sm="24" :md="12" >
      <img class="footer-logoa" :src="require(`@/assets/title.png`)">
      <a href="https://github.com/boostcampaitech3/final-project-level3-nlp-03">
        <img class="icon" :src="require(`@/assets/github.png`)">
      </a>
      <a href="https://boostcamp.connect.or.kr/program_ai.html">
      <img class="icon" :src="require(`@/assets/homepage.png`)">
      </a>
    </el-col>
    <el-col class="name" :sm="24" :md="12">
      <div>Copyright 2022 곰파다 All rights reserved.</div>
      <div>네이버 부스트캠프 AI Tech 3 - NLP 03</div>
      <div>김상렬, 김소연, 김은기, 박세연, 임수정</div>
    </el-col>
</el-row>
</template>

<script>


export default {
  components : {

  },
  data(){
    return{

    }
  }

}
</script>

<style lang="scss">
.footer-container{

  background-color : #f4f6f8;
  border : 1px solid red;
}
.logo-link{
  display : flex;
  align-items: center;
}
.footer-logoa{
  width : 150px;
}
.name{
  text-align:right;
}
.icon{
  width : 30px;
  height : 30px;
  margin-right : 10px;
}
</style>