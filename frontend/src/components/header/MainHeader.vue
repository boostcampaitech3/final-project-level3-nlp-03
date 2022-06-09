<template>
<div>
  <div class="logo-container">
    <img class="main-logo" :src="require(`@/assets/logo.png`)">
  </div>
  <div class="info-container">
    <el-button type="" @click="goGuide()"> 사용자 가이드</el-button>
    <a href="/files/demo_student_answer.csv" download>
      <el-button type="warning" @click="doDemo()"> 데모 결과 보기</el-button>
    </a>
  </div>
</div>
</template>

<script>
import Api from '@/api/api'

export default {
  methods : {

    goGuide(){
      let url = "https://maylilyo.notion.site/6e830bd4d4b1490692312a57b18942a5"
      window.open(url)
    },
    doDemo(){

      this.$store.commit("loadingPending")
      let data = {
        "problem" : [
          {
            "question" : "뜨거운 물이 들어있는 냄비에 나무국자와 금속 국자를 넣었는데, 나무국자는 뜨거워지지않고 금속국자는 뜨거워지는 이유는 무엇일까요?",
            "gold_answer" : "나무 재질보다 금속 재질이 열 전달이 더 잘 되기 때문이다",
            "keywords" : [
              "열",
              "전달",
              "나무",
              "금속"
            ],
            "answers" : [
              [
                "0",
                "나무보다 금속이 열 전달을 더 잘하기 때문에"
              ],
              [
                "18",
                "나무 국자보다 금속 국자가 열 전도성이 좋기 때문이다."
              ],
              [
                "1058",
                "열 전도성이 달라서 금속 국자의 열 전도성이 더 높아짐"
              ],
              [
                "3715",
                "열이 이동해서"
              ],
              [
                "3729",
                "나무는 전도가 된다."
              ],
            ]
          }
        ]
      }
      let problem = "뜨거운 물이 들어있는 냄비에 나무국자와 금속 국자를 넣었는데, 나무국자는 뜨거워지지않고 금속국자는 뜨거워지는 이유는 무엇일까요?"
      let gold_answer = "나무 재질보다 금속 재질이 열 전달이 더 잘 되기 때문이다"
      let keywords = [
              "열",
              "전달",
              "나무",
              "금속"
            ]

      this.$notify({
        title : "데모 실행",
        dangerouslyUseHTMLString: true,
        duration : 10000,
        type : "sucess",
        message: `
        <span>문제 : ${problem}</span> 
        <hr>
        <span>모범답안 : ${gold_answer}</span>
        <hr>
        <span>키워드 : ${keywords.toString()}</span>         
        `
      })


        Api.sendData(data)
        .then((res)=>{
          let result = res.data.problem
          //console.log("결과")
          //console.log(result)
          this.$store.commit("initializeResult")
          result.forEach((v, i)=>{
            
            this.$store.commit('addResult', v)
            this.$store.commit("loadingDone")
          })
          //console.log(this.$store.getters.getResult)
          this.loading = false
          this.$notify({
            title: '성공',
            message: `성공적으로 채점이 되었습니다.`,
            type: 'success'
          });
        })
        .catch(error =>{ 
        this.$notify({
          title: '실폐',
          message: `서버 전송에 실폐하였습니다.`,
          type: 'failure'
        });
      });


    },
  }
}
</script>

<style lang="scss">
.logo-container{
  display : flex;
  width : 100%;
  justify-content: center;
  align-items: center;
  margin : 1rem 0;
  .main-logo{
    margin : 0 auto;
    width : 150px;
    height : 150px;
  }
}
.info-container{
  display : flex;
  justify-content: right;
  margin-bottom : 15px;
  .el-button{
    margin : 0 10px 0 0;
  }
  
}

</style>