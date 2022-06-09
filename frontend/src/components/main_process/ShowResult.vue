<template>
<div v-show="loading">
    <h2 class="result" ref="result">결과 확인</h2>
    <el-button size="small" class="result-button" type="primary" @click="tsvExport()">결과 추출 (.TSV)</el-button>
    <table-view></table-view>
    <!-- <h2>결과 통계</h2>
    <graph-view></graph-view> -->
</div>
</template>

<script>
import TableView from '@/components/main_process/TableView.vue'
import GraphView from '@/components/main_process/GraphView.vue'

export default {
  components : {
    TableView,
    GraphView,

  },
  watch : {
    loading(val, old){
     // console.log(val, old)
     //this.$refs.result.$el.scrollIntoView({ behavior: 'smooth' });
    }
  },
  computed : {
    loading(){
      return !this.$store.getters.getLoadingStatus
    },
    testName(){
      return this.$store.getters.getTestname
    }
  },
  data(){
    return{
      studentResult : [],
    }
  },
  methods : {
    checkSimText(sim_score){
      let message = ""
      if(sim_score >= 0.75){
        message = "유사"
      }else if(sim_score >= 0.35){
        message = "모호"
      }else{  
        message = "유의"
      }
      return message
    },
    preprocessExport(){
      let data = this.$store.getters.getResult
      data = data.problem

      this.studentResult = []
        data.forEach((v, i) => {
          //문제별
          //학생
          v.result.forEach((value, index) => {
            let student_id = value.student_id
            let property = {
              student_id : student_id,
              question : v.question,
              gold_answer : v.gold_answer,
              answer : value.answer,
              keyword : v.keywords.join("/"),
              user_keyword : value.match_info.keyword.join("/"),
              similarity_keyword : value.match_info.similarity_keyword.join("/"),
              sim_score : value.sim_score,
              sim_message : this.checkSimText(value.sim_score),
              keyword_score : value.keyword_score,
              final_score :  Math.round(
                (value.keyword_score * 0.5 + value.sim_score * 0.5) * 100
                )

            }
            console.log(property)
            this.studentResult.push(property)
          
        })

    })

    },
    tsvExport() {

      let name = this.testName ? `${this.textName}.tsv`: "export.tsv"
      this.preprocessExport()
      let arrData = this.studentResult
      let csvContent = "data:text/tsv;charset=utf-8,";
      csvContent += [
        Object.keys(arrData[0]).join("\t"),
        ...arrData.map(item => Object.values(item).join("\t"))
      ]
        .join("\n")
        .replace(/(^\[)|(\]$)/gm, "");

      const data = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", data);
      link.setAttribute("download", name);
      link.click();
    }
  },
  mounted(){

  }
}
</script>


<style lang="scss" scoped>
.result-button{
  margin : 0 0 0 10px;
}
</style>