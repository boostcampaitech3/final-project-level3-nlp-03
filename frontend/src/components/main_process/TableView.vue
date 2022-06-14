<template>
<div>
  <el-table
    :data="tableData"
    style="width: 100%">
    <el-table-column
      label="Index"
      type="index"
      width="100"
      prop="index">
    </el-table-column>
    <el-table-column
      sortable
      label="Student ID"
      prop="student_id">
    </el-table-column>
    <el-table-column
      sortable
      label="Score"
      prop="score">
    </el-table-column>
    <el-table-column
      align="right">
      <template slot-scope="scope">
        <el-button
          size="mini"
          @click="handleEdit(scope.$index, scope.row); dialogTableVisible = !dialogTableVisible">자세히 보기
        </el-button>
      </template>
    </el-table-column>
    
  </el-table>
  <el-dialog title="채점 결과" :visible.sync="dialogTableVisible" width="80%">
    <el-table :data="gridData[this.current_id]">
    <el-table-column type="expand">
      <template slot-scope="scope">
        <span class="spe-score">문제 : {{scope.row.question}}</span>
        <span class="spe-score">모범답안 : {{scope.row.gold_answer}}</span>
      </template>
    </el-table-column>
      <el-table-column sortable align="center" type="index" property="problem_num" label="문제번호" width="100"></el-table-column>
      <el-table-column 
      align="center" property="keyword" label="키워드">
        <template slot="header">
            <el-popover ref="fromPopOver" after-leave=""placement="top-start" width="200" trigger="hover">
              <span>푸른색은 정확히 일치한 키워드를, 초록색은 유의어 키워드를 나타냅니다</span>
            </el-popover>
            <span>키워드 <i
                v-popover:fromPopOver
                class="el-icon-info
                text-blue" />
            </span>
          </template>
        <template slot-scope="scope">
          <div class="can-keyword" v-for="(v, i) in scope.row.keyword">
            <el-tag :type="isIn(v, scope.row.user_keyword, scope.row.user_similarity_keyword)" size="medium">
              {{ v }}
            </el-tag>
          </div>
           <el-divider></el-divider>
          <WordHighlighter :query="array2text(scope.row.user_keyword)" :splitBySpace=True>
            {{scope.row.answer}}
          </WordHighlighter>
        </template>
      </el-table-column>
      <el-table-column align="center" property="similarity" label="세부 점수">
        <template slot-scope="scope">
          <el-tag :type="checkSim(scope.row.sim_score)">{{checkSimText(scope.row.sim_score)}}</el-tag>
          <span class="spe-score">Similarity Score : {{scope.row.sim_score}}</span>
          <span class="spe-score">Keyword Score : {{scope.row.keyword_score}}</span>
        </template>
      </el-table-column>
      <el-table-column align="center" property="correctness" label="최종 점수" width="150">
        <template slot-scope="scope">
          <span class="fin-score">{{scope.row.final_score}}</span> / 100
        </template>
      </el-table-column>
    </el-table>
  </el-dialog>
</div>
</template>

<script>
import WordHighlighter from "vue-word-highlighter";

export default {
    components: {
      WordHighlighter,
    },
    props : [
      'alpha'
    ],
    data() {
      return {
        True : true,
        studentResult : {},
        current_id : null,
        tableData: [
        //   {
        //   student_id: '20165020',
        //   score : "18/20"
        // },{
        //   student_id: '20182222',
        //   score: "17/20"
        // }
        ],
      gridData: {
        // "20165020" : [{
        //   problem_num: 'Q1',
        //   question : "여러 제과점이 서로 경쟁을 하면 소비자에게 어떤 점이 좋을까요?",
        //   gold_answer : "제품의 가격이 낮아지고, 품질이 올라간다(좋아진다, 높아진다). 또 제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.",
        //   answer : "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다.",
        //   keyword : ["가격", "품질", "다양성", "혜택", "이득"],
        //   user_keyword : ["가격", "이득"],
        //   sim_score : 0.67,
        //   keyword_score : 0.25,
        //   final_score : 0.46,
        //   correctness : 1,
        // },{
        //   problem_num: 'Q2',
        //   question : "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다.",
        //   keyword : ["가격", "품질", "다양성", "혜택"],
        //   user_keyword : ["가격"],
        //   sim_score : 0.67,
        //   keyword_score : 0.25,
        //   final_score : 0.46,
        //   correctness : 1,
        // }],
        // "20182222" : [{
        //   problem_num: 'Q1',
        //   answer : "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다.",
        //   keyword : ["가격", "품질", "다양성", "혜택"],
        //   user_keyword : ["가격"],
        //   sim_score : 0.67,
        //   keyword_score : 0.25,
        //   final_score : 0.46,
        //   correctness : 1,
        // },{
        //   problem_num: 'Q2222',
        //   answer : "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다.",
        //   keyword : ["가격", "품질", "다양성", "혜택"],
        //   user_keyword : ["가격"],
        //   sim_score : 0.67,
        //   keyword_score : 0.25,
        //   final_score : 0.46,
        //   correctness : 1,
        // }]
        },
        dialogTableVisible: false,
        rawData : null,
      }
    },
    computed : {
    loading(){
      return !this.$store.getters.getLoadingStatus
    }

    },
    watch : {
      alpha(val, old){
        this.preprocess_modal(this.rawData.problem, val)
        this.preprocess_table(this.gridData)
      },
      loading(val, old){
        this.rawData = this.$store.getters.getResult
        //console.log(data)
        this.preprocess_modal(this.rawData.problem)
        this.preprocess_table(this.gridData)
      }
    },
    methods: {
      checkSim(sim_score){
        let type = ""
        if(sim_score >= 0.75){
          type = "success"
        }else if(sim_score >= 0.35){
          type = "warning"
        }else{  
          type = "danger"
        }
        return type
      },
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
      array2text(arr){
        return arr.join(" ")
      },
      isIn(val, user_keyword, user_similarity_keyword){
        let type = ""
        if(user_keyword.includes(val)){
          type=""
        }else if(user_similarity_keyword.includes(val)){
          type="success"
        }else{
          type = "info"
        }
        return type
      },
      handleEdit(index, row) {
        this.current_id = row.student_id
        //console.log(this.current_id)
      },
      calTotalScore(key_score, sim_score, alpha){
        
        let a = alpha / 100
        let final_score =  ((key_score * a  + sim_score * (1 - a)) * 100).toFixed(2)
        
        return final_score

      },
      preprocess_modal(data, alpha=50){
        this.studentResult = {}

        data.forEach((v, i) => {
          //문제별
          //학생
          v.result.forEach((value, index) => {
            let student_id = value.student_id

            let property = {
              question : v.question,
              gold_answer : v.gold_answer,
              answer : value.answer,
              keyword : v.keywords,
              user_keyword : value.match_info.keyword,
              user_similarity_keyword : value.match_info.similarity_keyword,
              sim_score : value.sim_score,
              sim_message : this.checkSimText(value.sim_score),
              keyword_score : value.keyword_score,
              final_score : this.calTotalScore(value.keyword_score, value.sim_score, alpha)

            }
            // console.log(property)
            if(this.studentResult.hasOwnProperty(student_id)){
              this.studentResult[student_id].push(property)
            }else{
              this.studentResult[student_id] = [property]
            }            
          })
          
        })
      
      this.gridData = this.studentResult

      },

      preprocess_table(data){
        this.tableData = []
        
        for(let student_id in data){
          
          let score = 0
          data[student_id].forEach((v, i) => {
            score += v.final_score
          })
          //console.log(score)
          this.tableData.push({
            student_id : student_id,
            score : `${Number(score)} / ${data[student_id].length * 100}`

          })
        }
        
      }
    },
    mounted(){
      //변경감지
    // let data = { "problem": [ { "problem_idx": 0, "question": "여러 제과점이 서로 경쟁을 하면 소비자에게 어떤 점이 좋을까요?", "gold_answer": "제품의 가격이 낮아지고, 품질이 올라간다(좋아진다, 높아진다). 또 제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.", "keywords": [ "가격", "품질", "다양성", "혜택" ], "result": [ { "student_id": "20165020", "answer": "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다.", "sim_score": 0.5621, "keyword_score": 0.25, "total_score": 0.8121, "final_score": 0.2, "match_info": { "keyword": [ "가격" ], "start_idx": [ [ 0 ], [], [], [] ], "end_idx": [ [ 2 ], [], [], [] ] } }, { "student_id": "20182222", "answer": "물건의 가격이 더 싸지고, 서비스가 더 좋아진다.", "sim_score": 0.5621, "keyword_score": 0.25, "total_score": 0.8121, "final_score": 0.2, "match_info": { "keyword": [ "가격" ], "start_idx": [ [ 4 ], [], [], [] ], "end_idx": [ [ 6 ], [], [], [] ] } } ] }, { "problem_idx": 1, "question": "높은 산에서 과자봉지가 부풀어 오르는 이유는 무엇일까요?", "gold_answer": "고도가 높아지면 공기의 압력이 낮아지는데, 온도가 일정할 때 압력이 작아지면 기체의 부피는 증가하므로 과자 봉지 내부 기체의 부피가 증가하기 때문이다.", "keywords": [ "압력", "공기", "고도", "온도", "부피" ], "result": [ { "student_id": "20165020", "answer": "고도를 높아지면 공기의 압력이 낮아지기 때문에", "sim_score": 0.5621, "keyword_score": 0.6, "total_score": 1.1621, "final_score": 0.29, "match_info": { "keyword": [ "압력", "공기", "고도" ], "start_idx": [ [ 13 ], [ 9 ], [ 0 ], [], [] ], "end_idx": [ [ 15 ], [ 11 ], [ 2 ], [], [] ] } }, { "student_id": "20182222", "answer": "기체의 부피가 증가하기 때문에", "sim_score": 0.5621, "keyword_score": 0.2, "total_score": 0.7621, "final_score": 0.19, "match_info": { "keyword": [ "부피" ], "start_idx": [ [], [], [], [], [ 4 ] ], "end_idx": [ [], [], [], [], [ 6 ] ] } } ] } ] }

    // this.preprocess_modal(data.problem)
    // this.preprocess_table(this.gridData)

    }
    
}
</script>

<style lang="scss" scoped>
.can-keyword{
  display : inline-block;
  margin-right : 5px;
}
.el-divider{
  margin : 10px 0 ;
}
.spe-score{
  display:block;
  
}
.fin-score{
  font-weight: bold;
  font-size : 26px;
}
</style>