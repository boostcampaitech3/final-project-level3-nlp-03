<template>
<div>
    <div style="margin-bottom: 20px;">
    <div>
      <h2>문제, 모법답안, 학생답안 추가</h2>
      <el-tooltip :content="basic_info" placement="top" effect="light">
        <i class="el-icon-info"></i>
      </el-tooltip>
    </div>  
    <el-button
        type="info"
        @click="addTab(editableTabsValue)"
    >
        문제 추가
    </el-button>
       <el-button class="submit-button"
        type="success"
        :loading=loading
        @click="sendResult()"
    >
      제출 및 결과보기
    </el-button> 
    </div>
    <el-tabs v-model="editableTabsValue" type="card" closable @tab-remove="removeTab">
    <el-tab-pane
        v-for="(item, index) in editableTabs"
        :key="item.name"
        :label="item.title"
        :name="item.name"
        
    >
        <add-problem :tab_name="item.title"></add-problem>
    </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>

import Api from '@/api/api'

import AddProblem from '@/components/main_input/AddProblem.vue'
export default {
    components: {
        AddProblem
    },
    props : [],
    data(){
        return {
        loading : false,
        basic_info : "문제 추가 버튼을 통해서 문제를 추가할 수 있고 문제 작성이후 문제 저장 버튼을 눌러야 합니다",
        editableTabsValue: '1',
        editableTabs: [{
          title: '문제 1',
          name: '1',
        },],
        tabIndex: 1
        }
    },
    computed : {
 
      getData() {
        return this.$store.getters.getProblem
      }
    },
    methods : {
      sendResult(){
        //console.log(JSON.stringify(this.getData))
        this.loading = true
        let data = this.getData
        //console.log(data)
        this.$store.commit("loadingPending")

//         let data = {
// "problem": [
//       {
//           "question": "여러 제과점이 서로 경쟁을 하면 소비자에게 어떤 점이 좋을까요?",
//           "gold_answer": "제품의 가격이 낮아지고, 품질이 올라간다(좋아진다, 높아진다). 또 제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.",
//           "keywords": [
//               "가격",
//               "품질",
//               "다양성",
//               "혜택"
//           ],
//           "answers": [
//               [
//                   "20165020",
//                   "가격이 내려가거나 양이 많아진다. 그래서 소비자에게 이득이 된다."
//               ],
//               [
//                   "20182222",
//                   "물건의 가격이 더 싸지고, 서비스가 더 좋아진다."
//               ]
//           ]
//       },
//       {
//           "question": "높은 산에서 과자봉지가 부풀어 오르는 이유는 무엇일까요?",
//           "gold_answer": "고도가 높아지면 공기의 압력이 낮아지는데, 온도가 일정할 때 압력이 작아지면 기체의 부피는 증가하므로 과자 봉지 내부 기체의 부피가 증가하기 때문이다.",
//           "keywords": [
//               "압력",
//               "공기",
//               "고도",
//               "온도",
//               "부피"
//           ],
//           "answers": [
//               [
//                   "20165020",
//                   "고도를 높아지면 공기의 압력이 낮아지기 때문에"
//               ],
//               [
//                   "20182222",
//                   "기체의 부피가 증가하기 때문에"
//               ]
//           ]
//       }
//   ]
// }
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
        this.loading = false
      });
      },
      addTab(targetName) {
        let newTabName = ++this.tabIndex + '';
        this.editableTabs.push({
          title: `문제 ${this.tabIndex}`,
          name: newTabName,
  
        });
        this.editableTabsValue = newTabName;
        this.$notify({
          title: '문제 추가',
          message: `문제가 추가되었습니다.`,
        });
      },
      removeTab(targetName) {
        let tabs = this.editableTabs;
        let activeName = this.editableTabsValue;
        if (activeName === targetName) {
          tabs.forEach((tab, index) => {
            if (tab.name === targetName) {
              let nextTab = tabs[index + 1] || tabs[index - 1];
              if (nextTab) {
                activeName = nextTab.name;
              }
            }
          });
        }
        
        this.editableTabsValue = activeName;
        this.editableTabs = tabs.filter(tab => tab.name !== targetName);
      }
    },
    mounted(){

    }
}
</script>

<style lang="scss">
h2{
  display : inline-block;
}
.submit-button{
  float : right;
}
</style>