<template>
<div>
    <div style="margin-bottom: 20px;">
    <h2>문제, 모법답안, 학생답안 추가</h2>  
    <el-button
        size="small"
        @click="addTab(editableTabsValue)"
    >
        문제 추가
    </el-button>
    </div>
    <el-tabs v-model="editableTabsValue" type="card" closable @tab-remove="removeTab">
    <el-tab-pane
        v-for="(item, index) in editableTabs"
        :key="item.name"
        :label="item.title"
        :name="item.name"
    >
        <add-problem></add-problem>
    </el-tab-pane>
    </el-tabs>
     <el-button class="submit-button"
        size="small"
        @click=""
    >
      제출 및 결과보기
    </el-button> 
  </div>
</template>

<script>

import AddProblem from '@/components/main_input/AddProblem.vue'
export default {
    components: {
        AddProblem
    },
    props : [],
    data(){
        return {
        editableTabsValue: '1',
        editableTabs: [{
          title: '문제 1',
          name: '1',
        },],
        tabIndex: 2
        }
    },
    computed : {

    },
    methods : {
      addTab(targetName) {
        let newTabName = ++this.tabIndex + '';
        this.editableTabs.push({
          title: 'New Tab',
          name: newTabName,
  
        });
        this.editableTabsValue = newTabName;
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
.submit-button{
  float : right;
}
</style>