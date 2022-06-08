<template>
<el-container>
    <el-header>
      <main-header></main-header>
    </el-header>
    <el-main>
      <input-basic></input-basic>
      <input-problem></input-problem>
      <show-result></show-result>
    </el-main>
    <el-footer>
      <main-footer></main-footer>
    </el-footer>
</el-container>
</template>

<script>
// @ is an alias to /src

//header
import MainHeader from '@/components/header/MainHeader.vue'
//
import InputBasic from '@/components/main_input/InputBasic.vue'
import InputProblem from '@/components/main_input/InputProblem.vue'
import ShowResult from '@/components/main_process/ShowResult.vue'

//footer
import MainFooter from '@/components/footer/MainFooter.vue'

export default {
  name: 'HomeView',
  components: {
    MainHeader,
    InputBasic,
    InputProblem,
    ShowResult,
    MainFooter

  }
}
</script>

<style lang="scss">
@import url("//unpkg.com/element-ui@2.15.8/lib/theme-chalk/index.css");
.el-header{
  height : auto !important; 
  padding : 0;
}
  
  .el-aside {
    background-color: #D3DCE6;
    color: #333;
    text-align: center;
    line-height: 200px;
  }
  
  .el-main {
     background-color: #F2F6FC;
     border-radius: 20px;
     padding : 25px;
  }
  .el-footer{
    background-color : #f4f6f8;
    border-top : 1px solid #dddddd;
    height : auto !important;
    padding: 20px;
  }
  
  body > .el-container {
    margin-bottom: 40px;
  }
  
  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }
  
  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }

</style>
