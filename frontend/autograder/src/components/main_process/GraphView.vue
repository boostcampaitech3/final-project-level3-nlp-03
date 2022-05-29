<template>

  <el-row :gutter="10">
  <el-col :sm="24" :md="12" >
    <h3>점수 통계</h3>
    <GChart
    type="ColumnChart"
    :data="chartData"
    :options="chartOptions"
    />
  </el-col>
  <el-col :sm="24" :md="12">
    
  </el-col>
  <el-col :sm="24" :md="12" ><div class="grid-content bg-purple"></div></el-col>
  <el-col :sm="24" :md="12"><div class="grid-content bg-purple-light"></div></el-col>
</el-row>
</template>

<script>
import { GChart } from 'vue-google-charts/legacy'

export default {
  components: {
    GChart
  },
  data(){
    return{
      chartData: [
        ['Year', 'Sales', 'Expenses', 'Profit'],
        ['2014', 1000, 400, 200],
        ['2015', 1170, 460, 250],
        ['2016', 660, 1120, 300],
        ['2017', 1030, 540, 350]
      ],
      chartOptions: {
        chart: {
          title: 'Company Performance',
          subtitle: 'Sales, Expenses, and Profit: 2014-2017',
        }
      }
    }
  },
  methods : {

  },
  mounted(){

  }
    
}
</script>

<style lang="scss">
  .el-col {
    border-radius: 4px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
</style>