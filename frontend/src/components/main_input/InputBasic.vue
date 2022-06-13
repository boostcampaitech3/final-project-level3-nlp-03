<template>
<div class="input-basic">
  <h2>기본 정보 추가</h2>
  <el-tooltip :content="basic_info" placement="top" effect="light">
    <i class="el-icon-info"></i>
  </el-tooltip>
  <el-input class="input-form" placeholder="시험 이름을 입력하세요" v-model="input1">
    <template slot="prepend">시험 이름 </template>
  </el-input>
  
</div>
</template>

<script>
export default {
  data(){
    return {
      input1 : "",
      basic_info : "기본정보 추가에서는 시험 이름을 추가합니다"
    }
  },
  mounted(){
  }
}
</script>

<style lang="scss">
.input-basic{
  margin : 0 0 20px 0;
  h2{
    display : inline-block;
  }
  i{
    margin : 0 5px;
  }
}
</style>