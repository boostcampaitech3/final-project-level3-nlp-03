<template>
<div class="input-basic">
  <h2>기본 정보 추가</h2>
  <el-input class="input-form" placeholder="Please input" v-model="input1">
    <template slot="prepend">시험 이름 </template>
  </el-input>
  
</div>
</template>

<script>
export default {
  data(){
    return {
      input1 : "",
    }
  },
}
</script>

<style lang="scss">
.input-basic{
  margin : 0 0 20px 0;
}
</style>