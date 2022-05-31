<template>
<div class="logo-container">
  <img class="main-logo" :src="require(`@/assets/logo.png`)">
</div>
</template>

<script>
export default {

}
</script>

<style lang="scss">
.logo-container{
  display : flex;
  justify-content: center;
  align-items: center;
  margin : 1rem 0;
  .main-logo{
    margin : 0 auto;
    width : 150px;
    height : 150px;
  }
}
</style>