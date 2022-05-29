<template>
<div>
    <h2>결과 확인</h2>
    <table-view></table-view>
    <h2>결과 통계</h2>
    <graph-view></graph-view>
</div>
</template>

<script>
import TableView from '@/components/main_process/TableView.vue'
import GraphView from '@/components/main_process/GraphView.vue'

export default {
  components : {
    TableView,
    GraphView
  }
}
</script>


<style lang="scss" scoped>

</style>