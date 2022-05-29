<template>
<div>
  <el-table
    :data="tableData.filter(data => !search || data.name.toLowerCase().includes(search.toLowerCase()))"
    style="width: 100%">
    <el-table-column
      label="Index"
      prop="index">
    </el-table-column>
    <el-table-column
      label="Student ID"
      prop="student_id">
    </el-table-column>
    <el-table-column
      label="Score"
      prop="score">
    </el-table-column>
    <el-table-column
      align="right">
      <template slot="header" slot-scope="scope">
        <el-input
          v-model="search"
          size="mini"
          placeholder="Type to search"/>
      </template>
      <template slot-scope="scope">
        <el-button
          size="mini"
          @click="handleEdit(scope.$index, scope.row); dialogTableVisible = !dialogTableVisible">자세히 보기
        </el-button>
      </template>
    </el-table-column>
  </el-table>
  <el-dialog title="채점 결과" :visible.sync="dialogTableVisible" width="90%">
    <el-table :data="gridData">
      <el-table-column property="problem_num" label="문제번호"></el-table-column>
      <el-table-column property="keyword" label="키워드">
        <template slot-scope="scope">
          <div slot="reference" class="name-wrapper">
            <el-tag size="medium">{{ scope.row.keyword }}</el-tag>
          </div>
      </template>
      </el-table-column>
      <el-table-column property="similarity" label="유사도 점수"></el-table-column>
      <el-table-column property="correctness" label="정답"></el-table-column>
    </el-table>
  </el-dialog>
</div>
</template>

<script>
export default {
data() {
      return {
        tableData: [{
          index: '1',
          student_id: '20165020',
          score : "18/20"
        },{
          index: '2',
          student_id: '20182222',
          score: "17/20"
        }],
        search: '',


  gridData: [{
          problem_num: 'Q1',
          keyword: "대류 ",
          similarity: 20.5,
          correctness : 1,
        }],
        dialogTableVisible: false,
      }
    },
    methods: {
      handleEdit(index, row) {
        console.log(index, row);
      },
      handleDelete(index, row) {
        console.log(index, row);
      }
    },
}
</script>

<style lang="scss" scoped>

</style>