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
  <el-dialog title="Shipping address" :visible.sync="dialogTableVisible">

    <el-table :data="gridData">
      <el-table-column property="date" label="Date" width="150"></el-table-column>
      <el-table-column property="name" label="Name" width="200"></el-table-column>
      <el-table-column property="address" label="Address"></el-table-column>
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
          date: '2016-05-02',
          name: 'John Smith',
          address: 'No.1518,  Jinshajiang Road, Putuo District'
        }, {
          date: '2016-05-04',
          name: 'John Smith',
          address: 'No.1518,  Jinshajiang Road, Putuo District'
        }, {
          date: '2016-05-01',
          name: 'John Smith',
          address: 'No.1518,  Jinshajiang Road, Putuo District'
        }, {
          date: '2016-05-03',
          name: 'John Smith',
          address: 'No.1518,  Jinshajiang Road, Putuo District'
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