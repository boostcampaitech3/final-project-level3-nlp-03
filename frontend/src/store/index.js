import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    test_name : "",
    loading : true,
    input_data : {
      problem : []
    },
    result_data : {
      problem : []
    }
  },
  getters: {
    getProblem(state){
      return state.input_data
    },
    getTestname(state){
      return state.test_name
    },
    getResult(state){
      return state.result_data
    },
    getLoadingStatus(state){
      return state.loading
    }
  },
  mutations: {
    initializeResult(state){
      state.result_data.problem = []
    },
    addProblem(state, payload){
      state.input_data.problem.push(payload)
    },
    addResult(state, payload){
      state.result_data.problem.push(payload)
    },
    loadingPending(state){
      state.loading = true
    },
    loadingDone(state){
      state.loading = false
    }
  },
  actions: {

  },
  modules: {
    
  }
})


