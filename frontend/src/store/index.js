import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    test_name : "",
    input_data : {
      data : []
    }
  },
  getters: {
    getProblem(state){
      return state.input_data
    }
  },
  mutations: {
    addProblem(state, payload){
      state.input_data.data.push(payload)
    },
  },
  actions: {

  },
  modules: {
    
  }
})
