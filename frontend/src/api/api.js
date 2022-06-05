// 파일경로: src/api/api.js
// 작성한 axios 인터셉터를 가져옵니다.
import { Send } from '@/utils/send'
import qs from 'qs';
//params나 query는 모두 javascript object 형태로 넘겨줘야 합니다.

/*
const headers = {
    'Authorization' : `Bearer ${store.state.accessToken}`
}
*/


export default {
    //sample
    requireAuth(type){

        let content = "application/json"
        const headers = {
            'Content-Type': content
        }
        return headers
    },
    getHome(){
        //홈화면
        return Send({
            url: `/home/`,
            method: 'get',
            headers : this.requireAuth()
            
        })
    },
    ping(){
        return Send({
            url : '/',
            method : 'get',
            headers : this.requireAuth(),
        })
    },
    sendData(data){

      return Send({
        url:'/input',
        method : 'post',
        data : data,
        headers : this.requireAuth()
      })
    }

  
}