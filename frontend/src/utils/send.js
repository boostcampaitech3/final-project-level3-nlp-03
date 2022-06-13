// 파일경로: src/utils/Send.js

import axios from 'axios'

/*
    axios 인스턴스를 생성합니다.
    생성할때 사용하는 옵션들 (baseURL, timeout, headers 등)은 다음 URL에서 확인할 수 있습니다.
    https://github.com/axios/axios 의 Request Config 챕터 확인
*/
const Send = axios.create({
    baseURL: process.env.VUE_APP_SERVER_URL,
    //https://helpdiana.site/api
    //http://localhost:8081/api
    timeout: 10000,
    headers : {
        "Access-Control-Allow-Origin" : "*",
        "Access-Control-Allow-Headers" : "*",
        "Access-Control-Allow-Methods" : "GET, DELETE, PUT, POST"
        //
    }

  });

/*
    1. 요청 인터셉터를 작성합니다.
    2개의 콜백 함수를 받습니다.
    1) 요청 바로 직전 - 인자값: axios config
    2) 요청 에러 - 인자값: error
*/
Send.interceptors.request.use(
    function (config) {
        // 요청 바로 직전
        // axios 설정값에 대해 작성합니다.
        config.withCredentials = true
        config.headers["Access-Control-Allow-Origin"] = '*'
        config.headers["Access-Control-Allow-Headers"] = "*";
        config.headers['Access-Control-Allow-Methods'] = "GET, DELETE, PUT, POST";
        
        return config;
    }, 
    function (error) {
        
        return Promise.reject(error);
    }
);

/*
    2. 응답 인터셉터를 작성합니다.
    2개의 콜백 함수를 받습니다.
    1) 응답 정성 - 인자값: http response
    2) 응답 에러 - 인자값: http error
*/
Send.interceptors.response.use(
    function (response) {
    /*
        http status가 200인 경우
        응답 바로 직전에 대해 작성합니다. 
        .then() 으로 이어집니다.
    */
        return response;
    },

    function (error) {
        console.log(error)


        if(error.response.status == 401){
            //권한이 없음 재로그인 시도해야함
        }
        console.log(error.response.status)
        //400번대 Error 처리할것
    /*
        http status가 200이 아닌 경우
        응답 에러 처리를 작성합니다.
        .catch() 으로 이어집니다.    
    */
        return Promise.reject(error);
    }
);
// 생성한 인스턴스를 익스포트 합니다.
export {
    Send,
};