# Gompada - Backend

## 1. environtment
* FastAPI
* python 3.8.x (tested on 3.8.10)

## 2. install

* Make your virtual envorionment
```bash
$ python -m venv <your-virtual-env-name>
$ source <your-virtual-env-name>/bin/activate
```

* Install modules on your virtual environment
```bash
$ pip install -r requirements.txt
```

## 3. Execute

### Development
```bash
'if you want to run this with frontend, use port 30001, cuase the default devServer proxy is http://127.0.0.1:30001/api in frontend.
see https://github.com/boostcampaitech3/final-project-level3-nlp-03/blob/main/frontend/.env.dev
'
$ cd app
$ uvicorn main:app --host=0.0.0.0 --port=30001 --reload
```
### Production
```bash
$ cd app
$ gunicorn -k uvicorn.workers.UvicornWorker --access-logfile ./gunicorn-access.log main:app --bind 0.0.0.0:30001 --workers 1 --daemon
```
## 4. DOCS
1. Execute server(local)
2. Goto http://127.0.0.1/docs
3. swagger
## 5. Reference

