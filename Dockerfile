# ./Dockerfile
FROM python:3

MAINTAINER sh Lee <lmn1654@gmail.com>
# Docker의 컨테이너를 생성 및 관리 하는 사람의 정보를 기입해줍니다.
# 컨테이너 내에서 코드가 실행될 경로 설정

## Install packages
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
# django 서버의 포트를 8000로 지정하였으므로 Docker의 컨테이너 또한 8000 포트를 열어줍니다.

CMD ["python", "manage.py", "runserver", "localhost:8000"]
# 이동한 디렉토리에서 django를 가동시켜주는 코드를 작성합니다. 여기서 port는 8000로 실행시키겠습니다.

