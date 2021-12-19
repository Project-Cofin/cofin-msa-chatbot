FROM python:3.8

WORKDIR /backend

COPY . .
COPY requirements.txt requirements.txt

CMD ["python", "manage.py", "runserver", "localhost:8000"]

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000