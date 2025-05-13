FROM python:3.13-slim

RUN pip install -U pip

COPY . /

RUN pip install -r ./requirements.txt --upgrade

CMD [ "python", "main.py" ]
