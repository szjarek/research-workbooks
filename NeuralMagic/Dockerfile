FROM python:3.8.8-slim-buster

RUN apt-get update && apt-get -y install wget

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
CMD jupyter notebook --allow-root --ip 0.0.0.0