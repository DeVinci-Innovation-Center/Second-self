FROM ubuntu:20.04

RUN apt update && \
    apt install -y python3-dev python3-pip

RUN pip3 install rasa[full]

COPY . /rasa

WORKDIR /rasa

CMD rasa train

# CMD cd rasa && rasa run --enable-api
