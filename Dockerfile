FROM python:3.10-bullseye

RUN apt-get update
RUN apt-get install -y apt-utils \
    && apt-get install -y vim \
    && apt-get install -y telnet \
    && apt-get install -y iputils-ping \
    && apt-get install -y curl\
    && apt-get install -y libgl1

RUN mkdir -p /usr/local/bin/cde

WORKDIR /usr/local/bin/cde

COPY requirements.txt requirements.txt

RUN echo 'Installing libraries' \
    && pip install -r requirements.txt \
    && echo 'Finished'

# Command to keep the container running
CMD ["sleep", "infinity"]

