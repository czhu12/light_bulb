FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y curl software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get install apt-transport-https

RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN curl -sL https://deb.nodesource.com/setup_9.x | bash

RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y curl software-properties-common yarn nodejs unzip

RUN mkdir -p /app/code/ui/build
RUN mkdir -p /app/code/ui/public
WORKDIR /app

ADD Makefile /app/Makefile

# Install vendor dependencies
#RUN make vendor vendor/glove.6B

# Install python dependencies
ADD requirements.txt /app/requirements.txt

RUN pip3 install virtualenv
RUN make .virt

# Add code
ADD . /app
