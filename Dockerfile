FROM python:3.9.6
#RUN apt-get install apt-transport-https

RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN apt-get update -qq && apt-get install -y nodejs yarn

RUN mkdir -p /app/light_bulb/ui/
WORKDIR /app
ADD Makefile /app/Makefile

ADD light_bulb/ui/public /app/light_bulb/ui/public
ADD light_bulb/ui/src /app/light_bulb/ui/src
ADD light_bulb/ui/package.json /app/light_bulb/ui/package.json
ADD light_bulb/ui/yarn.lock /app/light_bulb/ui/yarn.lock

# Install vendor dependencies
##RUN make vendor vendor/glove.6B

# Install python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel                                                                                                                                                                                                
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

#RUN pip3 install virtualenv
#RUN make .virt
#
## Add code
#ADD . /app
#
#WORKDIR /app
RUN make light_bulb/ui/build/index.html
