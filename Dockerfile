FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime


COPY . /usr/src/app
RUN chmod +x /usr/src/app

WORKDIR /usr/src/app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3 python3-pip build-essential libssl-dev libffi-dev python3-dev python3-numpy python3-lxml libxml2-dev libxslt1-dev -y
RUN pip3 install --only-binary :all: -r requirements.txt --break-system-packages
RUN pip3 install -r src/dashboard/requirements.txt --break-system-packages
RUN pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless -y

CMD ["bash", "start.sh"]