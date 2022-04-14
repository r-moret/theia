FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install python3-venv -y

ENV VIRTUAL_ENV="theia-env"

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="/$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN pip install black numpy matplotlib jupyter tensorflow

CMD ["bash"]