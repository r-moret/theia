FROM tensorflow/tensorflow:latest-gpu

COPY .. "/workspaces/theia"

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install python3-venv -y

ENV VIRTUAL_ENV="theia-env"

RUN python3 -m venv $VIRTUAL_ENV

WORKDIR "/workspaces/theia"

RUN source "/theia-env/bin/activate" \
    && python -m pip install --upgrade pip
RUN source "/theia-env/bin/activate" \
    && pip install black numpy matplotlib jupyter tensorflow
RUN source "/theia-env/bin/activate" \
    && pip install -e theia

CMD ["bash"]