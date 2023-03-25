FROM tensorflow/tensorflow:2.5.1-gpu-jupyter
WORKDIR /app

RUN python3 -m pip install --upgrade pip

ENV NVIDIA_VISIBLE_DEVICES all

COPY dockerfile-requirements.txt /app/dockerfile-requirements.txt
RUN pip install -r dockerfile-requirements.txt

COPY . /app

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]