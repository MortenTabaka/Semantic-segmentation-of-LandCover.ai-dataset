FROM tensorflow/tensorflow:2.5.1-gpu-jupyter
WORKDIR /app

RUN python3 -m pip install --upgrade pip

COPY dockerfile-requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]