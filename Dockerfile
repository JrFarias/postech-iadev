FROM python:3.10-slim

USER root

RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y \
    libgomp1 \
    pandoc \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyterlab poetry==1.8.3

COPY . /tech-challenge

WORKDIR /tech-challenge

RUN poetry config virtualenvs.create false

RUN poetry install

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--browser=google-chrome-stable %s"]