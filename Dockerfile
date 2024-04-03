
FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG REQUIREMENTS_FILE=cu_11_8.txt


RUN mkdir /workspace/NN


# RUN git clone https://inkve.ddns.net:42379/inkve/NN
COPY ./requirements/${REQUIREMENTS_FILE} /workspace
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace && pip install -r ${REQUIREMENTS_FILE}

WORKDIR /workspace/NN
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0"]
