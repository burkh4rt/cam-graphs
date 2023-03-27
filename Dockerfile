# use this file with:
# docker build --no-cache -t thistle .
# docker run --rm -ti -v $(pwd):/src thistle model.py

FROM python:3.9.16-bullseye

# create venv in the path
RUN python -m venv /opt/venv
ENV PATH /opt/venv/bin:$PATH

# install requirements
RUN pip install --upgrade pip \
                          wheel \
                          numpy==1.23.5 \
                          Pillow==9.4.0 \
                          MarkupSafe==2.1.2 \
 && pip install torch==2.0.0 \
                torchvision==0.15.1 \
                torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install torch_geometric==2.3.0 \
 && pip install torch_scatter==2.1.1 \
                torch_sparse==0.6.17 \
                torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cpu.html \
 && pip install shap==0.41.0 \
                optuna==3.1.0 \
                matplotlib==3.7.1 \
                pandas==1.5.3

# switch to non-root user
# RUN useradd felixity
# USER felixity
# WORKDIR /home/felixity
WORKDIR /src

ENTRYPOINT [ "python" ]
