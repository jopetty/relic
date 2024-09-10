ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential curl bash

SHELL ["/bin/bash", "-c"]

ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

COPY pyproject.toml /tmp/

RUN /root/.cargo/bin/uv pip install --system -e /tmp/
