# The existing official local base was resolved to this immutable digest.
ARG BASE_IMAGE=python@sha256:420310dd2ff7895895f0f1f9d15cae5a95dabceb8f1d6b9a23ef33c2c1c542c3
FROM ${BASE_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential fonts-dejavu-core fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.11.16
RUN uv venv /opt/venv
COPY containers/requirements/core.txt /opt/requirements/core.txt
RUN uv pip install --python /opt/venv/bin/python -r /opt/requirements/core.txt
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/work/pkgs/symbolic/src:/work/pkgs/spdml_layer/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_ADDOPTS="-p no:cacheprovider" \
    WANDB_MODE=disabled \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /work
CMD ["python", "-m", "ml_platform_symbolic", "--help"]

FROM dependencies AS core
COPY pkgs/symbolic /opt/symbolic
RUN uv pip install --python /opt/venv/bin/python -e /opt/symbolic
ENV PYTHONPATH="/work/pkgs/spdml_layer/src"

FROM dependencies AS classic-dependencies
RUN uv pip install --python /opt/venv/bin/python --torch-backend cpu torch==2.7.1
COPY containers/requirements/classic.txt /opt/requirements/classic.txt
RUN uv pip install --python /opt/venv/bin/python -c /opt/requirements/core.txt -r /opt/requirements/classic.txt
COPY pkgs/symbolic/src/ml_platform_symbolic/engines/torchsisso_precision.py /tmp/torchsisso_precision.py
RUN python /tmp/torchsisso_precision.py --package-dir /opt/venv/lib/python3.11/site-packages/TorchSisso
RUN python -c "import physo, ProGED, parfam, TorchSisso; print('native imports passed')"
# Pin and verify the official Julia binary; cold setup is separate from search budgets.
ADD https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.7-linux-x86_64.tar.gz /tmp/julia.tar.gz
RUN echo 'aa5924114ecb89fd341e59aa898cd1882b3cb622ca4972582c1518eff5f68c05  /tmp/julia.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/julia && tar -xzf /tmp/julia.tar.gz --strip-components=1 -C /opt/julia && rm /tmp/julia.tar.gz
ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    PYTHON_JULIAPKG_EXE=/opt/julia/bin/julia \
    JULIA_NUM_THREADS=2
RUN python -c "import pysr; print(pysr.__version__)"
COPY pkgs/symbolic/src/ml_platform_symbolic/engines/parfam_runtime.py /tmp/parfam_runtime.py
RUN python /tmp/parfam_runtime.py --package-dir /opt/venv/lib/python3.11/site-packages/parfam
COPY containers/patch_physo_unit_roundoff.py /tmp/patch_physo_unit_roundoff.py
COPY containers/patch_physo_runtime.py /tmp/patch_physo_runtime.py
RUN python /tmp/patch_physo_runtime.py --package-dir /opt/venv/lib/python3.11/site-packages/physo

FROM classic-dependencies AS classic
ENV PYTHONPATH="/work/pkgs/spdml_layer/src"
COPY pkgs/symbolic /opt/symbolic
RUN uv pip install --python /opt/venv/bin/python -e /opt/symbolic
RUN python -m ml_platform_symbolic.engines.torchsisso_budget --package-dir /opt/venv/lib/python3.11/site-packages/TorchSisso

FROM classic-dependencies AS posterior-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*
COPY containers/requirements/build.txt /opt/requirements/build.txt
RUN CMAKE_BUILD_PARALLEL_LEVEL=2 uv pip install --python /opt/venv/bin/python \
    --build-constraint /opt/requirements/build.txt -c /opt/requirements/core.txt \
    bingo-nasa==0.5.7 'git+https://github.com/nasa/pysips.git@e155455d1dcd569495655d49f8438ab4e2636c3f'

FROM posterior-dependencies AS posterior
COPY pkgs/symbolic /opt/symbolic
RUN uv pip install --python /opt/venv/bin/python -e /opt/symbolic

FROM classic-dependencies AS gpu-dependencies
RUN uv pip install --python /opt/venv/bin/python --torch-backend cu128 torch==2.7.1+cu128
RUN uv pip install --python /opt/venv/bin/python -c /opt/requirements/core.txt \
    'git+https://github.com/x66ccff/PSRN.git@a4f6f951c95af6a1f5c9c0f057be432e319c73f8'

FROM gpu-dependencies AS gpu
COPY pkgs/symbolic/src/ml_platform_symbolic/engines/pse_runtime.py /tmp/pse_runtime.py
RUN python /tmp/pse_runtime.py --package-dir /opt/venv/lib/python3.11/site-packages/psrn
COPY pkgs/symbolic /opt/symbolic
RUN uv pip install --python /opt/venv/bin/python -e /opt/symbolic

FROM dependencies AS e2e-dependencies
RUN uv pip install --python /opt/venv/bin/python --torch-backend cpu torch==2.7.1
COPY containers/requirements/e2e.txt /opt/requirements/e2e.txt
RUN uv pip install --python /opt/venv/bin/python -c /opt/requirements/core.txt -r /opt/requirements/e2e.txt
RUN git clone https://github.com/facebookresearch/symbolicregression.git /opt/e2e \
    && git -C /opt/e2e checkout --detach c4144c99d078dd611795338e24fb3da49a32b9d8
COPY pkgs/symbolic/src/ml_platform_symbolic/engines/e2e_runtime.py /tmp/e2e_runtime.py
RUN python /tmp/e2e_runtime.py --package-dir /opt/e2e/symbolicregression
COPY pkgs/symbolic/src/ml_platform_symbolic/engines/decoder_runtime.py /tmp/decoder_runtime.py
RUN python /tmp/decoder_runtime.py --engine e2e --source /opt/e2e/symbolicregression/model/transformer.py
ENV PYTHONPATH="/work/pkgs/symbolic/src:/work/pkgs/spdml_layer/src:/opt/e2e"
RUN python -c "from symbolicregression.model import SymbolicTransformerRegressor; import numexpr, sympytorch"

FROM e2e-dependencies AS e2e
COPY pkgs/symbolic /opt/symbolic
RUN uv pip install --python /opt/venv/bin/python -e /opt/symbolic
ENV PYTHONPATH="/opt/e2e"
