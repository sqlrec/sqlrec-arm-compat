ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS dev

ARG NUMPY_SPEC="numpy>=1.24,<3"
ARG PYARROW_SPEC="pyarrow>=14,<26"

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python -m pip install --no-cache-dir \
      "$NUMPY_SPEC" \
      "$PYARROW_SPEC" \
      'pyfarmhash==0.4.0' \
      'pytest>=8,<10' \
      'coverage>=7,<8'

COPY . .
RUN python -m pip install --no-cache-dir --no-deps \
      -e ./packages/pyfg \
      -e ./packages/graphlearn

FROM dev AS wheelbuilder
RUN python -m pip wheel --no-deps -w /wheelhouse \
      ./packages/pyfg ./packages/graphlearn pyfarmhash==0.4.0

FROM scratch AS wheels
COPY --from=wheelbuilder /wheelhouse/ /

FROM wheelbuilder AS test
RUN python -m pip uninstall -y pyfg graphlearn pyfarmhash \
    && python -m pip install --no-cache-dir --no-deps \
      /wheelhouse/pyfg-*.whl /wheelhouse/graphlearn-*.whl /wheelhouse/pyfarmhash-*.whl
CMD ["sh", "-c", "python -m coverage run --branch --source=pyfg,graphlearn,sqlrec_arm_compat -m pytest -q && python -m coverage report --fail-under=100"]
