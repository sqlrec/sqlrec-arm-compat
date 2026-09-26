FROM python:3.11-slim-bookworm AS dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python -m pip install --no-cache-dir \
      'numpy>=1.24,<3' \
      'pyarrow>=14,<26' \
      'pyfarmhash==0.4.0' \
      'pytest>=8,<10'

COPY . .
RUN python -m pip install --no-cache-dir --no-deps \
      -e ./packages/pyfg \
      -e ./packages/graphlearn

FROM dev AS wheelbuilder
RUN python -m pip wheel --no-deps -w /wheelhouse \
      ./packages/pyfg ./packages/graphlearn pyfarmhash==0.4.0

FROM scratch AS wheels
COPY --from=wheelbuilder /wheelhouse/ /

FROM dev AS test
CMD ["python", "-m", "pytest", "-q"]
