FROM python:3.11-slim-bookworm AS wheelbuilder

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python -m pip wheel --no-deps -w /wheelhouse pyfarmhash==0.4.0
COPY packages ./packages
RUN python -m pip wheel --no-deps -w /wheelhouse ./packages/pyfg ./packages/graphlearn

FROM scratch AS wheels
COPY --from=wheelbuilder /wheelhouse/ /
