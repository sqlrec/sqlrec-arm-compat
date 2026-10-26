FROM python:3.11-slim-bookworm AS wheelbuilder

WORKDIR /workspace
COPY packages ./packages
RUN python -m pip wheel --no-deps -w /wheelhouse ./packages/pyfg ./packages/graphlearn

FROM scratch AS wheels
COPY --from=wheelbuilder /wheelhouse/ /
