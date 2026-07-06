ARG CUDA_VERSION="12"
ARG ENVIRONMENT="default"

FROM ghcr.io/prefix-dev/pixi:noble AS build

# Redeclaring ARGS in a stage without a value inherits the global default
ARG CUDA_VERSION
ARG ENVIRONMENT

WORKDIR /app
COPY . .
# Providing git so that the pypi installs from Git can proceed
RUN pixi global install git
# CONDA_OVERRIDE_CUDA can be set to a non-empty value with no problems for CPU
# specific builds as the environment is already locked and this just installs it.
RUN export CONDA_OVERRIDE_CUDA="$CUDA_VERSION" && \
    pixi install --locked --environment $ENVIRONMENT
# Activate ENVIRONMENT at container runtime by using the Pixi environment
# activation script as the container ENTRYPOINT
RUN echo "#!/usr/bin/env bash" > /app/entrypoint.sh && \
    pixi shell-hook --environment $ENVIRONMENT --shell bash >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:noble AS final

ARG ENVIRONMENT

WORKDIR /app
COPY --from=build /app/.pixi/envs/$ENVIRONMENT /app/.pixi/envs/$ENVIRONMENT
COPY --from=build /app/pixi.toml /app/pixi.toml
COPY --from=build /app/pixi.lock /app/pixi.lock
# The ignore files are needed for 'pixi run' to work in the container
COPY --from=build /app/.pixi/.gitignore /app/.pixi/.gitignore
COPY --from=build /app/.pixi/.condapackageignore /app/.pixi/.condapackageignore
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]