
#############################
## Install python packages ##
#############################

# Create image
FROM python:slim AS py_builder

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# set python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        build-essential \
        # some project dependencies \
        cdo nco \
        # to install cartopy
        proj-bin libproj-dev libgeos-dev  && \
    rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /usr/src/app

# upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels \
        numpy \
        xarray \
        scipy \
        astropy \
        matplotlib \
        pathos \
        netcdf4 \
        cdo \
        nco \
        python-crontab \
        PyYAML
# Shapely y cartopy deben instalarse sin binarios (ver: https://github.com/SciTools/cartopy/issues/837)
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels \
        --no-binary :all: shapely Cartopy



########################
## CREATE FINAL IMAGE ##
########################

# Create image
FROM python:slim AS final_image

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq --no-install-recommends install \
        # to run wheels installation
        gcc \
        # to be able to use cartopy (Python)
        proj-bin libproj-dev libgeos-dev \
        # install Tini (https://github.com/krallin/tini#using-tini)
        tini \
        # to see process with pid 1
        htop \
        # to run sudo
        sudo \
        # to allow edit files
        vim \
        # to run process with cron
        cron && \
    rm -rf /var/lib/apt/lists/*

# Setup cron to allow it run as a non root user
RUN sudo chmod u+s $(which cron)

# Install python dependencies from py_builder
COPY --from=py_builder /usr/src/app/wheels /wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache /wheels/* && \
    rm -rf /wheels

# Create work directory
RUN mkdir -p /opt/ereg

# Set work directory
WORKDIR /opt/ereg

# Copy app code
COPY . .

# Create input and output folders (these folders are too big so they must be used them as volumes)
RUN mkdir -p /data/ereg/descargas
RUN mkdir -p /data/ereg/generados



#######################
## SETUP FINAL IMAGE ##
#######################

# Create image
FROM final_image

# Set passwords
ARG ROOT_PWD="nonroot"
ARG NON_ROOT_PWD="nonroot"

# Pasar a root
USER root

# Modify root password
RUN echo "root:$ROOT_PWD" | chpasswd

# Create a non-root user, so the container can run as non-root
# OBS: the UID and GID must be the same as the user that own the
# input and the output volumes, so there isn't perms problems!!
ARG NON_ROOT_USR="nonroot"
ARG NON_ROOT_UID="1000"
ARG NON_ROOT_GID="1000"
RUN groupadd --gid $NON_ROOT_GID $NON_ROOT_USR
RUN useradd --uid $NON_ROOT_UID --gid $NON_ROOT_GID --comment "Non-root User Account" --create-home $NON_ROOT_USR

# Modify the password of non-root user
RUN echo "$NON_ROOT_USR:$NON_ROOT_PWD" | chpasswd

# Add non-root user to sudoers
RUN adduser $NON_ROOT_USR sudo

# Setup ereg
RUN chown -R $NON_ROOT_UID:$NON_ROOT_GID /data/ereg
RUN chown -R $NON_ROOT_UID:$NON_ROOT_GID /opt/ereg

# Disable group switching
RUN sed -i "s/^group_for_files/# group_for_files/g" /opt/ereg/config.yaml

# Setup cron for run twice a month -- Download files from IRIDL
ARG D_CRON_TIME_STR="0 0 15,16 * *"
ARG D_PYTHON_CMD="cd /opt/ereg && /usr/local/bin/python download_inputs.py --download operational --re-check"
RUN (echo "${D_CRON_TIME_STR} ${D_PYTHON_CMD} >> /proc/1/fd/1 2>> /proc/1/fd/1") | crontab -u $NON_ROOT_USR -
# Setup cron for run once a month -- Run operational forecast
ARG R_CRON_TIME_STR="0 0 17 * *"
ARG R_PYTHON_CMD="cd /opt/ereg && /usr/local/bin/python run_operational_forecast.py --overwrite --combination wsereg --weighting mean_cor --ignore-plotting"
RUN (crontab -u $NON_ROOT_USR -l; echo "${R_CRON_TIME_STR} ${R_PYTHON_CMD} >> /proc/1/fd/1 2>> /proc/1/fd/1") | crontab -u $NON_ROOT_USR -

# Add Tini (https://github.com/krallin/tini#using-tini)
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]

# Run your program under Tini (https://github.com/krallin/tini#using-tini)
CMD ["cron", "-f"]
# or docker run your-image /your/program ...

# Access non-root user directory
WORKDIR /home/$NON_ROOT_USR

# Switch back to non-root user to avoid accidental container runs as root
USER $NON_ROOT_USR

# CONSTRUIR CONTENEDOR
# export DOCKER_BUILDKIT=1
# docker build --file dockerfile \
#        --build-arg ROOT_PWD=nonroot \
#        --build-arg NON_ROOT_PWD=nonroot \
#        --build-arg NON_ROOT_UID=$(stat -c "%u" .) \  # ideally, the user id must be the uid of files in /data/ereg
#        --build-arg NON_ROOT_GID=$(stat -c "%g" .) \  # ideally, the group id must be the gid of files in /data/ereg
#        --tag ereg:latest .

# CORRER OPERACIONALMENTE CON CRON
# docker run --name ereg \
#        --volume /data/ereg/descargas:/data/ereg/descargas \
#        --volume /data/ereg/generados:/data/ereg/generados \
#        --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#        --detach ereg:latest

# CORRER MANUALMENTE EN PRIMER PLANO Y BORRANDO EL CONTENEDOR AL FINALIZAR
# docker run --name ereg \
#        --volume /data/ereg/descargas:/data/ereg/descargas \
#        --volume /data/ereg/generados:/data/ereg/generados \
#        --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#        --rm ereg:latest python /opt/ereg/<script> <args>

# CORRER MANUALMENTE EN SEGUNDO PLANO Y SIN BORRAR EL CONTENEDOR AL FINALIZAR
# NO BORRAR EL CONTENEDOR AL FINALIZAR PERMITE VER LOS ERRORES (EN CASO QUE HAYA ALGUNO)
# docker run --name ereg \
#        --volume /data/ereg/descargas:/data/ereg/descargas \
#        --volume /data/ereg/generados:/data/ereg/generados \
#        --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#        --detach ereg:latest python /opt/ereg/<script> <args>

# VER RAM USADA POR LOS CONTENEDORES CORRIENDO
# docker stats --format "table {{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.PIDs}}\t{{.MemUsage}}" --no-stream

# VER LOGS (CON COLORES) DE CONTENEDOR CORRIENDO EN SEGUNDO PLANO
# docker logs --follow ereg 2>&1 | ccze -m ansi

#
# README
#
# El parámetro " --env DROP_COMBINED_FORECASTS='YES' " crea una variable de entorno en el OS del contenedor
# que establece la respuesta a la siguiente pregunta lanzada por EREG cuando es corrido fuera de un contenedor:
# --> Model/s was added or deleted. Do you want to drop current combined forecasts and update combined_models file?
# Esta pregunta es lanzada solo cuando se detecta que han sido modificados los modelos a ser combinados. Responder
# a esta pregunta con un NO implica que no se actualizarán los modelos utilizados para la calibración, es decir, que
# EREG se seguirá ejecutando utilizando los archivos de calibración producidos antes del cambio detectado en los
# modelos a ser combiandos. Es importante tener en cuenta que el comportamiento por defecto ante esta situación es no
# borrar los archivos de calibración!! Esto para evitar el borrado accidental de los mismos, puesto que producirlos es
# lleva bastante tiempo, principalmente para el periodo retrospectivo o hindcast.
#
