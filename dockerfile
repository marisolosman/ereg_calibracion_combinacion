

##########################
## Set GLOBAL arguments ##
##########################

# Set python version
ARG PYTHON_VERSION=3.10

# Set APP installation folder
ARG APP_HOME=/opt/ereg

# App data folder
ARG APP_DATA=/data/ereg

# Set user name and id
ARG USR_NAME="nonroot"
ARG USER_UID="1000"

# Set group name and id
ARG GRP_NAME="nonroot"
ARG USER_GID="1000"

# Set users passwords
ARG ROOT_PWD="root"
ARG USER_PWD=$USR_NAME

# Set global CRON args
ARG D_CRON_TIME_STR="0 0 15,16 * *"
ARG R_CRON_TIME_STR="0 0 17 * *"

# Set Pycharm version
ARG PYCHARM_VERSION=2023.1



######################################
## Stage 1: Install Python packages ##
######################################

# Create image
FROM python:${PYTHON_VERSION}-slim AS py_builder

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Set python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq upgrade && \
    apt-get -y -qq --no-install-recommends install \
        build-essential \
        # some project dependencies \
        cdo nco \
        # to install cartopy
        proj-bin libproj-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Upgrade pip and install dependencies
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



###############################################
## Stage 2: Copy Python installation folders ##
###############################################

# Create image
FROM python:${PYTHON_VERSION}-slim AS py_final

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq upgrade && \
    apt-get -y -qq --no-install-recommends install \
        # some project dependencies \
        cdo nco \
        # to be able to use cartopy (Python)
        proj-bin libproj-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies from py_builder
RUN python3 -m pip install --upgrade pip
COPY --from=py_builder /usr/src/app/wheels /wheels
RUN python3 -m pip install --no-cache /wheels/* && \
    rm -rf /wheels



###################################
## Stage 3: Create non-root user ##
###################################

# Create image
FROM py_final AS non_root

# Renew ARGs
ARG USR_NAME
ARG USER_UID
ARG GRP_NAME
ARG USER_GID
ARG ROOT_PWD
ARG USER_PWD

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq upgrade && \
    apt-get -y -qq --no-install-recommends install \
        # to run sudo
        sudo && \
    rm -rf /var/lib/apt/lists/*

# Modify root password
RUN echo "root:$ROOT_PWD" | chpasswd

# Create a non-root user, so the container can run as non-root
# OBS: the UID and GID must be the same as the user that own the
# input and the output volumes, so there isn't perms problems!!
# Se recomienda crear usuarios en el contendor de esta manera,
# ver: https://nickjanetakis.com/blog/running-docker-containers-as-a-non-root-user-with-a-custom-uid-and-gid
# Se agregar --no-log-init para prevenir un problema de seguridad,
# ver: https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
RUN groupadd --gid $USER_GID $GRP_NAME
RUN useradd --no-log-init --uid $USER_UID --gid $USER_GID --shell /bin/bash \
    --comment "Non-root User Account" --create-home $USR_NAME

# Modify the password of non-root user
RUN echo "$USR_NAME:$USER_PWD" | chpasswd

# Add non-root user to sudoers and to adm group
# The adm group was added to allow non-root user to see logs
RUN usermod -aG sudo $USR_NAME && \
    usermod -aG adm $USR_NAME

# To allow sudo without password
# RUN echo "$USR_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USR_NAME && \
#     chmod 0440 /etc/sudoers.d/$USR_NAME



###########################################
## Stage 4: Install management packages  ##
###########################################

# Create image
FROM non_root AS base_builder

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install OS packages
RUN apt-get -y -qq update && \
    apt-get -y -qq upgrade && \
    apt-get -y -qq --no-install-recommends install \
        # install Tini (https://github.com/krallin/tini#using-tini)
        tini \
        # to see process with pid 1
        htop \
        # to allow edit files
        vim \
        # to run process with cron
        cron && \
    rm -rf /var/lib/apt/lists/*

# Setup cron to allow it run as a non root user
RUN chmod u+s $(which cron)

# Add Tini (https://github.com/krallin/tini#using-tini)
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]



####################################
## Stage 5: Install and setup APP ##
####################################

# Create image
FROM base_builder AS app_builder

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Renew ARGs
ARG APP_HOME
ARG APP_DATA
ARG USR_NAME
ARG GRP_NAME

# Renew CRON ARGs
ARG D_CRON_TIME_STR
ARG R_CRON_TIME_STR

# Create APP_HOME folder and change its owner
RUN mkdir -p $APP_HOME && chown -R $USR_NAME:$GRP_NAME $APP_HOME

# Copy project
COPY --chown=$USR_NAME:$GRP_NAME . $APP_HOME

# Disable group switching
RUN sed -i "s/^group_for_files/# group_for_files/g" $APP_HOME/config.yaml

# Create input and output folders (these folders are too big so they must be used them as volumes)
RUN mkdir -p $APP_DATA/descargas
RUN mkdir -p $APP_DATA/generados

# Setup folder's correct owner and group
RUN chown -R $USR_NAME:$GRP_NAME $APP_HOME
RUN chown -R $USR_NAME:$GRP_NAME $APP_DATA

# Definir comandos para descarga y calibración de pronósticos
ARG D_PYTHON_CMD="/usr/local/bin/python download_inputs.py --download operational --re-check"
ARG R_PYTHON_CMD="/usr/local/bin/python run_operational_forecast.py --overwrite --combination wsereg --weighting mean_cor --ignore-plotting"

# Crear archivo de configuración de CRON
RUN printf "\n\
# Download input data \n\
${D_CRON_TIME_STR}  cd ${APP_HOME} && ${D_PYTHON_CMD} >> /proc/1/fd/1 2>> /proc/1/fd/1 \n\
# Run operational forecasts \n\
${R_CRON_TIME_STR}  cd ${APP_HOME} && ${R_PYTHON_CMD} >> /proc/1/fd/1 2>> /proc/1/fd/1 \n\
\n" > /tmp/crontab.txt

# Crear script para verificar salud del contendor
RUN printf "#!/bin/bash\n\
if [ \$(ls /tmp/ereg-download.pid 2>/dev/null | wc -l) != 0 ] && \n\
   [ \$(ps | grep download_inputs.py | wc -l) == 0 ] || \n\
   [ \$(ls /tmp/ereg-run-operational-fcst.pid 2>/dev/null | wc -l) != 0 ] && \n\
   [ \$(ps | grep run_operational_forecast.py | wc -l) == 0 ] \n\
then \n\
  exit 1 \n\
else \n\
  exit 0 \n\
fi \n\
\n" > /opt/check-healthy.sh

# Definir variables de entorno para el contendor final
ENV D_CRON_TIME_STR=${D_CRON_TIME_STR}
ENV R_CRON_TIME_STR=${R_CRON_TIME_STR}

# Crear script de inicio.
RUN printf "#!/bin/bash \n\
set -e \n\
\n\
# Reemplazar tiempo ejecución de la descarga de los datos de entrada \n\
crontab -l | sed \"/download_inputs.py/ s|^\S* \S* \S* \S* \S*|\$D_CRON_TIME_STR|g\" | crontab - \n\
crontab -l | sed \"/run_operational_forecast.py/ s|^\S* \S* \S* \S* \S*|\$R_CRON_TIME_STR|g\" | crontab - \n\
\n\
# Ejecutar cron \n\
cron -fL 15 \n\
\n" > /startup.sh
RUN chmod a+x /startup.sh


############################################
## Stage 6.1: Install Pycharm (for debug) ##
############################################

# Create image
FROM app_builder AS pycharm

# Renew ARGs
ARG APP_HOME
ARG USR_NAME
ARG GRP_NAME

# Become root
USER root

# Updata apt cache and install wget
RUN apt-get -y -qq update && \
    apt-get -y -qq upgrade && \
    apt-get -y -qq --no-install-recommends install \
        wget  \
        git

# Renew ARGs
ARG PYCHARM_VERSION

# Download Pycharm IDE
RUN wget https://download.jetbrains.com/python/pycharm-community-${PYCHARM_VERSION}.tar.gz -P /tmp/

# Install packages required to run PyCharm IDE
RUN count=$(ls /tmp/pycharm-*.tar.gz | wc -l) && [ $count = 1 ] \
    && apt-get -y -qq --no-install-recommends install \
        # Without this packages, PyCharm don't start
        libxrender1 libxtst6 libxi6 libfreetype6 fontconfig \
        # Without this packages, PyCharm start, but reports that they are missing
        libatk1.0-0 libatk-bridge2.0-0 libdrm-dev libxkbcommon-dev libdbus-1-3 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr-dev libgbm1 libasound2 \
        libcups2 libatspi2.0-0 libxshmfence1 \
        # Without this packages, PyCharm start, but shows errors when running
        procps libsecret-1-0 gnome-keyring libxss1 libxext6 firefox-esr \
	#libnss3 libxext-dev libnspr4 \
    || :  # para entender porque :, ver https://stackoverflow.com/a/49348392/5076110

# Install PyCharm IDE
RUN count=$(ls /tmp/pycharm-*.tar.gz | wc -l) && [ $count = 1 ] \
    && mkdir /opt/pycharm \
    && tar xzf /tmp/pycharm-*.tar.gz -C /opt/pycharm --strip-components 1 \
    && chown -R $USR_NAME:$GRP_NAME /opt/pycharm \
    || :  # para entender porque :, ver https://stackoverflow.com/a/49348392/5076110

# Renew ARGs
ARG PYTHON_VERSION

# Pycharm espera que los paquetes python estén en dist-packages, pero están en site-packages.
# Esto es así porque python no se instaló usando apt o apt-get, y cuando esto ocurre, la carpeta
# en la que se instalan los paquetes es site-packages y no dist-packages.
RUN mkdir -p /usr/local/lib/python${PYTHON_VERSION}/dist-packages \
    && ln -s /usr/local/lib/python${PYTHON_VERSION}/site-packages/* \
             /usr/local/lib/python${PYTHON_VERSION}/dist-packages/

# Change to non-root user
USER $USR_NAME

# Set work directory
WORKDIR $APP_HOME

# Run pycharm under Tini (https://github.com/krallin/tini#using-tini)
CMD ["sh", "/opt/pycharm/bin/pycharm.sh", "-Dide.browser.jcef.enabled=false"]
# or docker run your-image /your/program ...


# Ejecución de pycharm:
#
# 1- docker volume create ereg-home
#
# 2- export DOCKER_BUILDKIT=1
#
# 3- docker build --force-rm \
#      --target pycharm \
#      --tag ereg:pycharm \
#      --build-arg USER_UID=$(stat -c "%u" .) \
#      --build-arg USER_GID=$(stat -c "%g" .) \
#      --file dockerfile .
#
# 4- docker run -ti --rm \
#      --name ereg-pycharm \
#      --env DISPLAY=$DISPLAY \
#      --volume /tmp/.X11-unix:/tmp/.X11-unix \
#      --volume $(pwd):/opt/ereg/ \
#      --volume /data/ereg:/data/ereg \
#      --volume ereg-home:/home/nonroot \
#      --detach ereg:pycharm



##############################################
## Stage 6.2: Setup and run final APP image ##
##############################################

# Create image
FROM app_builder AS final_app_image

# Become root
USER root

# Renew the ARG
ARG USR_NAME

# Setup cron to allow it run as a non root user
RUN chmod u+s $(which cron)

# Setup cron
RUN (cat /tmp/crontab.txt) | crontab -u $USR_NAME -

# Add Tini (https://github.com/krallin/tini#using-tini)
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]

# Run your program under Tini (https://github.com/krallin/tini#using-tini)
CMD [ "bash", "-c", "/startup.sh" ]
# or docker run your-image /your/program ...

# Verificar si hubo alguna falla en la ejecución del replicador
HEALTHCHECK --interval=3s --timeout=3s --retries=3 CMD bash /check-healthy.sh

# Access non-root user directory
WORKDIR /home/$USR_NAME

# Switch back to non-root user to avoid accidental container runs as root
USER $USR_NAME



####################################
## Stage 7: Set the DEFAULT image ##
####################################

FROM final_app_image



# CONSTRUIR CONTENEDOR
# export DOCKER_BUILDKIT=1
# docker build --force-rm \
#   --target final_app_image \
#   --tag ereg:latest \
#   --build-arg USER_UID=$(stat -c "%u" .) \  # ideally, the user id must be the uid of files in /data/ereg
#   --build-arg USER_GID=$(stat -c "%g" .) \  # ideally, the group id must be the gid of files in /data/ereg
#   --file dockerfile .

# CORRER OPERACIONALMENTE CON CRON
# docker run --name ereg \
#   --volume /data/ereg/descargas:/data/ereg/descargas \
#   --volume /data/ereg/generados:/data/ereg/generados \
#   --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#   --detach ereg:latest

# CORRER MANUALMENTE EN PRIMER PLANO Y BORRANDO EL CONTENEDOR AL FINALIZAR
# docker run --name ereg \
#   --volume /data/ereg/descargas:/data/ereg/descargas \
#   --volume /data/ereg/generados:/data/ereg/generados \
#   --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#   --rm ereg:latest python /opt/ereg/<script> <args>

# CORRER MANUALMENTE EN SEGUNDO PLANO Y SIN BORRAR EL CONTENEDOR AL FINALIZAR
# NO BORRAR EL CONTENEDOR AL FINALIZAR PERMITE VER LOS ERRORES (EN CASO QUE HAYA ALGUNO)
# docker run --name ereg \
#   --volume /data/ereg/descargas:/data/ereg/descargas \
#   --volume /data/ereg/generados:/data/ereg/generados \
#   --env DROP_COMBINED_FORECASTS='YES' --memory="4g" \
#   --detach ereg:latest python /opt/ereg/<script> <args>

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
