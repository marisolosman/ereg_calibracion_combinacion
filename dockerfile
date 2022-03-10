
#############################
## Install python packages ##
#############################

# Create image
FROM python:3.10-slim-bullseye AS py_builder

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
        PyYAML \
        Cartopy==0.19.0.post1



########################
## CREATE FINAL IMAGE ##
########################

# Create image
FROM python:3.10-slim-bullseye AS final_image

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive

# Install OS packages
RUN apt-get -y -qq update &&\
    apt-get -y -qq --no-install-recommends install \
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
RUN mkdir -p /data



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
RUN chown -R $NON_ROOT_UID:$NON_ROOT_GID /data
RUN chown -R $NON_ROOT_UID:$NON_ROOT_GID /opt/ereg

#
RUN sed -i "s/^group_for_files/# group_for_files/g" /opt/ereg/config.yaml

# Setup cron for run twice a month -- Download files from IRIDL
RUN (echo "* * 15,16 * * cd /opt/ereg && python download_inputs.py --download operational >> /proc/1/fd/1 2>> /proc/1/fd/1") | crontab -u $NON_ROOT_USR -
# Setup cron for run once a month -- Run operational forecast
RUN (crontab -u $NON_ROOT_USR -l; echo "* * 17 * * cd /opt/ereg && python run_operational_forecast.py >> /proc/1/fd/1 2>> /proc/1/fd/1") | crontab -u $NON_ROOT_USR -

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
# docker build -f dockerfile \
#        --build-arg ROOT_PWD=nonroot \
#        --build-arg NON_ROOT_PWD=nonroot \
#        --build-arg NON_ROOT_UID=$(stat -c "%u" .) \  # ideally, the user id must be the uid of files in /data
#        --build-arg NON_ROOT_GID=$(stat -c "%g" .) \  # ideally, the group id must be the gid of files in /data
#        -t ereg .

# CORRER OPERACIONALMENTE CON CRON
# docker run --name ereg --rm \
#        --volume /data:/data \
#        --env DROP_COMBINED_FORECASTS='YES' \
#        --detach ereg:latest

# CORRER MANUALMENTE
# docker run --name ereg --rm \
#        --volume /data:/data \
#        --env DROP_COMBINED_FORECASTS='YES' \
#        ereg:latest python /opt/ereg/download_inputs.py --download operational
