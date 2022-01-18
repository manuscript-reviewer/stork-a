FROM python:3.7.10-stretch
#===============================#
# Docker Image Configuration	#
#===============================#
LABEL org.opencontainers.image.source='https://github.com/eipm/stork-a' \
    vendor='Englander Institute for Precision Medicine' \
    description='STORK-A' \
    maintainer='paz2010@med.cornell.edu' \
    base_image='python' \
    base_image_version='3.7.10-stretch'

ENV APP_NAME='stork-a' \
    TZ='US/Eastern'

#===================================#
# Install Prerequisites             #
#===================================#
COPY requirements.txt /${APP_NAME}/requirements.txt
RUN pip install -r /${APP_NAME}/requirements.txt
#===================================#
# Copy Files and set work directory	#
#===================================#
COPY src /${APP_NAME}/src/
WORKDIR /${APP_NAME}
#===================================#
# Startup                           #
#===================================#
EXPOSE 80
VOLUME uploads

HEALTHCHECK --interval=30s --timeout=30s --retries=3 \
    CMD curl -f -k http://0.0.0.0/api/healthcheck || exit 1

CMD python3 /${APP_NAME}/src/main.py