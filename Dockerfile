# Use the recommended base image
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Switch to root user to install additional dependencies
USER root

# Install dependencies from the environment.yml file
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean --all --yes

# Set default environment
ENV PATH /opt/conda/envs/environment.yml/bin:$PATH
RUN echo "source activate <env-name>" > ~/.bashrc

# Return to the notebook user
USER jovyan
