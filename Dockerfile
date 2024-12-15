# FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# COPY conda-linux-64.lock /tmp/conda-linux-64.lock



# RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
#     && mamba clean --all -y -f \
#     && fix-permissions "${CONDA_DIR}" \
#     && fix-permissions "/home/${NB_USER}"

# RUN pip install pyyaml==6.0.2
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy the Conda lock file
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Create the 'loan_risk522' environment
RUN mamba create --quiet --name loan_risk522 --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Add the environment to Jupyter as a kernel
RUN /opt/conda/envs/loan_risk522/bin/python -m ipykernel install --user --name loan_risk522 --display-name "Python (loan_risk522)"

# Ensure permissions for the environment
RUN fix-permissions "/opt/conda/envs/loan_risk522"

# Activate the 'loan_risk522' environment by default in Jupyter
ENV CONDA_DEFAULT_ENV=loan_risk522
ENV PATH="/opt/conda/envs/loan_risk522/bin:$PATH"
