# Use Jupyter's minimal notebook as the base image
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

USER root

# Install system-level dependencies
RUN apt-get update && apt-get install -y texlive-full make chromium-browser && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Conda lockfile
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Install Quarto
RUN wget https://quarto.org/download/latest/quarto-linux-amd64.deb && \
    dpkg -i quarto-linux-amd64.deb && \
    rm quarto-linux-amd64.deb

# Install Quarto tools and fix YAML path
RUN quarto install tools || echo "Quarto tools installation failed" && \
    mkdir -p /opt/conda/share/editor/tools/yaml/ && \
    ln -s /opt/conda/share/quarto/editor/tools/yaml/yaml-intelligence-resources.json \
        /opt/conda/share/editor/tools/yaml/yaml-intelligence-resources.json

# Install Deno and fix path
ENV DENO_INSTALL=/opt/deno
RUN curl -fsSL https://deno.land/x/install/install.sh | sh && \
    mkdir -p /opt/conda/bin/tools/x86_64 && \
    cp /opt/deno/bin/deno /opt/conda/bin/tools/x86_64/deno && \
    chmod +x /opt/conda/bin/tools/x86_64/deno && \
    ln -s /opt/deno/bin/deno /usr/local/bin/deno

# Update Conda environment
RUN mamba update --quiet --file /tmp/conda-linux-64.lock && \
    mamba clean --all -y -f

# Copy project files
COPY scripts /home/jovyan/scripts
COPY Makefile /home/jovyan/Makefile
COPY reports /home/jovyan/reports
COPY results /home/jovyan/results
COPY src /home/jovyan/src

# Set permissions
RUN fix-permissions "/home/jovyan" && chmod -R u+w /home/jovyan

# Install Python dependencies
RUN pip install pyyaml==6.0.2

# Set the working directory
WORKDIR /home/jovyan

USER ${NB_UID}

CMD ["make", "all"]
