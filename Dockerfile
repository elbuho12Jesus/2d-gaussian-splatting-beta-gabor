# Utilizamos una imagen base de NVIDIA con CUDA Toolkit para poder compilar las extensiones
# CUDA 12.8: primera versión con soporte para Blackwell (sm_120, RTX PRO 6000)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Evitar prompts interactivos durante la instalación de paquetes de sistema
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema requeridas por OpenCV, ffmpeg y Conda
RUN apt-get update && apt-get install -y \
    wget git build-essential ninja-build \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Configurar el directorio de trabajo
WORKDIR /workspace

# Copiar TODO el proyecto al contenedor (incluyendo environment.yml y submodules)
# Esto es necesario porque el environment.yml instala pip packages desde rutas locales
COPY . /workspace/

# (Asegúrate de tener los comandos del Paso anterior aquí arriba)

# Definir variables de entorno para la compilación de CUDA
ENV CUDA_HOME=/usr/local/cuda
# Compilar para Ampere, Ada, Hopper y Blackwell (sm_120). 12.0 = RTX PRO 6000 Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Aceptar los Términos de Servicio de Anaconda para uso desatendido
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Crear el entorno de conda base (sin los submódulos)
RUN conda env create -f environment.yml

# Activar el entorno e instalar las extensiones de CUDA manualmente
# Usar la ruta absoluta del pip del entorno
RUN /root/miniconda3/envs/surfel_splatting_modificate/bin/pip install ./submodules/simple-knn
RUN /root/miniconda3/envs/surfel_splatting_modificate/bin/pip install ./submodules/diff-surfel-rasterization

# Configurar el contenedor para que active automáticamente el entorno
RUN echo "conda activate surfel_splatting_modificate" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PATH=/root/miniconda3/envs/surfel_splatting_modificate/bin:$PATH

CMD ["/bin/bash"]
