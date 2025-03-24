# $\varphi$-FEM for particulate flows and Stokes

This repository contains the code used in "$\varphi$-FEM: an optimally convergent and easily implementable immersed boundary method for particulate flows and Stokes equations" M. Duprez, V. Lleras and A. Lozinski. *ESAIM: M2AN 57 (2023), 1111-1142* [https://doi.org/10.1051/m2an/2023010](https://doi.org/10.1051/m2an/2023010).

## This repository is for reproducibility purposes only

It is "frozen in time" and not maintained.
To use our latest $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Usage

### Prerequisites

- [Git](https://git-scm.com/),
- [Docker](https://www.docker.com/)/[podman](https://podman.io/).

The image is based on the legacy FEniCS image: quay.io/fenicsproject/stable:2019.1.0.r3 and contains in addition the (legacy) [`multiphenics`](https://github.com/multiphenics/multiphenics/tree/7b23c85c070a092775666c7dad84c8d6471c0b0c) python library.

### Install the image and launch the container

1) Clone this repository in a dedicated directory:
   
   ```bash
   mkdir stokes-phifem/
   git clone https://github.com/PhiFEM/Particulate-flows-Stokes.git stokes-phifem
   ```

2) Download the images from the docker.io registry, in the main directory:
   
   ```bash
   export CONTAINER_ENGINE=docker
   cd stokes-phifem
   sudo -E bash pull-image.sh
   ```

3) Launch the container:

   ```bash
   sudo -E bash run-image.sh
   ```

### Example of usage

From the main directory `stokes-phifem`, launch e.g. the Stokes test case:

```bash
python3 phiFEM_stokes.py
```

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

[Michel Duprez](https://michelduprez.fr/), Inria Nancy Grand-Est  
[Vanessa Lleras](https://vanessalleras.wixsite.com/lleras), Université de Montpellier  
[Alexei Lozinski](https://orcid.org/0000-0003-0745-0365), Université de Franche-Comté  
