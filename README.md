# phi-FEM-an-optimally-convergent-and-easily-implementable-method-for-particulate-flows-and-Stokes

This repository corresponds to the codes of the publication : https://hal.archives-ouvertes.fr/hal-03588715v1

particulate flow equations : phiFEM_particulate_flow.py
stokes equations : phiFEM_stokes.py

To run these codes, you have to install FEniCS and multiphenics, which can be done thanks to the following docker image :
sudo docker run --rm -ti -m=8g -v $(pwd):/home/fenics/shared:z multiphenics/multiphenics

