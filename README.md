#Fermat

Fermat is a CUDA physically based research renderer designed and developed by Jacopo Pantaleoni at NVIDIA.
Its purpose is mostly educational: it is primarily intended to teach how to write rendering algorithms,
ranging from simple forward path tracing, to bidirectional path tracing, to Metropolis light transport with many
of its variants, and do so on massively parallel hardware.

The choice of CUDA C++ has been made for various reasons: the first was to allow the highest level of expression and
flexibility in terms of programmability (for example, with template based meta-programming); the second was perhaps
historical: when Fermat's development was started, other ray tracing platforms like Microsoft DXR did not yet exist.
The ray tracing core employed by Fermat is OptiX prime - though it is highly likely that future versions will switch
to OptiX, or even offer a DXR backend.

Fermat is built on top of another library co-developed for this project: CUGAR - CUDA Graphics AcceleratoR.
This is a template library of low-level graphics tools, including algorithms for BVH, Kd-tree and octree construction,
sphericals harmonics, sampling, and so on and so on.
While packaged together, CUGAR can be thought of a separate educational project by itself.
More information can be found in the relevant Doxygen documentation.

DEPENDENCIES:

Fermat has the following dependencies:

 - cub         : contained in the package
 - freeglut    : contained in the package
 - CUDA 8.0    : not contained - it should be separately downloaded and installed on the system
 - OptiX 4.0.1 : not contained - it should be separately downloaded and copied in the folder contrib/OptiX

Its distribution also contains a set of adapted models originally taken from Benedikt Bitterli's rendering resources:
https://benedikt-bitterli.me/resources.
 
COMPILATION:

Once all dependencies are sorted out, the Visual Studio 2015 solution file vs/fermat/fermat.sln can be opened
and the project can be compiled.

USE:

After compilation, you can launch Fermat's path tracer with the following command line:
-view -pt -r 1600 900 -i ../../models/bathroom2/bathroom.obj -c ../../models/bathroom2/camera2.txt
