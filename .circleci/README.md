# Docker image and CI
---
Here, we include an empty docker image based on `conda/miniconda3`. 
If you update the docker file, be sure to change the tag so that an image version change can be tracked.

# update docker image
```
name : nec4/pytorch_geometric_cpu:v*
```
For `v1.3`, after linking docker to `nec4` account on `DockerHub`, here is an example:
```
docker build  -t nec4/pytorch_geometric_cpu:v1.3 .
docker push nec4/pytorch_geometric_cpu:v1.3
```

To setup multiple arch images have a look at `https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/`.

Then do
```
docker buildx build --platform linux/amd64 -t nec4/pytorch_geometric_cpu:v1.4 --push .
```