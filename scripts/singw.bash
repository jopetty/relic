#!/bin/bash

singularity exec --nv --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash
