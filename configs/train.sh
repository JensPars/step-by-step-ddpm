#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ddpm
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u jens@parslov.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o bjob-out/gpu_%J.out
#BSUB -e bjob-out/gpu_%J.err
# -- end of LSF options --
cd /zhome/ca/9/146686/step-by-step-ddpm
/work3/s194649/miniconda3/envs/ddpm/bin/python /zhome/ca/9/146686/step-by-step-ddpm/ddpm.py