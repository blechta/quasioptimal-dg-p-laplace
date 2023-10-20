#!/bin/bash

IMAGE=firedrakeproject/firedrake:2023-10
CONTAINER=quasioptimal

udocker pull $IMAGE
udocker create --name=$CONTAINER $IMAGE
udocker setup --execmode=R1 $CONTAINER
