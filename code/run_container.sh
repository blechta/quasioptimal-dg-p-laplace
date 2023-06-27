#!/bin/bash
docker run --rm -ti -v $PWD:/home/firedrake/shared -v $PWD/.bash_history:/home/firedrake/.bash_history -v $PWD/.python_history:/home/firedrake/.python_history firedrakeproject/firedrake
