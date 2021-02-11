#!/usr/bin/env bash
sbatch --array=1-100 inject_and_propagate.sh
