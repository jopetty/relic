#!/bin/bash

srun --gres=gpu:2 --cpus-per-task=2 --mem=32G --time=02:00:00 --pty /bin/bash
