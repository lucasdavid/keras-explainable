#!/bin/bash
isort --sl src/keras_explainable
black --line-length 80 src/keras_explainable
flake8 src/keras_explainable
