#!/bin/bash

# isort --check --sl -c src/keras_explainable
# if ! [ $? -eq 0 ]
# then
#   echo "Please run \"sh shell/format.sh\" to format the code."
#   exit 1
# fi
# echo "no issues with isort"

flake8 src/keras_explainable
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
echo "no issues with flake8"
black --check --line-length 90 src/keras_explainable
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
echo "no issues with black"
echo "linting success!"
