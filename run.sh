#/bin/sh
source ./.venv/bin/activate
trap "rm -rf ./datasets_masked" SIGINT
time python main.py 225 1 vgg11 IMAGENET1K_V1 CrossEntropyLoss Adam
