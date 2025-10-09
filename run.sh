#/bin/sh
source ./.venv/bin/activate
time python main.py 225 1 vgg11 IMAGENET1K_V1 CrossEntropyLoss Adam
rm -rf ./datasets_masked
