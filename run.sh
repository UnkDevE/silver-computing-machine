#/bin/sh
source ./.venv/bin/activate
python main.py 225 1 vgg11 IMAGENET1K_V1 CrossEntropyLoss Adam
