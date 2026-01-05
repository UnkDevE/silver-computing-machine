#/bin/sh
source ./.venv/bin/activate
echo "reproducable seed (set to 0 if new test)"
read seed
python main.py 225 1 vgg11 IMAGENET1K_V1 CrossEntropyLoss Adam $seed 
