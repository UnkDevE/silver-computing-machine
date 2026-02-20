#/bin/sh
ulimit -n 2048
source ./.venv/bin/activate
echo "reproducable seed (set to 0 if new test)"
read seed
HIP_VISIBLE_DEVICES=0 python main.py 225 1 vgg11 IMAGENET1K_V1 CrossEntropyLoss Adam 0 $seed 
