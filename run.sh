#/bin/sh

ulimit -n 2048
source ./.venv/bin/activate
echo "do you want to download models? Y/N"
read download
if [ $download == "Y" ] 
then
    let dl=1
else 
    let dl=0
fi
echo "reproducable seed (set to 0 if new test)"
read seed
echo "train on imagenet dataset (Y only if you know what you're doing)"
read imagenet
if [$imagenet == "Y"]
then 
    let im=1 
else 
    let im=0
fi
JAX_PLATFORM_NAME='cpu' PYTHON_GIL=0 HIP_VISIBLE_DEVICES=0 python main.py 225 1 resnet50 IMAGENET1K_V1 CrossEntropyLoss Adam $dl $seed $im 
