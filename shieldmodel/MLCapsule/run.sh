echo "** Running MLcapsule baseline, DO NOT interrupt**"
graphene-direct vgg > ~/atc22-artifact/SOTER/script/data/mlcvgg.txt
echo "Finished vgg MLcapsule!"
graphene-direct alexnet > ~/atc22-artifact/SOTER/script/data/mlcalexnet.txt
echo "Finished alexnet MLcapsule!"
graphene-direct resnet > ~/atc22-artifact/SOTER/script/data/mlcresnet.txt
echo "Finished resnet MLcapsule!"
graphene-direct densenet > ~/atc22-artifact/SOTER/script/data/mlcdensenet.txt
echo "Finished densenet MLcapsule!"
graphene-direct mlp > ~/atc22-artifact/SOTER/script/data/mlcmlp.txt
echo "Finished mlp MLcapsule!"
graphene-direct transformer 1024 > ~/atc22-artifact/SOTER/script/data/mlctrans.txt
echo "Finished transformer MLcapsule!"