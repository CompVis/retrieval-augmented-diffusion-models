mkdir -p database
mkdir -p database/ImageNet
wget -P database/ImageNet https://ommer-lab.com/files/rdm/database/ImageNet/1281200x512-part_1.npz
mkdir -p database/OpenImages

wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_2.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_3.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_4.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_5.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_7.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_8.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_9.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/1999998x512-part_10.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/2000097x512-part_1.npz
wget -P database/OpenImages https://ommer-lab.com/files/rdm/database/OpenImages/2927826x512-part_6.npz

mkdir -p nn_memory
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/in_imagenet.p
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/oi_ffhq.p
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/oi_imagenet-animals.p
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/oi_imagenet-dogs.p
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/oi_imagenet-mammals.p
wget -P nn_memory https://ommer-lab.com/files/rdm/nn_memory/oi_imagenet.p
