cd edge_connect
pip install -r requirements.txt

mkdir datasets
cd datasets
wget http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
wget http://data.csail.mit.edu/places/places365/val_256.tar
wget http://data.csail.mit.edu/places/places365/test_256.tar
tar -xf train_256_places365standard.tar
tar -xf val_256.tar
tar -xf test_256.tar
rm *.tar

wget https://www.dropbox.com/s/qp8cxqttta4zi70/irregular_mask.zip
wget http://masc.cs.gmu.edu/wiki/uploads/partialconv/mask.zip
unzip irregular_mask.zip
unzip mask.zip
rm *.zip

cd ..
python ./scripts/flist.py --path ./datasets/data_256 --output ./datasets/places_train.flist
python ./scripts/flist.py --path ./datasets/val_256 --output ./datasets/places_val.flist
python ./scripts/flist.py --path ./datasets/test_256 --output ./datasets/places_test.flist
python ./scripts/flist.py --path ./datasets/irregular_mask/disocclusion_img_mask --output ./datasets/masks_train.flist
python ./scripts/flist.py --path ./datasets/irregular_mask/disocclusion_img_mask --output ./datasets/masks_val.flist
python ./scripts/flist.py --path ./datasets/mask/testing_mask_dataset --output ./datasets/masks_test.flist

bash ./scripts/download_model.sh

python ./train.py --checkpoints ./checkpoints/gresco --model 1
python ./train.py --checkpoints ./checkpoints/gresco --model 2
python ./train.py --checkpoints ./checkpoints/gresco --model 3
python ./test.py \
       --checkpoints ./checkpoints/gresco \
       --input ./examples/places2/images \
       --mask ./examples/places2/masks \
       --output ./checkpoints/results
