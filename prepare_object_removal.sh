cd edge-connect
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

cd ..
python ./scripts/flist.py --path ./datasets/data_256 --output ./datasets/places_train.flist
python ./scripts/flist.py --path ./datasets/val_256 --output ./datasets/places_val.flist
python ./scripts/flist.py --path ./datasets/test_256 --output ./datasets/places_test.flist

bash ./scripts/download_model.sh

python ./train.py --checkpoints ./checkpoints --model 1
python ./test.py \
       --checkpoints ./checkpoints/places2 \
       --input ./examples/places2/images \
       --mask ./examples/places2/masks \
       --output ./checkpoints/results
