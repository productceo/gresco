mv GP-GAN GP_GAN
cd GP_GAN
pip install -r requirements/test/requirements.txt
pip install git+git://github.com/mila-udem/fuel.git@stable

mkdir datasets
cd datasets
wget http://transattr.cs.brown.edu/files/aligned_images.tar
tar -xf aligned_images.tar
rm aligned_images.tar
wget http://efrosgans.eecs.berkeley.edu/iGAN/datasets/outdoor_64.zip
unzip outdoor_64.zip
rm outdoor_64.zip

cd ..
python crop_aligned_images.py --data_root ./datasets/imageAlignedLD --result_folder ./datasets/cropped_images
python train_blending_gan.py --data_root ./datasets/cropped_images
python train_wasserstein_gan.py --data_root ./datasets/outdoor_64.hdf5
