deactivate
. env/bin/activate
python --version

rm -rf datasets/release
mkdir datasets/release

rm -rf datasets/output/*
mkdir datasets/output/generalizability
mkdir datasets/output/robustness
mkdir datasets/output/extensibility

python gresco.py

sh build_gre_dataset.sh
ls -la datasets/release
sh evaluate_multi_savqa.sh