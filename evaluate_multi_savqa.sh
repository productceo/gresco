deactivate
. utils/env/bin/activate

echo '\n\n Evaluating on Dataset G \n\n'
python evaluate_multi_savqa.py \
    --dataset 'datasets/release/val_g.hdf5'
    
echo '\n\n Evaluating on Dataset R \n\n'
python evaluate_multi_savqa.py \
    --dataset 'datasets/release/val_r.hdf5'   

echo '\n\n Evaluating on Dataset E \n\n'
python evaluate_multi_savqa.py \
    --dataset 'datasets/release/val_e.hdf5'