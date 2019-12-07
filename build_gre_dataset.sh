deactivate
. utils/env/bin/activate

echo '\n\n Building Dataset G \n\n'
python build_gre_dataset.py \
    --train-annotations 'datasets/output/generalizability/train_annotations.json' \
    --val-annotations 'datasets/output/generalizability/val_annotations.json' \
    --train-questions 'datasets/output/generalizability/train_questions.json' \
    --val-questions 'datasets/output/generalizability/val_questions.json' \
    --train-images 'datasets/output/generalizability' \
    --val-images 'datasets/output/generalizability' \
    --train-output 'datasets/release/train_g.hdf5' \
    --val-output 'datasets/release/val_g.hdf5'
    
echo '\n\n Building Dataset R \n\n'
python build_gre_dataset.py \
    --train-annotations 'datasets/output/robustness/train_annotations.json' \
    --val-annotations 'datasets/output/robustness/val_annotations.json' \
    --train-questions 'datasets/output/robustness/train_questions.json' \
    --val-questions 'datasets/output/robustness/val_questions.json' \
    --train-images 'datasets/output/robustness' \
    --val-images 'datasets/output/robustness' \
    --train-output 'datasets/release/train_r.hdf5' \
    --val-output 'datasets/release/val_r.hdf5'
    
echo '\n\n Building Dataset E \n\n'
python build_gre_dataset.py \
    --train-annotations 'datasets/output/extensibility/train_annotations.json' \
    --val-annotations 'datasets/output/extensibility/val_annotations.json' \
    --train-questions 'datasets/output/extensibility/train_questions.json' \
    --val-questions 'datasets/output/extensibility/val_questions.json' \
    --train-images 'datasets/output/extensibility' \
    --val-images 'datasets/output/extensibility' \
    --train-output 'datasets/release/train_e.hdf5' \
    --val-output 'datasets/release/val_e.hdf5'