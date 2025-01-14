


MODEL=CoGCL
DADASET=instrument
config=./config/${DADASET}.yaml
params=./params/params_${DADASET}.txt
results_file=./results/${MODEL}_${DADASET}.txt

python ../run_hyper.py \
  --model=$MODEL \
  --dataset=$DADASET \
  --config_files=$config \
  --params_file=$params \
  --output_file=$results_file


