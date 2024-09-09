


MODEL=CoGCL
DADASET=office
config=./config/${DADASET}.yaml

python ../run_recbole_gnn.py \
  --model=$MODEL \
  --dataset=$DADASET \
  --config_files=$config



