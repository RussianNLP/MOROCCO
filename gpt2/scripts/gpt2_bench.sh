PYTHON_PATH="/path/to/python"
VOL_PATH="/path/to/volume"
VOL_CHECKPOINT="path/to/model_directory_in_volume"
DOCKER_IMAGE="gpt2_morocco"
OUTPUT_PATH="/path/to/logs"
DATA_PATH="/path/to/rsg_data"
MODEL_NAME="model_name"
DEVICE="cuda:0"
INPUT_SIZE=2000
BATCH_SIZE=8

for TASK in rwsd parus rcb danetqa muserc russe rucos terra lidirus
do
  mkdir -p "${OUTPUT_PATH}/${MODEL_NAME}/${TASK}"
  for INDEX in 01 02 03 04 05
    do
      ${PYTHON_PATH} bench/main.py bench \
      ${DOCKER_IMAGE} ${DATA_PATH} ${TASK} \
      --input-size=${INPUT_SIZE} --batch-size=${BATCH_SIZE} \
      --device=${DEVICE} --model-path=${VOL_CHECKPOINT} \
      --volume-path=${VOL_PATH} > "${OUTPUT_PATH}/${MODEL_NAME}/${TASK}/${INPUT_SIZE}_${BATCH_SIZE}_${INDEX}.jsonl"
  done
done