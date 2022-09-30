PYTHON_PATH="/home/vetrov/environments/onnxchat/bin/python"
VOL_PATH="/home/vetrov"
VOL_CHECKPOINT="rugpt3large_based_on_gpt2"
DOCKER_IMAGE="gpt_bench_device"
OUTPUT_PATH="/home/vetrov/rsg_logs"
DATA_PATH="/home/vetrov/combined"
MODEL_NAME="rugpt3large_based_on_gpt2"
INPUT_SIZE=2000
BATCH_SIZE=8

for TASK in rwsd parus rcb danetqa muserc russe rucos terra lidirus
do
  mkdir -p "${OUTPUT_PATH}/${MODEL_NAME}/${TASK}"
  for INDEX in 01 02 03 04 05
    do
      ${PYTHON_PATH} /home/vetrov/MOROCCO-master/bench/main.py bench \
      ${DOCKER_IMAGE} ${DATA_PATH} ${TASK} \
      --input-size=${INPUT_SIZE} --batch-size=${BATCH_SIZE} \
      --device="cuda:12" --model-path=${VOL_CHECKPOINT} \
      --volume-path=${VOL_PATH} > "${OUTPUT_PATH}/${MODEL_NAME}/${TASK}/${INPUT_SIZE}_${BATCH_SIZE}_${INDEX}.jsonl"
  done
done