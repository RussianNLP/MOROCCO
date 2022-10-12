PYTHON_PATH="/path/to/python"
PLOTS_PATH="/path/to/plots"
STATS_PATH="/path/to/stats"
LOGS_PATH="/path/to/logs"

for MODEL_NAME in "model_name"
    do
        for TASK in rwsd parus rcb danetqa muserc russe rucos terra lidirus
            do
                mkdir -p "${PLOTS_PATH}/${MODEL_NAME}"
                mkdir -p "${STATS_PATH}/${MODEL_NAME}"

                ${PYTHON_PATH} bench/main.py plot ${LOGS_PATH}/${MODEL_NAME}/${TASK}/*.jsonl  "${PLOTS_PATH}/${MODEL_NAME}/${TASK}.png"
                ${PYTHON_PATH} bench/main.py stats ${LOGS_PATH}/${MODEL_NAME}/${TASK}/*.jsonl  >> "${STATS_PATH}/${MODEL_NAME}/${TASK}.jsonl"
        done
done