PYTHON_PATH="/home/vetrov/environments/onnxchat/bin/python"
PLOTS_PATH="/home/vetrov/rsg_plots"
STATS_PATH="/home/vetrov/rsg_stats"
LOGS_PATH="/home/vetrov/rsg_logs"

for MODEL_NAME in "rugpt3large_based_on_gpt2"
    do
        for TASK in rwsd parus rcb danetqa muserc russe rucos terra lidirus
            do
                mkdir -p "${PLOTS_PATH}/${MODEL_NAME}"
                mkdir -p "${STATS_PATH}/${MODEL_NAME}"

                ${PYTHON_PATH} /home/vetrov/MOROCCO-master/bench/main.py plot ${LOGS_PATH}/${MODEL_NAME}/${TASK}/*.jsonl  "${PLOTS_PATH}/${MODEL_NAME}/${TASK}.png"
                ${PYTHON_PATH} /home/vetrov/MOROCCO-master/bench/main.py stats ${LOGS_PATH}/${MODEL_NAME}/${TASK}/*.jsonl  >> "${STATS_PATH}/${MODEL_NAME}/${TASK}.jsonl"
        done
done