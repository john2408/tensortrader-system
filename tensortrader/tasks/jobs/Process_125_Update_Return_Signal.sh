
project_folder="/mnt/d/Tensor/tensortrader-system/"
export PYTHONPATH="${PYTHONPATH}:${project_folder}"
source ~/.bashrc_conda
conda activate Tensor
cd "${project_folder}/tensortrader/tasks"
python data_update.py
python price_return_calculation.py
python trading_signals.py
