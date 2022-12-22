
project_folder="/mnt/d/Tensor/tensortrader-system/tensortrader"
export PYTHONPATH="${PYTHONPATH}:${project_folder}"
source ~/.bashrc_conda
conda activate Tensor
cd "${project_folder}/tasks"
python price_return_calculation.py