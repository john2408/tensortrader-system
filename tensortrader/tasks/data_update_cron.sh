
project_folder="/mnt/d/Tensor/tensortrader/tensortrader"
export PYTHONPATH="${PYTHONPATH}:${project_folder}"
source ~/.bashrc_conda
conda activate Tensor
cd "${project_folder}/tasks"
python data_update.py