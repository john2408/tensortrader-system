
project_folder="/mnt/d/Tensor/tensortrader-system/"
export PYTHONPATH="${PYTHONPATH}:${project_folder}"
source ~/.bashrc_conda
conda activate Tensor
cd "${project_folder}/tensortrader/tasks"
python denoised_price_return.py
python tcn_return_training.py
