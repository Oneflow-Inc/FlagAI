# set nsight system path
nsys_path=/usr/local/cuda/nsight-systems-2022.1.3

# set params
ddim_steps=50
sed -i "s/^ddim_steps = [[:digit:]]*/ddim_steps = $ddim_steps/g " diffusion_demo.py

# save to nsys file
time=$(date "+%Y%m%d%H")
torch_version=$(python -c "import torch; print(torch.__version__)")
$nsys_path/bin/nsys profile --stats=true -o altdiffusion_${time}_${ddim_steps}steps_torch@${torch_version}.nsys-rep python3 diffusion_demo.py


# checkout changes
git checkout diffusion_demo.py
