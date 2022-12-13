# set nsight system path
nsys_path=/usr/local/cuda/nsight-systems-2022.1.3

# set params
ddim_steps=50
sed -i "s/^ddim_steps = [[:digit:]]*/ddim_steps = $ddim_steps/g " diffusion_demo.py

# save to nsys file
time=$(date "+%Y%m%d%H")
git_commit=$(python3 -m oneflow --doctor | grep "git_commit: ")
$nsys_path/bin/nsys profile --stats=true -o altdiffusion_${time}_${ddim_steps}steps_flow@${git_commit#*: }.nsys-rep python3 diffusion_demo.py


# checkout changes
git checkout diffusion_demo.py
