import oneflow as torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# Initialize 
prompt = "Anime portrait of natalie portman as an anime girl by stanley artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev, marc simonetti, and sakimichan, trending on artstation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="AltDiffusion-m9",
                    model_dir="./checkpoints")

model = loader.get_model()
model.eval()
model.to(device)
# Generate image
n_iter = 1 # generate n images
ddim_steps = 50 # n steps per image
predictor = Predictor(model)
predictor.predict_generate_images(prompt, n_iter=n_iter, ddim_steps=ddim_steps)
