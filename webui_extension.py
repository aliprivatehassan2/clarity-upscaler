import os, torch, gradio as gr
from modules import scripts, script_callbacks
from diffusers import StableDiffusionPipeline

class ClarityScript(scripts.Script):
    def title(self):
        return "Clarity Upscaler"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt for Clarity", value="ultra detailed")
            run_button = gr.Button("Run Clarity")
        return [prompt, run_button]

    def run(self, p, prompt, run_button):
        if not prompt or not run_button:
            return p
        in_path = os.path.join(p.outpath_samples, "clarity_in.png")
        p.init_images[0].save(in_path)
        pipe = StableDiffusionPipeline.from_pretrained(
            "philz1337x/clarity-upscaler",
            torch_dtype=torch.float16
        ).to("cuda")
        result = pipe(prompt=prompt, image=in_path).images[0]
        p.init_images = [result]
        return p

script_callbacks.on_ui_tabs(lambda: (ClarityScript(),))
