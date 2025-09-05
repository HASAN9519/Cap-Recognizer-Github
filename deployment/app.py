# -- NB: always make sure python version in which model was trained should match local or hugging face environment otherwise model won't work

# -- fix model loading issues when a .pkl file saved on Linux (using PosixPath) is loaded on Windows. 
# -- But it should be done before importing Fastai or loading the model. 
# -- comment the following when uploading to huggingface

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath 

# ---------------------             original demo code

# import gradio as gr
# import sys
 
# def greet(name):
#     return "Hello " + name + "!!" + "Running Python version:" + sys.version

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()

# --------------------               main code

from fastai.vision.all import *
import gradio as gr

model = load_learner('models/cap-recognizer-v0.pkl')

cap_labels = [
    "balaclava cap",
    "baseball cap",
    "beanie cap",
    "boater hat",
    "bowler hat",
    "bucket hat",
    "cowboy hat",
    "flat cap",
    "kepi cap",
    "taqiyah cap",
    "top hat",
    "turban cap",
    "visor cap"
]

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(cap_labels, map(float, probs)))

image = gr.Image()
label = gr.Label()
examples = ['test_images/test-1.jpg',
            'test_images/test-2.jpg',
            'test_images/test-3.jpg',
            'test_images/test-4.jpg',
            'test_images/test-5.jpg',
            'test_images/test-6.jpg',
            'test_images/test-7.jpg',
            'test_images/test-8.jpg',
            'test_images/test-9.jpg',
            'test_images/test-10.jpg',
            'test_images/test-11.jpg',
            'test_images/test-12.jpg',
            'test_images/test-13.jpg'
        ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)

iface.launch(inline=False)