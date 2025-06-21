# Ficheiro: main.py
# (Sem alterações na lógica da IA, apenas na forma como o servidor é iniciado para compatibilidade com Heroku)

from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import torch
import os
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# --- Verificação de Hardware e Aviso ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("---------------------------------------------------------")
    print("AVISO: Nenhuma GPU detetada.")
    print("O modelo de IA irá funcionar no CPU, o que será MUITO LENTO.")
    print("O primeiro redraw pode demorar vários minutos.")
    print("---------------------------------------------------------")

# --- Configuração do Modelo ---
pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# --- Configuração do Servidor Flask ---
app = Flask(__name__)

# Rota principal que exibe a nossa interface
@app.route('/')
def index():
    return render_template('index.html')

# Rota para o processo de redraw
@app.route('/redraw', methods=['POST'])
def redraw():
    data = request.json
    
    if not data or 'image' not in data or 'mask' not in data:
        return jsonify({'error': 'Dados da imagem ou máscara em falta'}), 400

    try:
        # Descodifica as imagens recebidas do frontend
        image_b64 = data['image'].split(',')[1]
        mask_b64 = data['mask'].split(',')[1]

        init_image_bytes = base64.b64decode(image_b64)
        mask_image_bytes = base64.b64decode(mask_b64)
        
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_image_bytes)).convert("RGB")

        # Define o prompt para a IA
        prompt = "um fundo detalhado e fotorrealista que se mistura perfeitamente com a área circundante, estilo de arte de alta qualidade"

        # Executa o pipeline de inpainting do Diffusers
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        
        # Converte a imagem resultante de volta para base64 para enviar ao frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({'image': 'data:image/png;base64,' + img_str})

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # O Heroku define a porta através de uma variável de ambiente.
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
```txt
