import requests

from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.title("Detecção de Componentes Cloud e Relatório STRIDE")
uploaded_file = st.file_uploader("Faça upload de uma imagem de arquitetura cloud", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_path = "uploaded_image.png"

    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())
    img = Image.open(img_path)
    st.image(img, caption="Imagem enviada")

    # Carrega o modelo YOLO
    model = YOLO("./runs/detect/train2/weights/best.pt")
    results = model(img_path)

    # Carrega as classes
    with open("./yolo/classes.txt", "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    # Salva as imagens com detecções
    result_img_path = "resultado_yolo.png"
    results[0].save(filename=result_img_path)
    st.image(result_img_path, caption="Resultado YOLO")

    # Encontra os componentes na imagem
    detected_components = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            component_name = classes[class_id]
            detected_components.append(component_name)
    detected_components = list(dict.fromkeys(detected_components))
    st.write("**Componentes detectados:**", detected_components)

    # Prompt para gerar o relatório STRIDE
    prompt = (
        "Gere um Relatório de Modelagem de Ameaças, baseado na metodologia STRIDE,"
        + " para os seguintes componentes de nuvem identificados: "
        + ", ".join(detected_components)
        + ". O relatório deve detalhar todas as ameaças STRIDE para cada componente tendo a seguinte ordem"
        + ": (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)"
        + ", não é preciso explicar o que é STRIDE"
        + ", se o componente identificado não for um componente de nuvem, ignore-o, por exemplo: REST, SOAP, usuario"
        + ", o resultado deve ser apresentado na estrutura Markdown com o componente no titulo e"
        + " uma tabela, com as seguintes colunas: Ameaça, Mitigação"
    )

    with st.spinner("Gerando relatório STRIDE com LLM..."):
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(ollama_url, json=payload)
        result = response.json()
        stride_report = result["response"]

    st.markdown("## Relatório STRIDE")
    st.markdown(stride_report)

    st.download_button(
        label="Baixar relatório STRIDE em Markdown",
        data=stride_report,
        file_name="relatorio_stride.md",
        mime="text/markdown"
    )