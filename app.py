import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import cv2
import pickle
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title = "Potato Tubers Disease Detection",
    layout = "wide"
)
st.markdown("""
<style>
.main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
                .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #C8E6C9;
        border-left: 5px solid #2E7D32;
    }
    .disease {
        background-color: #FFCDD2;
        border-left: 5px solid #C62828;
    }
    .info-text {
        font-size: 1rem;
        color: #333;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Potato Disease Detector</h1>',unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Potato Disease Detection</p>',unsafe_allow_html=True)

# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/000000/potato.png", width=100)

@st.cache_resource
def load_models():
    try:
        model = load_model('models/potato_disease_model.h5')
        with open('models/class_names.pkl','rb') as f:
            class_indices = pickle.load(f)
        class_names = list(class_indices.keys())
        return model, class_names, class_indices
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None,None,None
    
def gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap*alpha+img
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img

def predict_disease(image_file, model, class_names):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img_resized = img.resize((224,224))

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))

    predictions = model.predict({'input_layer': img_array}, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100

    top3_idx = np.argsort(predictions[0])[-3:]
    top3_predictions = [(class_names[i], predictions[0][i] * 100) for i in top3_idx]

    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name:
            heatmap = gradcam_heatmap(img_array, model, last_conv_layer_name)
            img_cv2 = np.array(img_resized)
            superimposed_img = superimpose_heatmap(heatmap, img_cv2)
        else:
            superimposed_img = None
    except:
        superimposed_img = None
    
    return{
        'original_image' : img_resized,
        'predicted_class' : predicted_class,
        'confidence' : confidence,
        'top_3' : top3_predictions,
        'heatmap_image' : superimposed_img
    }

def get_disease_info(disease_name):
    """Get information about the disease"""
    
    disease_info = {
        'Healthy': {
            'description': 'The potato appears healthy with no visible signs of disease or damage.',
            'symptoms': 'Smooth skin, no discoloration, firm texture, no unusual spots or rot.',
            'treatment': 'No treatment needed. Maintain proper storage conditions (cool, dark, well-ventilated).',
            'prevention': 'Store in a cool, dark place with good air circulation. Check regularly for any signs of spoilage.',
            'color': '#2E7D32',
        },
        'Blackspot_Bruising': {
            'description': 'Blackspot bruising appears as dark, sunken areas on the potato tuber surface, caused by mechanical damage during harvesting or handling.',
            'symptoms': 'Dark blue-black spots beneath the skin, irregular shape, sunken areas, no soft rot.',
            'treatment': 'Affected areas can be cut away before consumption if the rest of the potato is firm.',
            'prevention': 'Careful handling during harvest and transport. Avoid dropping potatoes. Ensure proper soil conditions during growth.',
            'color': '#F57C00',
        },
        'Soft_Rot': {
            'description': 'Soft rot is a bacterial disease causing rapid decay of potato tubers, leading to soft, mushy tissue with characteristic foul odor.',
            'symptoms': 'Water-soaked areas, soft and mushy texture, foul smell, cream to tan colored rot, tissue breakdown.',
            'treatment': 'Discard affected potatoes immediately to prevent spread. Do not consume.',
            'prevention': 'Avoid wounding potatoes during harvest. Ensure good air circulation in storage. Keep storage areas dry and clean.',
            'color': '#D32F2F',
        },
        'Brown_Rot': {
            'description': 'Brown rot is a serious bacterial disease that causes wilting of plants and rotting of tubers, with characteristic brown discoloration.',
            'symptoms': 'Brown discoloration of vascular ring, bacterial ooze from eyes, soil sticking to tubers, eventual complete rot.',
            'treatment': 'No effective treatment. Affected tubers should be destroyed. Practice crop rotation.',
            'prevention': 'Use certified disease-free seed potatoes. Practice long crop rotation (4-5 years). Avoid overhead irrigation.',
            'color': '#8B4513',
        },
        'Dry_Rot': {
            'description': 'Dry rot is a fungal disease that causes shriveled, mummified tubers with concentric rings of fungal growth.',
            'symptoms': 'Shriveled skin, concentric rings of fungal growth, internal cavities with fungal mycelium, dry and firm decay.',
            'treatment': 'Remove affected tubers. Do not store damaged potatoes as they are more susceptible.',
            'prevention': 'Avoid wounding potatoes during harvest. Store at proper temperature (40-45°F). Maintain good ventilation.',
            'color': '#795548',
        }
    }
    
    return disease_info.get(disease_name, {
        'description': 'Information not available.',
        'symptoms': 'N/A',
        'treatment': 'N/A',
        'prevention': 'N/A',
        'color': '#757575',
    })

def main():
    model, class_names, class_indices = load_models()
    if model is None:
        st.error("Failed to load model")
        return 
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a potato tuber"
        )
        if uploaded_file is not None:
            st.image(uploaded_file,caption="Uploaded Image", width='stretch')

    with col2:
        if uploaded_file is not None:
            st.markdown("Results")

            with st.spinner('Analyzing Image..'):
                result = predict_disease(uploaded_file, model, class_names)
                disease_info = get_disease_info(result['predicted_class'])

                if result['predicted_class'] == 'Healthy':
                    box_class = 'prediction-box healthy'
                else:
                    box_class = 'prediction-box disease'

                st.markdown(f"""
                <div class="{box_class}">
                <h2 style="margin:0">{result['predicted_class'].replace('_',' ')}</h2>
                <h3 style="margin:0">Confidence: {result['confidence']:.1f}%</h3>
                </div>
                """,unsafe_allow_html=True)

                df=pd.DataFrame(result['top_3'], columns=['Disease','Confidence'])
                fig = px.bar(df, x='Confidence', y='Disease', orientation='h',
                             title = 'Top 3 Predictions',
                             color = 'Confidence',
                             color_continuous_scale = 'RdYlGn',
                             text = df['Confidence'].apply(lambda x: f'{x:.1f}%'))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

    if uploaded_file is not None:
        st.markdown("----")
        col3,col4 = st.columns([1,1])
        with col3:
            st.markdown(f"""
            <div style="background-color:#f9f9f9; padding:1.5rem; border-radius:10px;">
                <p><strong>Description:</strong> {disease_info['description']}</p>
                <p><strong>Symptoms:</strong> {disease_info['symptoms']}</p>
                <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
                <p><strong>Prevention:</strong> {disease_info['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if result['heatmap_image'] is not None:
                st.markdown("Grad-CAM Heatmap")
                st.image(result['heatmap_image'], 
                        caption="Areas the model focused on (red = high importance)",
                        width='stretch')
                st.caption("The heatmap shows which parts of the image influenced the model's decision")

if __name__ == "__main__":
    main()