# importing the required libraries
import requests
import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from app__util import Resnet50FeatureExtractor, preprocessing_transform, load_patchcore_assets, load_sample_images, get_defect_name
import time

# setting the page configuration
st.set_page_config(layout="wide", page_title='Capsule Anomaly Detection System', page_icon="💊")

# specifying the path to the patchcore memory bank
MEMORY_BANK_PATH = "capsule_patchcore_assets.pt"


# checking for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Configuration
# FILE_ID = 'your_google_drive_file_id_here'
# OUTPUT_PATH = "capsule_patchcore_assets.pt"

# @st.cache_resource
# def initialize_assets():
#     """
#     Downloads the Coreset memory bank from Google Drive if not present 
#     and loads it into memory.
#     """
#     if not os.path.exists(OUTPUT_PATH):
#         with st.spinner("Downloading memory bank (this may take a moment)..."):
#             url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
#             gdown.download(url, OUTPUT_PATH, quiet=False)
#             st.success("Assets downloaded successfully.")
    
#     # Load to CPU for universal compatibility
#     return torch.load(OUTPUT_PATH, map_location=torch.device('cpu'))

# # Use the function
# assets = initialize_assets()


# loading the feature extraction system with caching
show_spinner="Loading model.."

@st.cache_resource(show_spinner=show_spinner, show_time=True)
def load_feature_extraction_system(url, filename):
    if not os.path.exists(filename):
        # with st.spinner("Fetching model assets..."):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Failed to download model. Check the URL.")
    # loading the PatchCore assets (memory bank and threshold)
    backbone = Resnet50FeatureExtractor()
    assets = load_patchcore_assets(filename)
    memory_bank = assets['memory_bank']
    threshold = assets['threshold']
    
   
    # specifying the output for the feature extraction system
    return backbone, memory_bank, threshold

# Usage
url = "https://drive.usercontent.google.com/download?id=1wnaR4x8_GCfLy1_uyQcRU1D49_gz5gsL&export=download&authuser=0&confirm=t"
# download_memory_bank(url, "capsule_patchcore_assets.pt")

# initializing the feature extraction system
backbone, memory_bank, best_threshold = load_feature_extraction_system(url, "capsule_patchcore_assets.pt")

# setting up the image preprocessing transform
img_transform = preprocessing_transform()

# initializing session state variables
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

if 'results_storage' not in st.session_state:
    st.session_state.results_storage = []

if 'good_item' not in st.session_state:
    st.session_state.good_item = 0

if 'defect_item' not in st.session_state:
    st.session_state.defect_item = 0

if 'selected_imgs' not in st.session_state:
    st.session_state.selected_imgs = []

if 'shown_once' not in st.session_state:
    st.session_state.shown_once = []




# custom styling for Streamlit Tabs
st.markdown("""
    <style>
    /* making the tab container full width */
    div[data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 0px; 
    }

    /* 2. styling the buttons */
    button[data-baseweb="tab"] {
        flex: 1; 
        height: 80px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }

    /* targetting the text inside the buttons */
    button[data-baseweb="tab"] p {
        font-size: 1.25rem !important; /* LARGE FONT SIZE */
        font-weight: 800 !important; /* Extra Bold */
        letter-spacing: 1px;
    }

    /* styling the active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1c96c5 !important; 
        border-bottom: 1px solid !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)      

# writing the main title and description
st.title("💊 Optimized PatchCore for Surface Anomaly Detection in Capsule Manufacturing")
st.markdown("[Link to the Source Code on GitHub](https://github.com/Oluwatobi-coder/Optimized-Capsule-Manufacturing-Defect-Detection-using-PatchCore-)")
st.markdown("**Author**: Bello Oluwatobi")
st.markdown("**Last Updated on**: 29th December, 2025")
st.markdown("""
            This project implements an optimized **PatchCore model** to detect manufacturing defects in drug capsules. By using a **ResNet50 backbone** with **weighted Layer-2** feature extraction and **high-resolution** spatial mapping, the system achieves a **99% AUROC** score in identifying capsule defects.
            """)

# setting up the tabs for the application
tab1, tab2 = st.tabs([" 💊 Capsule Anomaly Detector", "📋 Inspection Guide and Model Metrics"])

# tab 1: the Capsule Anomaly Detector
with tab1:
    col1, col2, col3 = st.columns([1.5, 1, 2])
    # setting up the image upload section via pre-loaded samples or custom upload
    with col1:
        st.write("#### Quick Test")
        st.write("**Try a pre-loaded batch of good and defect capsules**")
        sample_btn = st.button("Load Sample Batch", width="stretch", type="primary")
        if sample_btn:
            selected_imgs = load_sample_images("./sample_images")
            st.session_state.selected_imgs.extend(selected_imgs)
            if len(st.session_state.selected_imgs) >= 1:
                st.success(f"Loaded **{len(st.session_state.selected_imgs)} images** successfully!")
    with col3:
        st.write("#### Custom Test")
        st.markdown("[Download Sample Test Images](https://drive.google.com/drive/folders/1qCow9EmWC3V95jwlIkFp_iqI4RCw4hSE?usp=sharing)")
        uploaded_files = st.file_uploader("Upload Capsule Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}")

    st.divider()



    # setting up the analysis button and processing logic
    if (len(uploaded_files) >= 1 or len(st.session_state.selected_imgs) >=1) and backbone != None:
        # setting up the analysis button
        if(st.button("Analyze Image Batch")):
            # setting up the status bar for analysis progress
            with st.status(f"**:blue[Starting Analysis...]**") as status:
                # simulating a brief delay before starting analysis
                time.sleep(0.005)
                # initializing storage for results and counters
                results_storage = []
                good_item = 0
                defect_item = 0
                batch_num = 0
                # processing each uploaded image
                for uploaded_file in (uploaded_files or st.session_state.selected_imgs):
                    batch_num += 1
                    # updating the status bar for each image being processed
                    status.update(label=f"**:blue[Analyzing {batch_num} of {len(uploaded_files) or len(st.session_state.selected_imgs)}]**")
                    # loading the image via PIL
                    image = Image.open(uploaded_file).convert('RGB')

            
                    # preprocessing the image and preparing it for model input
                    input_img = img_transform(image).unsqueeze(0).to(DEVICE)
                    
                    # getting the defect label from the filename
                    defect_label = get_defect_name(uploaded_file)
                    
                    # performing feature extraction and anomaly detection
                    with torch.no_grad():
                        features = backbone(input_img)
                        distances = torch.cdist(features, memory_bank, p=2.0)
                        dist_score, dist_score_idxs = torch.min(distances, dim=1)
                        s_star = torch.max(dist_score)
                        # creating the heatmap
                        segm_map = dist_score.view(1, 1, 40, 40)
                        segm_map = torch.nn.functional.interpolate(
                                segm_map,
                                size=(320, 320),
                                mode='bilinear'
                            ).cpu().squeeze().numpy()
                        # generating the anomaly score and classifying image as good or defect
                        y_score_image = s_star.cpu().numpy()
                        y_pred_image = 1*(y_score_image >= best_threshold)

                        anomaly_score = y_score_image / best_threshold
                        # determining confidence percentage and updating counters
                        if anomaly_score >= 1:
                            percent = f'{((anomaly_score - 1) * 100):.2f}% above threshold value'
                            defect_item +=1
                        else:
                            percent = f'{((1 - anomaly_score) * 100):.2f}% below threshold value'
                            good_item +=1
                        class_label = ['Good','Defect']

                        # preparing the actual image for display
                        img_np = input_img.squeeze().permute(1,2,0).cpu().numpy()
                        img_normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                        # preparing the heatmap for display
                        heat_map = segm_map
                
                        # storing all relevant data for display in the results table
                        image_data = {
                        "actual_image": img_normalized,
                        "heatmap": heat_map,
                        "defect_type": defect_label,
                        "anomaly_score": y_score_image / best_threshold,
                        "prediction": class_label[y_pred_image],
                        "percent": percent,
                        }

                        results_storage.append(image_data)
                # updating the session state with results and counters        
                st.session_state.results_storage = []
                st.session_state.good_item = 0
                st.session_state.defect_item = 0
                st.session_state.results_storage.extend(results_storage)
                st.session_state.good_item += good_item
                st.session_state.defect_item += defect_item
                st.session_state.selected_imgs = []
                st.session_state.shown_once = True
                del st.session_state.selected_imgs
                

                
                st.session_state.uploader_key += 1

                st.rerun()
    else:
        st.button("Analyze Image Batch", disabled=True, help="Please upload a set of images or a single image first.")


    # displaying the results table
    if st.session_state.results_storage:
        results_storage = st.session_state.results_storage
        good_item = st.session_state.good_item
        defect_item = st.session_state.defect_item   
        
        if (st.session_state.good_item != 0 or  st.session_state.defect_item != 0) and st.session_state.shown_once:
            st.toast(f'Successfully processed {len(st.session_state.results_storage)} images. The set had {good_item} good product(s) and {defect_item} defect(s)', icon='✅')
            st.session_state.shown_once = False
        st.write("### Detection Results Table")
        st.write(f'**Threshold value:  {best_threshold:.2f}**')
        st.write(f'**Summary:**')
        col1, col2 = st.columns([0.25, 0.25])
        with col1:
            st.write(f'**:green[Good Capsule(s):  {good_item}]**')
        with col2:
            st.write(f'**:red[Defect Capsule(s):  {defect_item}]**')
        st.markdown('---')
        header_cols = st.columns([1, 1, 1.5])
        header_cols[0].write("**Actual Image**")
        header_cols[1].write("**Heatmap**")
        header_cols[2].write("**Analysis Details**")

        # displaying each result in a structured format
        for item in results_storage:
            row_cols = st.columns([1, 1, 1.5])
            
            # setting up column 1: Actual Image
            with row_cols[0]:
                st.image(item['actual_image'], width="stretch")
                
            # setting up column 2: Heatmap
            with row_cols[1]:
                # Display heatmap using matplotlib
                fig, ax = plt.subplots()
                # displaying heatmap with color mapping
                im = ax.imshow(item['heatmap'], cmap='jet', vmin=best_threshold, vmax=best_threshold*2)
                ax.axis('off')
                st.pyplot(fig, clear_figure=True, width="stretch")
                
            # setting up column 3: Analysis Details
            with row_cols[2]:
                st.markdown(f"""
                - **Fault Type:** `{item['defect_type']}`
                - **Prediction:** {item['prediction']}
                - **Anomaly Score:** `{item['anomaly_score']:.4f}`
                - **Confidence:** {item['percent']}
                """)


# tab 2: Inspection Guide and Model Metrics 
with tab2:
    # writing the inspection guide and model metrics
    st.markdown("### Inspection Guide: Good vs. Defective")
    st.info("This guide helps you understand the specific capsule defects detected by the model (Hover on each image and select the ⛶ for fullscreen view).")
    # displaying sample images of good and defective capsules
    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
    with col_a:
        st.markdown(f"###### ✅ Good")
        st.image("./sample_images/000_good.png", width="stretch")
    with col_b:
        st.markdown(f"###### ❌ Crack")
        st.image("./sample_images/007_crack.png", width="stretch")
    with col_c:
        st.markdown(f"###### ❌ Faulty Imprint")
        st.image("./sample_images/000_faulty_imprint.png", width="stretch")
    with col_d:
        st.markdown(f"###### ❌ Poke")
        st.image("./sample_images/008_poke.png", width="stretch")
    with col_e:
        st.markdown(f"###### ❌ Scratch")
        st.image("./sample_images/007_scratch.png", width="stretch")
    with col_f:
        st.markdown(f"###### ❌ Squeeze")
        st.image("./sample_images/000_squeeze.png", width="stretch")
    
    # displaying the model metrics with images
    st.markdown("### Model Metrics")
    st.info("The anomaly detector uses **ResNet50 model** with **weighted Layer-2 features** and **high spatial resolution** to identify more complex defects layers.")
    col1, col2 = st.columns(2)
    # displaying the ROC curve
    with col1:
        st.markdown("<h4 style='text-align: center;'>ROC Curve</h4>", unsafe_allow_html=True)
        # Replace 'path_to_roc.png' with your actual filename
        st.image("./results_images/roc_curve.png", 
                caption="**Receiver Operating Characteristic (0.99 AUROC)**", 
                width="stretch")
        st.markdown("""
                    The model achieved a **0.99 (99%) Area Under the Receiver Operating Characteristic (AUROC)** score, indicating a nearly perfect accuracy in differentiating between good and defective capsules. This high sensitivity allows the detection of complex flaws with fewer false alarms.
                    """)
    # displaying the confusion matrix
    with col2:
        st.markdown("<h4 style='text-align: center;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        # Replace 'path_to_cm.png' with your actual filename
        st.image("./results_images/confusion_matrix.png", 
                caption="**Classification Accuracy: Good vs. Defect**", 
                width="stretch")
        st.markdown("""
                    On a test set of 132 capsules, the model achieved a **100% detection rate** for defects with **zero false negatives**. **Only 3 good capsules** were flagged as defects. This high accuracy and minimal false alarm rate ensure operational efficiency and, most importantly, consumer safety.
                    """)
