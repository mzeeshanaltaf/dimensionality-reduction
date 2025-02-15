import streamlit as st
from util import *

# Page title of the application
page_title = "DimenSight"
page_icon = ""
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

# Application Title and description
st.title(f'{page_title}{page_icon}')
st.write('***:blue[Visualize Data, Simplify Dimensions 🎨✨]***')
st.write("""
*DimenSight is your go-to web app for exploring datasets in 2D or 3D! 🌐 Choose from popular datasets like Iris 🌸, 
Penguins 🐧, or MNIST 🔢, and apply powerful techniques like PCA, UMAP, LDA, or t-SNE to visualize data in stunning 
detail. Whether you're a data enthusiast or a pro, DimenSight makes dimensionality reduction simple, interactive, 
and fun! 🚀📊*
""")
# Display footer in the sidebar
display_footer()

st.header('Configuration:')
col1, col2, col3 = st.columns(3, border=True)
with col1:
    dataset_selection = st.radio('Select the Dataset:', ['Iris', 'Penguin'], horizontal=False,
                                  label_visibility="visible")
with col2:
    plot_selection = st.radio('Select the Plot Type:', ['2D', '3D'], horizontal=False,
                                 label_visibility="visible")
    num_dim = int(plot_selection[0]) # Based on the plot type, identify number of dimensions the data to be reduced to
with col3:
    technique_selection = st.pills('Select the Technique:', ['PCA', 'LDA', 't-SNE', 'UMAP'], selection_mode="multi",
                                 default=['PCA'], label_visibility="visible")
apply = st.button("Apply", type='primary')

if apply:
    with st.spinner('Processing ...', show_time=True):
        if not technique_selection:
            st.warning('Please select the Dimensionality Reduction Technique', icon=":material/warning:")
        else:
            for technique in technique_selection:
                if technique == 'PCA':
                    dimensionality_reduction_pca(dataset_selection, num_dim)
                if technique == 't-SNE':
                    dimensionality_reduction_tsne(dataset_selection, num_dim)
                if technique == 'UMAP':
                    dimensionality_reduction_umap(dataset_selection, num_dim)
                if technique == 'LDA':
                    dimensionality_reduction_lda(dataset_selection, num_dim)






