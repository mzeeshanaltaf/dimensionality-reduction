import streamlit as st
from sklearn.datasets import load_iris
from keras.datasets import mnist
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
import pandas as pd
import plotly.express as px

def load_dataset(dataset):
    if dataset == 'Iris':
        # Load the Iris dataset
        iris = load_iris()
        X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
        y = iris.target
        target_names = iris.target_names

        # Convert to DataFrame
        df = pd.DataFrame(X)
        df['Target'] = y
        df['Target_Names'] = df['Target'].apply(lambda x: target_names[x])
        df.drop(columns=['Target'], inplace=True)

    elif dataset == 'Penguin':
        penguins = sns.load_dataset('penguins')

        # Drop rows with missing values and non-numeric columns (species, sex, island)
        penguins = penguins.dropna()
        penguins_numeric = penguins.select_dtypes(include=['float64', 'int64'])

        # Separate features (X) and target variable (y)
        X = penguins_numeric

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        df = pd.DataFrame(X)
        df['Target_Names'] = penguins["species"]

    elif dataset == 'MNIST':
        (X, y), (_, _) = mnist.load_data()
        X = X.reshape(len(X), -1)
        df_mnist = pd.DataFrame(X)
        df_mnist['Target_Names'] = y.astype(str)
        df = df_mnist.sample(n=1000, random_state=42).reset_index(drop=True) # Get the random sample from MNIST dataset

    return df

def dimensionality_reduction_pca(dataset, num_dim):
    df = load_dataset(dataset)

    X = df.drop(columns=['Target_Names'])
    pca = PCA(n_components=num_dim)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca)
    df_pca['Target_Names'] = df['Target_Names']

    if num_dim == 2:
        plot_2d_chart(df_pca, dataset, 'PCA')
    else:
        plot_3d_chart(df_pca, dataset, 'PCA')

def dimensionality_reduction_tsne(dataset, num_dim):
    df = load_dataset(dataset)
    X = df.drop(columns=['Target_Names'])

    # Apply t-SNE to reduce the dimensions to 2
    tsne = TSNE(n_components=num_dim, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(X_tsne)
    df_tsne['Target_Names'] = df['Target_Names']

    if num_dim == 2:
        plot_2d_chart(df_tsne, dataset, 't-SNE')
    else:
        plot_3d_chart(df_tsne, dataset, 't-SNE')

def dimensionality_reduction_umap(dataset, num_dim):
    df = load_dataset(dataset)
    X = df.drop(columns=['Target_Names'])

    # Apply UMAP to reduce the dimensions to 2
    umap_model = umap.UMAP(n_components=num_dim)
    X_umap = umap_model.fit_transform(X)

    df_umap = pd.DataFrame(X_umap)
    df_umap['Target_Names'] = df['Target_Names']

    if num_dim == 2:
        plot_2d_chart(df_umap, dataset, 'UMAP')
    else:
        plot_3d_chart(df_umap, dataset, 'UMAP')

def dimensionality_reduction_lda(dataset, num_dim):
    df = load_dataset(dataset)
    df = df.dropna()
    X = df.drop(columns=['Target_Names'])
    y = df['Target_Names']

    # LDA has a fundamental constraint on the maximum number of components
    # n_components ≤ min(n_features, n_classes - 1)
    allowed_components = min(len(X.columns), len(y.unique()) - 1)

    if allowed_components < num_dim:
        st.warning(f'LDA for {dataset} dataset with {num_dim} components is not possible. Maximum allowed component for LDA '
                   f'is {allowed_components}.', icon=":material/warning:")

    else:
        # Apply LDA to reduce the dimensions to 2
        lda = LDA(n_components=num_dim)  # We want to reduce it to 2 components for visualization
        X_lda = lda.fit_transform(X, y)

        df_lda = pd.DataFrame(X_lda)
        df_lda['Target_Names'] = df['Target_Names']

        if num_dim == 2:
            plot_2d_chart(df_lda, dataset, 'LDA')

        else:
            plot_3d_chart(df_lda, dataset, 'LDA')

def plot_2d_chart(df, dataset, technique):

    df.columns = [f'{technique}1', f'{technique}2', 'Target_Names']
    df = df.dropna() # Drop NA values (if any)

    # Chart Title
    st.title(f"{dataset} Dataset {technique} Visualization")

    # Scatter Plot
    st.scatter_chart(df, x=f'{technique}1', y=f'{technique}2', color='Target_Names')

def plot_3d_chart(df, dataset, technique):

    df.columns = [f'{technique}1', f'{technique}2', f'{technique}3', 'Target_Names']
    df = df.dropna() # Drop NA values (if any)

    # Chart Title
    st.title(f"{dataset} Dataset {technique} Visualization")

    fig = px.scatter_3d(df, x=f'{technique}1', y=f'{technique}2', z=f'{technique}3', color='Target_Names', title="3D Scatter Plot")

    # Increase figure size
    fig.update_layout(width=1000,  height=1000)

    # Show the plot in Streamlit
    st.plotly_chart(fig)


def display_footer():
    footer = """
    <style>
    /* Ensures the footer stays at the bottom of the sidebar */
    [data-testid="stSidebar"] > div: nth-child(3) {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
    }

    .footer {
        color: grey;
        font-size: 15px;
        text-align: center;
        background-color: transparent;
    }
    </style>
    <div class="footer">
    Made with ❤️ by <a href="mailto:zeeshan.altaf@gmail.com">Zeeshan</a>.
    </div>
    """
    st.sidebar.markdown(footer, unsafe_allow_html=True)