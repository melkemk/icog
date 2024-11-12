import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from pcn_model import PCN
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, plot_latents

# Load and preprocess the IMDB dataset
def preprocess_imdb_data(csv_file='IMDB.csv'):
    df = pd.read_csv(csv_file)

    # Encode the sentiment labels as binary (positive = 1, negative = 0)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Use TF-IDF to convert text to a matrix of token counts
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['review']).toarray()
    Y = df['sentiment'].values

    # Convert labels to one-hot encoding
    Y = np.eye(2)[Y]

    # Split data into training and development (validation) sets
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert to jnp arrays
    _X = jnp.array(X_train)
    _Y = jnp.array(Y_train)
    Xdev = jnp.array(X_dev)
    Ydev = jnp.array(Y_dev)

    return _X, _Y, Xdev, Ydev

# Evaluation function
def eval_model(model, Xdev, Ydev, mb_size):
    n_batches = int(Xdev.shape[0] / mb_size)
    n_samp_seen, nll, acc = 0, 0., 0.
    for j in range(n_batches):
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        _nll = measure_CatNLL(yMu_0, Yb) * Xb.shape[0]
        _acc = measure_ACC(yMu_0, Yb) * Yb.shape[0]
        nll += _nll
        acc += _acc
        n_samp_seen += Yb.shape[0]

    nll /= Xdev.shape[0]
    acc /= Xdev.shape[0]
    return nll, acc

# Main program execution
if __name__ == '__main__':
    # Preprocess IMDB data
    _X, _Y, Xdev, Ydev = preprocess_imdb_data(csv_file='IMDB.csv')

    # Model parameters
    dkey = random.PRNGKey(1234)
    model = PCN(dkey=dkey, loadDir="/home/melke/Desktop/ngc/ngc-museum/exhibits/pc_discrim/exp/pcn")  
    # Evaluate performance on test set
    nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
    print("------------------------------------")
    print(f"=> NLL = {nll}  Acc = {acc}")

    # Extract latents and visualize via t-SNE
    latents = model.get_latents()
    codes = extract_tsne_latents(np.asarray(latents))
    plot_latents(codes, Ydev, plot_fname="exp/pcn_latents_imdb.jpg")

    # Optionally save the results if needed
    np.save('exp/imdb_test_latents.npy', latents)
