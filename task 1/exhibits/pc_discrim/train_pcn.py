import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from pcn_model import PCN
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

df = pd.read_csv("IMDB.csv")

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

# Model parameters
x_dim = _X.shape[1]
y_dim = _Y.shape[1]
n_iter = 2
mb_size = 250
n_batches = int(_X.shape[0] / mb_size)
save_point = 5

# Set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

# Build the PCN model
print("--- Building Model ---")
model = PCN(subkeys[1], x_dim, y_dim, hid1_dim=512, hid2_dim=512, T=20,
            dt=1., tau_m=20., act_fx="sigmoid", eta=0.001, exp_dir="exp",
            model_name="pcn")
model.save_to_disk()  # Save initial model state
print("--- Starting Simulation ---")

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

# Training loop
trAcc_set, acc_set, efe_set = [], [], []
sim_start_time = time.time()

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X, Y = _X[ptrs, :], _Y[ptrs, :]

    train_EFE, n_samp_seen = 0., 0
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        idx = j * mb_size
        Xb, Yb = X[idx: idx + mb_size, :], Y[idx: idx + mb_size, :]
        yMu_0, yMu, _EFE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        train_EFE += _EFE * mb_size
        n_samp_seen += Yb.shape[0]

    nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
    _, tr_acc = eval_model(model, _X, _Y, mb_size=1000)
    trAcc_set.append(tr_acc)
    acc_set.append(acc)
    efe_set.append(train_EFE / n_samp_seen)
    io_str = "{} Dev: Acc = {}, NLL = {} | Tr: Acc = {}, EFE = {}".format(i, acc, nll, tr_acc, train_EFE / n_samp_seen)
    print(io_str)

sim_end_time = time.time()
sim_time_hr = (sim_end_time - sim_start_time) / 3600.0
print("Trial.sim_time = {} h  Best Acc = {}".format(sim_time_hr, jnp.amax(jnp.array(acc_set))))

# Save results
jnp.save("exp/trAcc.npy", jnp.array(trAcc_set))
jnp.save("exp/acc.npy", jnp.array(acc_set))
jnp.save("exp/efe.npy", jnp.array(efe_set))
