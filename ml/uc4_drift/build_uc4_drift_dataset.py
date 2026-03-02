import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import shuffle

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

EMB_PATH = "ml/uc4_drift/datasets/vggface2_embeddings.npy"
IDS_PATH = "ml/uc4_drift/datasets/vggface2_identity_ids.npy"
OUTPUT_DIR = "ml/uc4_drift/datasets"

T = 120
NUM_SESSIONS = 1000  # total (500 genuine, 500 drift)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------

embeddings = np.load(EMB_PATH)
identity_ids = np.load(IDS_PATH)

print("Loaded embeddings:", embeddings.shape)

# Organize by identity
identity_dict = {}
for emb, idx in zip(embeddings, identity_ids):
    identity_dict.setdefault(idx, []).append(emb)

identity_keys = list(identity_dict.keys())

# ---------------------------------------------------
# SESSION BUILDING
# ---------------------------------------------------

def build_genuine_session(identity):
    emb_list = identity_dict[identity]
    seq = np.array(emb_list)
    indices = np.random.choice(len(seq), T, replace=True)
    session = seq[indices]

    enrollment = session[0]
    deltas = session - enrollment
    cosine = np.sum(session * enrollment, axis=1, keepdims=True)

    return np.concatenate([deltas, cosine], axis=1)


def build_drift_session(identity_A, identity_B):
    emb_A = np.array(identity_dict[identity_A])
    emb_B = np.array(identity_dict[identity_B])

    idx_A = np.random.choice(len(emb_A), T, replace=True)
    idx_B = np.random.choice(len(emb_B), T, replace=True)

    seq_A = emb_A[idx_A]
    seq_B = emb_B[idx_B]

    session = []

    for t in range(T):
        alpha = t / (T - 1)
        blended = (1 - alpha) * seq_A[t] + alpha * seq_B[t]
        blended = blended / np.linalg.norm(blended)
        session.append(blended)

    session = np.array(session)

    enrollment = session[0]
    deltas = session - enrollment
    cosine = np.sum(session * enrollment, axis=1, keepdims=True)

    return np.concatenate([deltas, cosine], axis=1)


# ---------------------------------------------------
# GENERATE DATASET
# ---------------------------------------------------

X = []
y = []

print("Building genuine sessions...")
for _ in tqdm(range(NUM_SESSIONS // 2)):
    identity = np.random.choice(identity_keys)
    X.append(build_genuine_session(identity))
    y.append(0)

print("Building drift sessions...")
for _ in tqdm(range(NUM_SESSIONS // 2)):
    identity_A, identity_B = np.random.choice(identity_keys, 2, replace=False)
    X.append(build_drift_session(identity_A, identity_B))
    y.append(1)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y)

np.save(os.path.join(OUTPUT_DIR, "uc4_drift_sequences.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "uc4_drift_labels.npy"), y)

print("\n✅ UC4 Drift Dataset Built")
print("X shape:", X.shape)
print("y shape:", y.shape)