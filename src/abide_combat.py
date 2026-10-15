import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from neuroCombat import neuroCombat

# --- 1. Load ABIDE Data (CC200 Atlas) ---
print("Fetching ABIDE dataset...")
abide = datasets.fetch_abide_pcp(
    derivatives=["rois_cc200"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True,
    n_subjects=None
)

pheno = pd.DataFrame(abide.phenotypic)
y = (pheno["DX_GROUP"].values == 1).astype(int)  # 1 = autism, 0 = control
sites = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in pheno["SITE_ID"]])
age = pheno["AGE_AT_SCAN"].values

print(f"Subjects: {len(y)} | Autism: {y.sum()} | Sites: {len(set(sites))}")

# --- 2. Extract Functional Connectivity Features ---
conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
X_raw = np.nan_to_num(conn.fit_transform(abide.rois_cc200))

# --- 3. Apply ComBat Harmonization ---
print("Applying ComBat harmonization across scanner sites...")
# neuroCombat expects shape: (features, subjects)
# --- 3. Apply ComBat Harmonization ---
print("Applying ComBat harmonization across scanner sites...")
data_to_harmonize = X_raw.T  # shape: (features, subjects)
covars = pd.DataFrame({
    'site': sites,
    'age': age,
    'diagnosis': y
})

combat_result = neuroCombat(
    dat=data_to_harmonize,
    covars=covars,
    batch_col='site',
    categorical_cols=['diagnosis'],
    continuous_cols=['age']
)

X_combat = combat_result['data'].T
# --- 4. Compare Subgroup Stability & Site Confound: Raw vs Harmonized ---
def evaluate_subtyping(Z, sites_sub, label=""):
    pca = PCA(10, random_state=0).fit(Z)
    km = KMeans(2, n_init=10, random_state=0).fit(pca.transform(Z))
    
    chi_p = chi2_contingency(pd.crosstab(sites_sub, km.labels_))[1]
    
    rng = np.random.default_rng(0)
    uniq = np.unique(sites_sub)
    aris = []
    for _ in range(30):
        half = rng.permutation(uniq)[: len(uniq) // 2]
        A, B = np.isin(sites_sub, half), ~np.isin(sites_sub, half)
        pA = PCA(10, random_state=0).fit(Z[A])
        kA = KMeans(2, n_init=10, random_state=0).fit(pA.transform(Z[A]))
        pB = PCA(10, random_state=0).fit(Z[B])
        kB = KMeans(2, n_init=10, random_state=0).fit(pB.transform(Z[B]))
        aris.append(adjusted_rand_score(kA.predict(pA.transform(Z[B])), kB.labels_))
    
    mean_ari = np.mean(aris)
    print(f"[{label}] Subgroup ARI stability: {mean_ari:.3f} | Cluster vs Site Chi-sq p: {chi_p:.2e}")
    return mean_ari, chi_p

print("\n--- Subtyping Results ---")
Xa_raw, sa = StandardScaler().fit_transform(X_raw[y == 1]), sites[y == 1]
Xa_com, _ = StandardScaler().fit_transform(X_combat[y == 1]), sites[y == 1]

ari_raw, p_raw = evaluate_subtyping(Xa_raw, sa, label="Raw Features")
ari_com, p_com = evaluate_subtyping(Xa_com, sa, label="ComBat Harmonized")

results = pd.DataFrame([
    {"Feature": "Raw", "ARI_Stability": ari_raw, "Site_Chi2_p": p_raw},
    {"Feature": "ComBat", "ARI_Stability": ari_com, "Site_Chi2_p": p_com}
])
results.to_csv("reports/combat_comparison.csv", index=False)
print("\nResults saved to reports/combat_comparison.csv")