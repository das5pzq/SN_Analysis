import ROOT
import numpy as np
from ROOT import TLorentzVector
from pathlib import Path

# ================================
MUON_MASS = 0.1056  # GeV
INPUT_FILE = Path("data/compact_mc_jpsi_target_pythia8_acc.root")
OUTPUT_NPY_FILE = Path("data/compact_mc_jpsi_target_pythia8_acc.npy")
file = ROOT.TFile(str(INPUT_FILE), "READ")
tree = file.Get("tree")
if not tree:
    raise RuntimeError("Tree 'tree' not found in ROOT file")

n_entries = tree.GetEntries()
features = []

for i, event in enumerate(tree):
    if i % 10000 == 0:
        print(f"Processing event {i}/{n_entries}")

    # Create TLorentzVectors for mu+ and mu-
    mu_pos = TLorentzVector()
    mu_neg = TLorentzVector()
    mu_pos.SetXYZM(event.mu_pos_px, event.mu_pos_py, event.mu_pos_pz, MUON_MASS)
    mu_neg.SetXYZM(event.mu_neg_px, event.mu_neg_py, event.mu_neg_pz, MUON_MASS)

    dimu = mu_pos + mu_neg



    if dimu.M()< 2.0 or dimu.M() > 4.0:
        continue

    # Compute basic observables
    dimu_mass = dimu.M()
    dimu_pt = dimu.Pt()
    dimu_phi = dimu.Phi()
    dimu_eta = dimu.Eta()
    dimu_y = dimu.Rapidity()
    dimu_E = dimu.E()
    dimu_pz = dimu.Pz()
    dimu_mT = (dimu.M()**2 + dimu.Pt()**2)**0.5

    cos_open = mu_pos.Vect().Dot(mu_neg.Vect()) / (mu_pos.Vect().Mag() * mu_neg.Vect().Mag())
    opening_angle = np.arccos(np.clip(cos_open, -1, 1))

    # Track-level angles
    theta_pos = np.arctan2(mu_pos.Pt(), mu_pos.Pz())
    theta_neg = np.arctan2(mu_neg.Pt(), mu_neg.Pz())

    # ΔpT and energies
    dpt = mu_pos.Pt() - mu_neg.Pt()
    Epos, Eneg = mu_pos.E(), mu_neg.E()

    # Station-1 info
    x_pos_st1 = getattr(event, "rec_track_pos_x_st1", 0.0)
    x_neg_st1 = getattr(event, "rec_track_neg_x_st1", 0.0)
    px_pos_st1 = getattr(event, "rec_track_pos_px_st1", 0.0)
    px_neg_st1 = getattr(event, "rec_track_neg_px_st1", 0.0)

    features.append([
        dimu_y, dimu_eta, dimu_E, dimu_pz,
        dimu_mass, dimu_pt, dimu_phi,
        theta_pos, theta_neg, opening_angle,
        dpt, dimu_mT, Epos, Eneg,
        x_pos_st1, x_neg_st1, px_pos_st1, px_neg_st1
    ])

file.Close()

# Convert to NumPy and save
features = np.array(features, dtype=np.float64)
np.save(OUTPUT_NPY_FILE, features)

print(f"\n Saved NumPy file: {OUTPUT_NPY_FILE}")
print(f"Shape: {features.shape}")

