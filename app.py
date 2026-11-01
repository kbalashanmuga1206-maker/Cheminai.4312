import random

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(page_title="Molecule Studio", page_icon="🧪", layout="wide")

STYLES = {
    "Stick": {"stick": {}},
    "Ball and stick": {"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}},
    "Spacefill": {"sphere": {}},
    "Line": {"line": {}},
}


# ---------- helpers ----------
def parse(smiles):
    smiles = smiles.strip()
    return Chem.MolFromSmiles(smiles) if smiles else None


def properties(mol):
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight": round(Descriptors.MolWt(mol), 3),
        "Exact mass": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "QED (drug-likeness)": round(QED.qed(mol), 3),
        "Lipinski violations": sum(
            [
                Descriptors.MolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                Lipinski.NumHDonors(mol) > 5,
                Lipinski.NumHAcceptors(mol) > 10,
            ]
        ),
    }


def svg_2d(mol, w=420, h=320):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def html_3d(mol, style):
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=42) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:
        pass
    v = py3Dmol.view(width=620, height=440)
    v.addModel(Chem.MolToMolBlock(mh), "mol")
    v.setStyle(STYLES[style])
    v.zoomTo()
    return v._make_html()


def report_html(smiles, mol):
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in properties(mol).items())
    return f"""<html><head><meta charset="utf-8"><title>Molecule report</title>
<style>body{{font-family:sans-serif;max-width:700px;margin:2rem auto}}
td{{border-bottom:1px solid #ddd;padding:6px 12px}}</style></head><body>
<h1>Molecule Report</h1><p><b>SMILES:</b> {smiles}</p>{svg_2d(mol)}
<table>{rows}</table>
<p><small>Created by K.Balashanmuga, AIML Student, SRM Institution of Science and Technology, Thiruchirappalli</small></p>
</body></html>"""


# ---------- sidebar ----------
st.title("🧪 Molecule Studio")
tab1, tab2, tab3 = st.tabs(["Analyze", "Compare", "Assessment"])

# ---------- Analyze ----------
with tab1:
    smiles = st.text_input("Enter a SMILES notation", "CC(=O)Oc1ccccc1C(=O)O")
    mol = parse(smiles)
    if not smiles.strip():
        st.info("Type a SMILES string to begin.")
    elif mol is None:
        st.error("❌ Invalid SMILES - the structure could not be parsed.")
    else:
        st.success("✅ Valid molecular structure")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2D structure")
            st.image(svg_2d(mol))
        with c2:
            st.subheader("3D structure")
            style = st.selectbox("Visualization style", list(STYLES))
            html = html_3d(mol, style)
            if html:
                components.html(html, height=460)
            else:
                st.warning("Could not generate 3D coordinates for this molecule.")
        st.subheader("Molecular properties")
        props = properties(mol)
        st.table(pd.DataFrame(props.items(), columns=["Property", "Value"]).astype(str))
        st.download_button("⬇️ Download report (HTML)", report_html(smiles, mol), "molecule_report.html", "text/html")

# ---------- Compare ----------
with tab2:
    text = st.text_area(
        "One SMILES per line (optionally 'name,SMILES')",
        "Ethanol,CCO\nAspirin,CC(=O)Oc1ccccc1C(=O)O\nCaffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        height=140,
    )
    items = []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        name, _, smi = line.rpartition(",")
        name = name.strip() or f"Molecule {i + 1}"
        m = parse(smi)
        if m is None:
            st.warning(f"Skipped invalid SMILES: {smi.strip()}")
        else:
            items.append((name, m))
    if len(items) >= 2:
        df = pd.DataFrame({n: properties(m) for n, m in items}).astype(str)
        st.dataframe(df, use_container_width=True)
        cols = st.columns(len(items))
        for col, (n, m) in zip(cols, items):
            col.caption(n)
            col.image(svg_2d(m, 280, 220))
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for _, m in items]
        sim = pd.DataFrame(
            [[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps],
            index=[n for n, _ in items],
            columns=[n for n, _ in items],
        ).round(3)
        st.subheader("Tanimoto similarity")
        st.dataframe(sim, use_container_width=True)
        st.download_button("⬇️ Download comparison (CSV)", df.to_csv(), "comparison.csv", "text/csv")
    else:
        st.info("Enter at least two valid molecules.")

# ---------- Assessment ----------
BANK = {
    "Ethanol": "CCO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Benzene": "c1ccccc1",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
}
QUESTIONS = {
    "heavy atoms": lambda m: m.GetNumHeavyAtoms(),
    "rings": rdMolDescriptors.CalcNumRings,
    "hydrogen-bond donors": Lipinski.NumHDonors,
    "rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds,
}


def new_quiz(n=5):
    quiz = []
    for _ in range(n):
        name = random.choice(list(BANK))
        label = random.choice(list(QUESTIONS))
        ans = int(QUESTIONS[label](Chem.MolFromSmiles(BANK[name])))
        opts = {ans} | {max(0, ans + d) for d in (-2, -1, 1, 2, 3)}
        opts = sorted(opts)[:4] if ans not in sorted(opts)[:4] else sorted(opts)[:4]
        if ans not in opts:
            opts[-1] = ans
        quiz.append((name, label, ans, sorted(set(opts))))
    return quiz


with tab3:
    if "quiz" not in st.session_state:
        st.session_state.quiz = new_quiz()
    with st.form("quiz_form"):
        picks = []
        for i, (name, label, ans, opts) in enumerate(st.session_state.quiz):
            st.markdown(f"**Q{i + 1}. How many {label} does {name} have?**")
            st.image(svg_2d(Chem.MolFromSmiles(BANK[name]), 220, 160))
            picks.append(st.radio("Answer", opts, index=None, key=f"q{i}", horizontal=True))
        submitted = st.form_submit_button("Submit assessment")
    if submitted:
        score = sum(p == q[2] for p, q in zip(picks, st.session_state.quiz))
        total = len(picks)
        (st.success if score >= total * 0.6 else st.warning)(f"Score: {score} / {total}")
        for i, (p, q) in enumerate(zip(picks, st.session_state.quiz)):
            st.write(f"Q{i + 1}: {'✅' if p == q[2] else '❌'} correct answer = {q[2]}")
    if st.button("New assessment"):
        st.session_state.quiz = new_quiz()
        st.rerun()

st.divider()
st.caption("Created by K.Balashanmuga · AIML Student · SRM Institution of Science and Technology, Thiruchirappalli")
