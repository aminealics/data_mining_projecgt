"""Test de bout en bout de l'application (données SYNTHÉTIQUES, hors-ligne).

    USE_SYNTHETIC_DEMO=1 python tests/smoke_test.py
"""
import os, sys
os.environ["USE_SYNTHETIC_DEMO"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
assert not at.exception, at.exception
def go(page):
    at.sidebar.radio[0].set_value(page).run(); assert not at.exception, (page, at.exception)
def click(label):
    [b for b in at.button if b.label.startswith(label)][0].click().run(); assert not at.exception, (label, at.exception)

go("2 · Préparation"); click("Préparer")
print("prep ok:", [m.value for m in at.metric][:4])
go("3 · Modélisation"); click("Entraîner"); assert not at.exception
print("models ok:", len(at.dataframe))
go("4 · Évaluation"); print("eval ok")
go("5 · IA explicable")
click("Calculer SHAP"); print("shap global ok")
click("Calculer l'importance par permutation"); print("perm ok")
click("Expliquer cet individu"); print("local ok:", [m.value for m in at.metric][-6:])
go("6 · API (FastAPI)"); click("Exporter vers l'API"); print("api export ok")
click("Appeler POST /explain"); assert not at.exception; print("api call ok:", [s.value[:60] for s in at.success][-1:])
