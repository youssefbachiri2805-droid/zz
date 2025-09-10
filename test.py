import pandas as pd
import re

def extraire_nom(beneficiaire: str, prenoms: set):
    """
    Extrait le nom (mot avant le prénom) si un prénom est trouvé dans la chaîne.
    """
    if pd.isna(beneficiaire):
        return None

    mots = re.findall(r"\b\w+\b", beneficiaire.upper())
    for i, mot in enumerate(mots):
        if mot in prenoms and i > 0:
            return mots[i-1]  # mot juste avant le prénom
    return None
