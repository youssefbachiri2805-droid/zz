import pandas as pd
import re

def extraire_nom(beneficiaires, prenoms):
    """
    Extraire le nom (mot avant le prénom ou après une civilité).
    1. Si un prénom est détecté => renvoyer le mot avant le prénom
    2. Sinon, chercher une civilité (M., Mme, etc.) et renvoyer le mot qui suit
    """
    prenoms_set = set(prenoms.str.upper().str.strip())
    
    # Liste des civilités possibles (tu peux en rajouter si besoin)
    civilites = {
        "M.", "MR", "MONSIEUR",
        "MME", "MADAME",
        "MLLE", "MADEMOISELLE",
        "DR", "DOCTEUR",
        "PR", "PROFESSEUR",
        "ING", "INGENIEUR",
    }

    def _extract(beneficiaire):
        if pd.isna(beneficiaire):
            return None
        
        mots = re.findall(r"\b\w+\b", beneficiaire.upper())
        
        # 1. Chercher le prénom
        for i, mot in enumerate(mots):
            if mot in prenoms_set and i > 0:
                return mots[i-1]  # mot avant le prénom
        
        # 2. Chercher la civilité
        for i, mot in enumerate(mots):
            if mot in civilites and i+1 < len(mots):
                return mots[i+1]  # mot après la civilité
        
        return None
    
    return beneficiaires.apply(_extract)
