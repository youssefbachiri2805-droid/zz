import pandas as pd
import re

ef extraire_nom(beneficiaires, prenoms):
    """
    Extraire le nom (mot avant le prénom) pour chaque bénéficiaire.
    Input :
        beneficiaires : pd.Series de chaînes
        prenoms : pd.Series de prénoms
    Output :
        pd.Series avec le nom extrait ou None si pas trouvé
    """
    prenoms_set = set(prenoms.str.upper().str.strip())
    
    def _extract(beneficiaire):
        if pd.isna(beneficiaire):
            return None
        mots = re.findall(r"\b\w+\b", beneficiaire.upper())
        for i, mot in enumerate(mots):
            if mot in prenoms_set and i > 0:
                return mots[i-1]  # mot avant le prénom
        return None
    
    return beneficiaires.apply(_extract)

# --- Appel de la fonction comme tu veux ---
beneficiaires_df["Nom"] = extraire_nom(beneficiaires_df["beneficiaire"], prenoms_df["prenom"])
