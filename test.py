import pandas as pd
import re

def extraire_nom(beneficiaires, prenoms):
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

# --- Lecture des fichiers DEL ---
# ⚠️ Remplace "beneficiaires.DEL" et "prenoms.DEL" par tes fichiers réels
beneficiaires_df = pd.read_csv("beneficiaires.DEL", sep="|", quotechar='"', dtype=str)
prenoms_df = pd.read_csv("prenoms.DEL", sep="|", quotechar='"', dtype=str)

# --- Application de la fonction ---
beneficiaires_df["Nom"] = extraire_nom(beneficiaires_df["beneficiaire"], prenoms_df["prenom"])

# --- Sauvegarde en fichier DEL ---
# ⚠️ Choisis le nom que tu veux pour la sortie
beneficiaires_df.to_csv("beneficiaires_output.DEL", sep="|", index=False, quoting=1)  # quoting=1 → csv.QUOTE_ALL
