import pandas as pd

# 1. Lire les fichiers .DEL
prenoms_df = pd.read_csv("PRENOM.DEL", sep="|")
test_df = pd.read_csv("TEST.DEL", sep="|")

# Mettre les prénoms en minuscule pour faciliter la comparaison
prenoms_set = set(prenoms_df["PRENOM"].astype(str).str.lower())

# 2. Fonction pour extraire le nom
def extraire_nom(beneficiaire):
    if pd.isna(beneficiaire) or str(beneficiaire).strip() == "":
        return ""  # si vide → renvoie vide
    mots = str(beneficiaire).split()
    # Chercher s'il y a un prénom connu dans la chaîne
    for mot in mots:
        if mot.lower() in prenoms_set:
            # Le NOM = les autres mots que le prénom
            nom = " ".join([m for m in mots if m.lower() not in prenoms_set])
            return nom
    # Si aucun prénom trouvé → tout le contenu est NOM
    return str(beneficiaire)

# 3. Colonnes à traiter
colonnes_benef = ["BENEFICIAIRE", "BENEFICIAIRE 1", "BENEFICIAIRE 2", "BENEFICIAIRE 3"]

# 4. Appliquer le traitement à chaque colonne
for col in colonnes_benef:
    if col in test_df.columns:  # vérifier que la colonne existe
        new_col = f"NOM_{col.replace(' ', '')}"  # ex: BENEFICIAIRE 1 -> NOM_BENEFICIAIRE1
        test_df[new_col] = test_df[col].apply(extraire_nom)

# 5. Sauvegarder le résultat
test_df.to_csv("TEST.DEL", sep="|", index=False, quoting=1)

print("✅ Fichier TEST.DEL mis à jour avec les colonnes NOM_BENEFICIAIRE, NOM_BENEFICIAIRE1, NOM_BENEFICIAIRE2, NOM_BENEFICIAIRE3")
