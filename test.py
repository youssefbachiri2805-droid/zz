import pandas as pd

# 1. Lire les fichiers .DEL
prenoms_df = pd.read_csv("PRENOM.DEL", sep="|")
test_df = pd.read_csv("TEST.DEL", sep="|")

# Mettre les prénoms en minuscule pour faciliter la comparaison
prenoms_set = set(prenoms_df["PRENOM"].dropna().str.lower())

# 2. Fonction pour extraire le nom
def extraire_nom(beneficiaire):
    if pd.isna(beneficiaire):  # si la case est vide
        return ""
    mots = str(beneficiaire).split()
    # Chercher s'il y a un prénom connu dans la chaîne
    for mot in mots:
        if mot.lower() in prenoms_set:
            # Le NOM = les autres mots que le prénom
            nom = " ".join([m for m in mots if m.lower() not in prenoms_set])
            return nom
    # Si aucun prénom trouvé → tout le contenu est NOM
    return str(beneficiaire)

# 3. Appliquer sur la colonne BENEFICIAIRE
test_df["NOM"] = test_df["BENEFICIAIRE"].apply(extraire_nom)

# 4. Sauvegarder le résultat dans TEST.DEL (en réécrivant le fichier)
test_df.to_csv("TEST.DEL", sep="|", index=False, quoting=1)

print("✅ Fichier TEST.DEL mis à jour avec la colonne NOM")
