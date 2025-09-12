# extract_nom_prenom.py
import re
import pandas as pd
from typing import Optional, Tuple, Set

# ==== CONFIG / listes utiles ====
TITLES = {"m", "mr", "mme", "madame", "monsieur", "mlle", "mme.", "m.", "dr", "pr", "prof"}
SKIP_TOKENS = {"de", "du", "la", "le", "les", "des", "d'", "bin", "ben", "ibn", "al", "el"}

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')  # permissive phone pattern
BRACKET_RE = re.compile(r'[\[\]\(\)]')
MULTISPACE = re.compile(r'\s+')

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = EMAIL_RE.sub(' ', s)         # enlever emails
    s = PHONE_RE.sub(' ', s)         # enlever numéros
    s = BRACKET_RE.sub(' ', s)       # remplacer parenteses/brackets par espace
    # remplace slash/pipe/; , - multiples par un séparateur uniforme
    s = re.sub(r'[\/|;,_]+', ' ', s)
    s = MULTISPACE.sub(' ', s)
    return s.strip()

def split_tokens(s: str):
    # split by common separators then by spaces, keep order
    parts = re.split(r'[,\n;/|]+', s)
    words = []
    for p in parts:
        for w in p.strip().split():
            w = w.strip(" .:")  # remove trailing dots/colons
            if w:
                words.append(w)
    return words

def is_title_token(tok: str) -> bool:
    return tok.lower().replace('.', '') in TITLES

def probable_name_from_label(s: str) -> Optional[Tuple[str,str]]:
    # Gère les formats "NOM: X", "PRENOM: Y" etc
    # cherche explicitement 'prenom' ou 'nom'
    m = re.search(r'prenom[:\s\-]+([A-Za-zÀ-ÿ\'\- ]{1,80})', s, flags=re.I)
    if m:
        prenom = m.group(1).strip().split()[0]
        nom = None
        m2 = re.search(r'nom[:\s\-]+([A-Za-zÀ-ÿ\'\- ]{1,80})', s, flags=re.I)
        if m2:
            nom = m2.group(1).strip().split()[0]
        return (prenom, nom)
    m = re.search(r'nom[:\s\-]+([A-Za-zÀ-ÿ\'\- ]{1,80})', s, flags=re.I)
    if m:
        nom = m.group(1).strip().split()[0]
        return (None, nom)
    return None

def parse_name(text: str, prenoms_set: Optional[Set[str]] = None) -> Tuple[Optional[str], Optional[str], str]:
    """
    Retourne (prenom, nom, confidence) où confidence in {"faible","moyen","élevé"}.
    """
    s = normalize(text)
    if not s:
        return (None, None, "faible")

    # cas explicite
    explicit = probable_name_from_label(text)
    if explicit:
        prenom, nom = explicit
        conf = "élevé" if prenom or nom else "faible"
        return (prenom if prenom else None, nom if nom else None, conf)

    words = split_tokens(s)
    if not words:
        return (None, None, "faible")

    # si forme "Last, First" -> detecte la virgule originelle (avant normalize on peut tester)
    if ',' in text:
        # traiter "Dupont, Jean" ou "DUPONT, JEAN PIERRE"
        left, right = text.split(',', 1)
        left = normalize(left)
        right = normalize(right)
        left_words = split_tokens(left)
        right_words = split_tokens(right)
        if right_words:
            prenom = right_words[0]
            nom = " ".join(left_words) if left_words else None
            return (prenom, nom, "élevé")

    # token based matching with prenom list (fortement conseillé de fournir la liste)
    if prenoms_set:
        # recherche première occurrence d'un token qui est un prénom connu
        lower_tokens = [w.lower() for w in words]
        for idx, lt in enumerate(lower_tokens):
            if lt in prenoms_set:
                # heuristique : prenom = token idx
                prenom = words[idx]
                # chercher nom à droite (préfère les tokens non stop)
                # assemble tout sauf tokens de titre et mots vides
                remaining = [w for i,w in enumerate(words) if i!=idx and not is_title_token(w)]
                # prefer name on right of prenom
                if idx+1 < len(words):
                    # choose next not-stop token as nom
                    for j in range(idx+1, len(words)):
                        cand = words[j]
                        if cand.lower() not in SKIP_TOKENS and not is_title_token(cand):
                            nom = " ".join(words[j:])  # prend le reste comme nom
                            return (prenom, nom, "élevé")
                # fallback: take left token(s)
                if idx-1 >= 0:
                    nom = words[idx-1]
                    return (prenom, nom, "moyen")
                # si aucun voisin -> seul prénom
                return (prenom, None, "moyen")

    # gestion des titres comme "M. Jean Dupont" ou "Mme Marie-Claire Legrand"
    if is_title_token(words[0]):
        # skip titre
        if len(words) >= 3:
            prenom = words[1]
            nom = " ".join(words[2:])
            return (prenom, nom, "élevé")
        elif len(words) == 2:
            prenom = words[1]
            return (prenom, None, "moyen")

    # si exactement 2 mots, on assume prénom NOM (ou NOM prénom?) -> on choisit prénom premier (prédominant)
    if len(words) == 2:
        return (words[0], words[1], "moyen")

    # pour >2 mots et pas de prénom dans liste : préférence -> premier = prénom, reste = nom
    if len(words) >= 3:
        prenom = words[0]
        nom = " ".join(words[1:])
        return (prenom, nom, "faible")

    # 1 mot -> on met dans NOM (si tu préfères le contraire, change la règle)
    if len(words) == 1:
        return (None, words[0], "faible")

    return (None, None, "faible")


# ==== fonction principale pour fichier Excel ====
def process_excel(input_excel: str,
                  output_excel: Optional[str] = None,
                  prenom_list_path: Optional[str] = None,
                  prenom_series: Optional[pd.Series] = None,
                  sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Lit le fichier Excel, traite la colonne 'BENEFICIARES' et retourne un DataFrame avec colonnes PRENOM, NOM, CONFIDENCE.
    - prenom_list_path : chemin vers un fichier texte/CSV contenant une colonne ou liste de prénoms (optionnel)
    - prenom_series : Series pandas déjà chargée si tu as la liste en mémoire
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name) if sheet_name else pd.read_excel(input_excel)
    if "BENEFICIARES" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne appelée 'BENEFICIARES'")

    prenoms_set = None
    if prenom_series is not None:
        prenoms_set = {str(x).lower() for x in prenom_series.dropna().astype(str)}
    elif prenom_list_path is not None:
        # on accepte csv ou txt (une colonne)
        try:
            # tentative CSV
            tmp = pd.read_csv(prenom_list_path, header=None)
            lst = tmp.iloc[:,0].astype(str).tolist()
        except Exception:
            with open(prenom_list_path, 'r', encoding='utf-8') as f:
                lst = [line.strip() for line in f if line.strip()]
        prenoms_set = {x.lower() for x in lst}

    # applique parse_name
    results = df["BENEFICIARES"].astype(object).apply(lambda x: parse_name(x, prenoms_set))
    df[["PRENOM","NOM","CONFIDENCE"]] = pd.DataFrame(results.tolist(), index=df.index)
    if output_excel:
        df.to_excel(output_excel, index=False)
    return df

# ==== exemple d'utilisation rapide ====
if __name__ == "__main__":
    # Exemple : python extract_nom_prenom.py
    # remplacer 'input.xlsx' par ton fichier
    try:
        out = process_excel("input.xlsx", output_excel="input_extracted.xlsx")
        print("Traitement terminé. Exemple lignes extraites :")
        print(out[["BENEFICIARES","PRENOM","NOM","CONFIDENCE"]].head(20))
    except Exception as e:
        print("Erreur :", e)
