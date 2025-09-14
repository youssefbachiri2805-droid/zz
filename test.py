# extract_nom_prenom.py
import re
import pandas as pd
from typing import Optional, Tuple, Set

# ==== CONFIG / listes utiles ====
TITLES = {"m", "mr", "mme", "madame", "monsieur", "mlle", "mme.", "m.", "dr", "pr", "prof"}
SKIP_TOKENS = {"de", "du", "la", "le", "les", "des", "d'", "bin", "ben", "ibn", "al", "el"}

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')
BRACKET_RE = re.compile(r'[\[\]\(\)]')
MULTISPACE = re.compile(r'\s+')

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = EMAIL_RE.sub(' ', s)         # enlever emails
    s = PHONE_RE.sub(' ', s)         # enlever numéros
    s = BRACKET_RE.sub(' ', s)       # remplacer parenthèses/brackets
    s = re.sub(r'[\/|;,_]+', ' ', s) # uniformiser séparateurs
    s = MULTISPACE.sub(' ', s)
    return s.strip()

def split_tokens(s: str):
    parts = re.split(r'[,\n;/|]+', s)
    words = []
    for p in parts:
        for w in p.strip().split():
            w = w.strip(" .:")
            if w:
                words.append(w)
    return words

def is_title_token(tok: str) -> bool:
    return tok.lower().replace('.', '') in TITLES

def probable_name_from_label(s: str) -> Optional[Tuple[str,str]]:
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

    explicit = probable_name_from_label(text)
    if explicit:
        prenom, nom = explicit
        conf = "élevé" if prenom or nom else "faible"
        return (prenom if prenom else None, nom if nom else None, conf)

    words = split_tokens(s)
    if not words:
        return (None, None, "faible")

    if ',' in text:
        left, right = text.split(',', 1)
        left = normalize(left)
        right = normalize(right)
        left_words = split_tokens(left)
        right_words = split_tokens(right)
        if right_words:
            prenom = right_words[0]
            nom = " ".join(left_words) if left_words else None
            return (prenom, nom, "élevé")

    if prenoms_set:
        lower_tokens = [w.lower() for w in words]
        for idx, lt in enumerate(lower_tokens):
            if lt in prenoms_set:
                prenom = words[idx]
                remaining = [w for i,w in enumerate(words) if i!=idx and not is_title_token(w)]
                if idx+1 < len(words):
                    for j in range(idx+1, len(words)):
                        cand = words[j]
                        if cand.lower() not in SKIP_TOKENS and not is_title_token(cand):
                            nom = " ".join(words[j:])
                            return (prenom, nom, "élevé")
                if idx-1 >= 0:
                    nom = words[idx-1]
                    return (prenom, nom, "moyen")
                return (prenom, None, "moyen")

    if is_title_token(words[0]):
        if len(words) >= 3:
            prenom = words[1]
            nom = " ".join(words[2:])
            return (prenom, nom, "élevé")
        elif len(words) == 2:
            prenom = words[1]
            return (prenom, None, "moyen")

    if len(words) == 2:
        return (words[0], words[1], "moyen")

    if len(words) >= 3:
        prenom = words[0]
        nom = " ".join(words[1:])
        return (prenom, nom, "faible")

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
    Lit le fichier Excel, traite la colonne 'BENEFICIARES' et retourne un DataFrame filtré :
    - garde confiance élevée
    - garde confiance moyenne seulement si le prénom est valide
    - enlève confiance faible
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name) if sheet_name else pd.read_excel(input_excel)
    if "BENEFICIARES" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne appelée 'BENEFICIARES'")

    prenoms_set = None
    if prenom_series is not None:
        prenoms_set = {str(x).lower() for x in prenom_series.dropna().astype(str)}
    elif prenom_list_path is not None:
        try:
            tmp = pd.read_csv(prenom_list_path, header=None)
            lst = tmp.iloc[:,0].astype(str).tolist()
        except Exception:
            with open(prenom_list_path, 'r', encoding='utf-8') as f:
                lst = [line.strip() for line in f if line.strip()]
        prenoms_set = {x.lower() for x in lst}

    results = df["BENEFICIARES"].astype(object).apply(lambda x: parse_name(x, prenoms_set))
    df[["PRENOM","NOM","CONFIDENCE"]] = pd.DataFrame(results.tolist(), index=df.index)

    # === Filtrage final ===
    def filter_row(row):
        if row["CONFIDENCE"] == "élevé":
            return True
        elif row["CONFIDENCE"] == "moyen":
            if prenoms_set and row["PRENOM"] and row["PRENOM"].lower() in prenoms_set:
                return True
            else:
                return False
        else:  # faible
            return False

    df = df[df.apply(filter_row, axis=1)].reset_index(drop=True)

    if output_excel:
        df.to_excel(output_excel, index=False)

    return df


# ==== exemple d'utilisation rapide ====
if __name__ == "__main__":
    try:
        out = process_excel("input.xlsx", output_excel="input_extracted.xlsx", prenom_list_path="prenoms.csv")
        print("Traitement terminé. Exemple lignes extraites :")
        print(out[["BENEFICIARES","PRENOM","NOM","CONFIDENCE"]].head(20))
    except Exception as e:
        print("Erreur :", e)
