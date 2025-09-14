# extract_nom_prenom_v2.py
import re
import unicodedata
import pandas as pd
from typing import Optional, Tuple, Set

# ==== CONFIG / listes utiles ====
TITLES = {"m", "mr", "mme", "madame", "monsieur", "mlle", "mme.", "m.", "dr", "pr", "prof"}
SKIP_TOKENS = {"de", "du", "la", "le", "les", "des", "d'", "bin", "ben", "ibn", "al", "el"}

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')  # permissive phone pattern
BRACKET_RE = re.compile(r'[\[\]\(\)]')
MULTISPACE = re.compile(r'\s+')

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

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
    (La fonction garde la même logique d'extraction qu'avant.)
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
                # rechercher nom à droite d'abord
                if idx+1 < len(words):
                    for j in range(idx+1, len(words)):
                        cand = words[j]
                        if cand.lower() not in SKIP_TOKENS and not is_title_token(cand):
                            nom = " ".join(words[j:])
                            return (prenom, nom, "élevé")
                # fallback: token précédent
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


def _load_prenoms(prenom_list_path: Optional[str], prenom_series: Optional[pd.Series]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """
    Retourne (prenoms_set, prenoms_set_noaccents) ou (None, None) si non fourni.
    Le fichier peut être :
     - CSV/Excel avec une colonne 'PRENOM'
     - CSV/texte simple (première colonne)
     - pandas.Series fournie directement
    """
    if prenom_series is not None:
        lst = prenom_series.dropna().astype(str).tolist()
    elif prenom_list_path is not None:
        try:
            # essayer CSV/Excel avec header; si 'PRENOM' existe on le prend
            tmp = pd.read_csv(prenom_list_path, dtype=str)
            if 'PRENOM' in tmp.columns:
                lst = tmp['PRENOM'].dropna().astype(str).tolist()
            else:
                # fallback: première colonne
                lst = tmp.iloc[:, 0].dropna().astype(str).tolist()
        except Exception:
            # si lecture csv échoue, tenter lecture en tant que txt simple
            try:
                with open(prenom_list_path, 'r', encoding='utf-8') as f:
                    lst = [line.strip() for line in f if line.strip()]
            except Exception:
                lst = []
    else:
        return None, None

    prenoms_set = {x.strip().lower() for x in lst if x and str(x).strip()}
    prenoms_set_noacc = {strip_accents(x) for x in prenoms_set}
    return prenoms_set, prenoms_set_noacc


def _prenom_matches(prenom: Optional[str], prenoms_set: Optional[Set[str]], prenoms_set_noacc: Optional[Set[str]]) -> bool:
    """
    Vérifie si une partie (token) du prenom existe dans prenoms_set.
    Utilise aussi une version sans accents pour améliorer matching.
    """
    if not prenom or (not prenoms_set and not prenoms_set_noacc):
        return False
    # split tokens sur espace, tiret, apostrophe, etc.
    tokens = re.split(r"[ \-'\u2019’]+", str(prenom))
    for t in tokens:
        t_clean = t.strip(" .").lower()
        if not t_clean:
            continue
        if prenoms_set and t_clean in prenoms_set:
            return True
        if prenoms_set_noacc and strip_accents(t_clean) in prenoms_set_noacc:
            return True
    return False


def process_excel(input_excel: str,
                  output_excel: Optional[str] = None,
                  prenom_list_path: Optional[str] = None,
                  prenom_series: Optional[pd.Series] = None,
                  sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Lit l'Excel input (colonne 'BENEFICIARES'), extrait NOM/PRENOM/CONFIDENCE,
    puis pour chaque ligne : si une partie du PRENOM extrait est trouvée dans la
    liste fournie, on ajoute un '+' à CONFIDENCE (ex: 'élevé' -> 'élevé+').
    On **ne supprime aucune ligne**.
    Retourne un DataFrame où l'ordre de colonnes inclut au minimum : NOM, PRENOM, CONFIDENCE.
    """
    # lecture du fichier principal
    df = pd.read_excel(input_excel, sheet_name=sheet_name) if sheet_name else pd.read_excel(input_excel)
    if "BENEFICIARES" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne appelée 'BENEFICIARES'")

    # charger la liste des prénoms (si fournie)
    prenoms_set, prenoms_set_noacc = _load_prenoms(prenom_list_path, prenom_series)

    # extraction
    results = df["BENEFICIARES"].astype(object).apply(lambda x: parse_name(x, prenoms_set))
    # parse_name renvoie (prenom, nom, confidence)
    df[["PRENOM", "NOM", "CONFIDENCE"]] = pd.DataFrame(results.tolist(), index=df.index)

    # pour chaque ligne : si une partie du PRENOM match la liste, ajouter '+'
    def update_conf(row):
        conf = row["CONFIDENCE"] or ""
        prenom = row["PRENOM"]
        if prenom and _prenom_matches(prenom, prenoms_set, prenoms_set_noacc):
            if not conf.endswith('+'):
                conf = conf + '+'
        return conf

    df["CONFIDENCE"] = df.apply(update_conf, axis=1)

    # Réordonner colonnes : on met BENEFICIARES (si présent), puis NOM, PRENOM, CONFIDENCE, puis le reste
    cols_rest = [c for c in df.columns if c not in ("BENEFICIARES", "NOM", "PRENOM", "CONFIDENCE")]
    new_order = []
    if "BENEFICIARES" in df.columns:
        new_order.append("BENEFICIARES")
    new_order += ["NOM", "PRENOM", "CONFIDENCE"] + cols_rest
    df = df[new_order]

    if output_excel:
        df.to_excel(output_excel, index=False)

    return df


# ==== exemple d'utilisation rapide ====
if __name__ == "__main__":
    try:
        # exemple : 'prenoms.csv' peut contenir une colonne 'PRENOM' ou être une liste en 1ère colonne
        out = process_excel("input.xlsx", output_excel="input_extracted.xlsx", prenom_list_path="prenoms.csv")
        print("Traitement terminé. Exemple lignes extraites :")
        print(out.head(20)[["BENEFICIARES","NOM","PRENOM","CONFIDENCE"]])
    except Exception as e:
        print("Erreur :", e)
