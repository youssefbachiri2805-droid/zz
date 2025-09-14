# extract_nom_prenom_multiple.py
import re
import unicodedata
import pandas as pd
from typing import Optional, Tuple, Set

# construction du NOM en retirant les titles
def clean_nom(nom: Optional[str]) -> Optional[str]:
    if not nom:
        return None
    words = [w for w in nom.split() if not is_title_token(w)]
    if not words:
        return None
    return " ".join(words)


# ==== CONFIG / listes utiles ====
TITLES = {"m", "mr", "mme", "madame", "monsieur", "mlle", "mme.", "m.", "dr", "pr", "prof"}
SKIP_TOKENS = {"de", "du", "la", "le", "les", "des", "d'", "bin", "ben", "ibn", "al", "el"}
COMPANY_WORDS = {"SAS","SARL","SA","CBT","SCI","SERM","SOC","PSF","SCCV","C","ETC","SYND","CAB","EURL","AUTO"}

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')
BRACKET_RE = re.compile(r'[\[\]\(\)]')
MULTISPACE = re.compile(r'\s+')
NUM_RE = re.compile(r'\d+')

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = EMAIL_RE.sub(' ', s)
    s = PHONE_RE.sub(' ', s)
    s = BRACKET_RE.sub(' ', s)
    s = re.sub(r'[\/|;,_]+', ' ', s)
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
    s = normalize(text)
    if not s:
        return (None, None, "faible")

    if any(w.upper() in COMPANY_WORDS for w in s.split()):
        return (None, text.strip(), "faible")

    if NUM_RE.search(text):
        return (None, text.strip(), "faible")

    explicit = probable_name_from_label(text)
    if explicit:
        prenom, nom = explicit
        conf = "élevé" if prenom or nom else "faible"
        return (prenom if prenom else None, nom if nom else None, conf)

    words = split_tokens(s)
    if not words:
        return (None, None, "faible")

    last_title_idx = -1
    for i, w in enumerate(words):
        if is_title_token(w):
            last_title_idx = i
    if last_title_idx != -1 and last_title_idx < len(words)-1:
        nom = words[last_title_idx+1]
        prenom = " ".join(words[last_title_idx+2:]) if len(words) > last_title_idx+2 else None
        return (prenom, nom, "élevé")

    if prenoms_set:
        lower_tokens = [w.lower() for w in words]
        for idx, lt in enumerate(lower_tokens):
            if lt in prenoms_set:
                prenom = words[idx]
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

    if len(words) == 2:
        return (words[0], words[1], "moyen")
    if len(words) >= 3:
        prenom = words[0]
        nom = " ".join(words[1:])
        return (prenom, nom, "moyen")
    if len(words) == 1:
        return (None, words[0], "faible")

    return (None, None, "faible")

def _load_prenoms(prenom_list_path: Optional[str], prenom_series: Optional[pd.Series]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    if prenom_series is not None:
        lst = prenom_series.dropna().astype(str).tolist()
    elif prenom_list_path is not None:
        try:
            tmp = pd.read_csv(prenom_list_path, dtype=str)
            if 'PRENOM' in tmp.columns:
                lst = tmp['PRENOM'].dropna().astype(str).tolist()
            else:
                lst = tmp.iloc[:, 0].dropna().astype(str).tolist()
        except Exception:
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
    if not prenom or (not prenoms_set and not prenoms_set_noacc):
        return False
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

def process_excel_multi(input_excel: str,
                        output_excel: Optional[str] = None,
                        prenom_list_path: Optional[str] = None,
                        prenom_series: Optional[pd.Series] = None,
                        sheet_name: Optional[str] = None) -> pd.DataFrame:

    df = pd.read_excel(input_excel, sheet_name=sheet_name) if sheet_name else pd.read_excel(input_excel)
    prenoms_set, prenoms_set_noacc = _load_prenoms(prenom_list_path, prenom_series)

    cols = ["BENEFICIAIRE 1", "BENEFICIAIRE 2", "BENEFICIAIRE 3"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Le fichier Excel doit contenir une colonne appelée '{col}'")
        results = df[col].astype(object).apply(lambda x: parse_name(x, prenoms_set))
        df[[f"PRENOM_{col}", f"NOM_{col}", f"CONFIDENCE_{col}"]] = pd.DataFrame(results.tolist(), index=df.index)

        # Ajouter + si prénom trouvé
        def update_conf(row):
            conf = row[f"CONFIDENCE_{col}"]
            prenom = row[f"PRENOM_{col}"]
            if prenom and _prenom_matches(prenom, prenoms_set, prenoms_set_noacc):
                if conf.lower().startswith("élevé") and not conf.endswith('+'):
                    return "SURE"
                elif conf.lower().startswith("moyen") and not conf.endswith('+'):
                    return "CRITIQUE"
            return conf
        df[f"CONFIDENCE_{col}"] = df.apply(update_conf, axis=1)

        # TRAITEMENT FAIBLE et MOYEN
        mask_faible_moyen = df[f"CONFIDENCE_{col}"].str.lower().str.startswith(("faible","moyen"))
        df.loc[mask_faible_moyen, f"NOM_{col}"] = df.loc[mask_faible_moyen, col]
        df.loc[mask_faible_moyen, f"PRENOM_{col}"] = None
        df.loc[mask_faible_moyen & df[f"CONFIDENCE_{col}"].str.lower().str.startswith("moyen"), f"CONFIDENCE_{col}"] = "élevé"
        df.loc[mask_faible_moyen & df[f"CONFIDENCE_{col}"].str.lower().str.startswith("faible"), f"CONFIDENCE_{col}"] = "SURE"

        # Vérification longueur NOM >=3 lettres
        df[f"NOM_{col}"] = df[f"NOM_{col}"].apply(lambda x: x if x and len(str(x).strip()) >=3 else None)

        # Supprimer PRENOM intermédiaire
        df = df.drop(columns=[f"PRENOM_{col}"])

    if output_excel:
        df.to_excel(output_excel, index=False)

    return df


# ==== Exemple d'utilisation ====
if __name__ == "__main__":
    try:
        out = process_excel_multi("input.xlsx", output_excel="input_extracted.xlsx", prenom_list_path="prenoms.csv")
        print("Traitement terminé. Exemple lignes extraites :")
        print(out.head(20))
    except Exception as e:
        print("Erreur :", e)
