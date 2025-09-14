# extract_nom_prenom.py
import re
import pandas as pd
from typing import Optional, Tuple, Set
import argparse
import sys

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
    s = BRACKET_RE.sub(' ', s)       # remplacer parentheses/brackets par espace
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

# ==== utilitaires pour charger les listes de prénoms ====
def load_prenom_set(prenom_series: Optional[pd.Series] = None, prenom_list_path: Optional[str] = None) -> Optional[Set[str]]:
    """
    Charge une liste/colonne de prénoms et renvoie un set en lowercase.
    Accepté: pd.Series, CSV, Excel, TXT (une colonne par ligne).
    Pour CSV/Excel, si une colonne s'appelle 'PRENOM' (insensible à la casse) elle sera utilisée,
    sinon la première colonne sera utilisée.
    """
    if prenom_series is not None:
        return {str(x).strip().lower() for x in prenom_series.dropna().astype(str)}
    if prenom_list_path is None:
        return None

    # tenter CSV/Excel puis fallback txt simple
    try:
        tmp = pd.read_csv(prenom_list_path, dtype=str, encoding='utf-8', engine='python')
        cols = [c for c in tmp.columns if str(c).strip().lower() == 'prenom']
        if cols:
            lst = tmp[cols[0]].dropna().astype(str).tolist()
        else:
            lst = tmp.iloc[:,0].dropna().astype(str).tolist()
    except Exception:
        try:
            tmp = pd.read_excel(prenom_list_path, dtype=str)
            cols = [c for c in tmp.columns if str(c).strip().lower() == 'prenom']
            if cols:
                lst = tmp[cols[0]].dropna().astype(str).tolist()
            else:
                lst = tmp.iloc[:,0].dropna().astype(str).tolist()
        except Exception:
            # txt fallback
            with open(prenom_list_path, 'r', encoding='utf-8') as f:
                lst = [line.strip() for line in f if line.strip()]
    return {x.strip().lower() for x in lst}

# ==== fonction principale pour fichier Excel ====
def process_excel(input_excel: str,
                  output_excel: Optional[str] = None,
                  prenom_list_path: Optional[str] = None,
                  prenom_series: Optional[pd.Series] = None,
                  prenom_verif_list_path: Optional[str] = None,
                  prenom_verif_series: Optional[pd.Series] = None,
                  sheet_name: Optional[str] = None,
                  drop_faible: bool = True) -> pd.DataFrame:
    """
    Lit le fichier Excel, traite la colonne 'BENEFICIARES' et retourne un DataFrame avec colonnes PRENOM, NOM, CONFIDENCE.
    - prenom_list_path / prenom_series : (optionnel) liste utilisée par parse_name pour détecter les prénoms (augmente la précision).
    - prenom_verif_list_path / prenom_verif_series : (optionnel) liste de référence (colonne PRENOM) utilisée pour vérifier les cas 'moyen'.
      Si non fourni mais que prenom_list_path est fourni, la même liste sera utilisée comme vérification.
    - drop_faible : si True (par défaut) on supprime les lignes avec CONFIDENCE == 'faible' (règle demandée).
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name) if sheet_name else pd.read_excel(input_excel)
    if "BENEFICIARES" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne appelée 'BENEFICIARES'")

    # charger sets de prénoms
    prenoms_set_for_parse = load_prenom_set(prenom_series, prenom_list_path)
    prenoms_verif_set = load_prenom_set(prenom_verif_series, prenom_verif_list_path)

    # fallback : si pas de verif fournie, utiliser la première liste si disponible
    if prenoms_verif_set is None and prenoms_set_for_parse is not None:
        prenoms_verif_set = prenoms_set_for_parse

    # appliquer parse_name
    results = df["BENEFICIARES"].astype(object).apply(lambda x: parse_name(x, prenoms_set_for_parse))
    df[["PRENOM","NOM","CONFIDENCE"]] = pd.DataFrame(results.tolist(), index=df.index)

    initial_count = len(df)

    # supprimer 'faible' si demandé
    if drop_faible:
        df = df[df["CONFIDENCE"].str.lower() != 'faible']
        removed_faible = initial_count - len(df)
    else:
        removed_faible = 0

    # traiter les cas 'moyen' : garder uniquement si PRENOM est dans la liste de vérification
    # si on a des cas 'moyen' mais aucune liste fournie, c'est impossible d'appliquer la règle strictement -> on lève une erreur
    mask_moyen = df["CONFIDENCE"].str.lower() == 'moyen'
    if mask_moyen.any():
        if prenoms_verif_set is None:
            raise ValueError(
                "Il y a des cas avec CONFIDENCE == 'moyen' mais aucune liste de prénoms de vérification fournie. "
                "Passe 'prenom_verif_list_path' ou 'prenom_verif_series' pour appliquer la règle de filtrage."
            )
        # garder les 'moyen' uniquement si PRENOM existe dans prenoms_verif_set
        def prenom_valide(pr):
            if pr is None:
                return False
            s = str(pr).strip().lower()
            if not s:
                return False
            return s in prenoms_verif_set

        keep_mask = (~mask_moyen) | df["PRENOM"].apply(prenom_valide)
        before_moyen = mask_moyen.sum()
        after_moyen_kept = df[keep_mask]["CONFIDENCE"].str.lower().eq('moyen').sum()
        df = df[keep_mask]
        removed_moyen = before_moyen - after_moyen_kept
    else:
        removed_moyen = 0

    final_count = len(df)

    # sauvegarde si demandé
    if output_excel:
        df.to_excel(output_excel, index=False)

    # résumé (affiché)
    print(f"Total lignes initiales : {initial_count}")
    print(f"Lignes supprimées (confidence 'faible') : {removed_faible}")
    print(f"Lignes supprimées (confidence 'moyen' non vérifiées) : {removed_moyen}")
    print(f"Lignes finales retenues : {final_count}")

    return df

# ==== CLI simple ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraire prénom/nom depuis une colonne 'BENEFICIARES' d'un fichier Excel.")
    parser.add_argument("-i", "--input", required=True, help="Fichier Excel d'entrée (doit contenir la colonne 'BENEFICIARES').")
    parser.add_argument("-o", "--output", default=None, help="Fichier Excel de sortie (optionnel).")
    parser.add_argument("--prenom-list", default=None, help="Chemin vers fichier (CSV/Excel/txt) contenant prénoms (optionnel, utilisé par parse_name).")
    parser.add_argument("--prenom-verif-list", default=None, help="Chemin vers fichier (CSV/Excel/txt) contenant la colonne PRENOM utilisée pour vérifier les cas 'moyen'.")
    parser.add_argument("--sheet", default=None, help="Nom de la feuille Excel si nécessaire.")
    args = parser.parse_args()

    try:
        out = process_excel(
            args.input,
            output_excel=args.output,
            prenom_list_path=args.prenom_list,
            prenom_verif_list_path=args.prenom_verif_list,
            sheet_name=args.sheet
        )
        print("Traitement terminé. Exemple lignes extraites :")
        print(out[["BENEFICIARES","PRENOM","NOM","CONFIDENCE"]].head(20))
    except Exception as e:
        print("Erreur :", e)
        sys.exit(1)



python extract_nom_prenom.py -i input.xlsx -o output.xlsx --prenom-list prenoms_connus.csv --prenom-verif-list prenoms_master.csv

