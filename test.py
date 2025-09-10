import stanza
stanza.download('fr')  # télécharge le modèle français
nlp = stanza.Pipeline(lang='fr', processors='tokenize,ner')
doc = nlp("MME IMANE BACHIRI")
for ent in doc.ents:
    print(ent.text, ent.type)  # e.g. "Marie Curie PERSON"
    nom_famille = ent.text.split()[-1]
    print("Nom de famille :", nom_famille)  # => Curie
