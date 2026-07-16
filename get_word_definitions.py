import requests
from constants import ALL_LANGUAGES
from tqdm import tqdm
import json

q_limit = 300
samples_per_lang = 5
headers = {'User-Agent': 'DefinitonCrawler/0.1 (https://ufal.mff.cuni.cz/adnan-al-ali; alali@ufal.mff.cuni.cz)'}

def_dict = dict()

for lang in tqdm(ALL_LANGUAGES):
    defs = []
    for qid in range(1, q_limit + 1):
        label = requests.get(f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/Q{qid}/labels/{lang}", headers=headers)
        desc = requests.get(f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/Q{qid}/descriptions/{lang}", headers=headers)
        
        if label.ok and desc.ok and label.text and desc.text and len(json.loads(desc.text)) > 10:
            defs.append({
                "label": json.loads(label.text),
                "description": json.loads(desc.text),
                "qid": qid
            })
            if len(defs) == samples_per_lang:
                def_dict[lang] = defs
                break
        else:
            print(label.text)

with open("definitions.json", "w") as out_file:
    json.dump(def_dict, out_file, ensure_ascii=False, indent=True)