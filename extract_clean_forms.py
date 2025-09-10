# extract_clean_forms.py

import pdfplumber, os, re, json

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_fields(text):
    common_keywords = ['Name', 'Father', 'DOB', 'Aadhar', 'Phone', 'Gender', 'Account', 'Address', 'Amount']
    found_fields = []
    for keyword in common_keywords:
        if re.search(fr'{keyword}[\s:]*', text, re.IGNORECASE):
            found_fields.append(keyword)
    return list(set(found_fields))

def process_forms(folder_path):
    structured_forms = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder_path, file)) as pdf:
                raw = "\n".join([p.extract_text() or '' for p in pdf.pages])
                cleaned = clean_text(raw)
                fields = extract_fields(cleaned)
                structured_forms.append({
                    "form_name": file,
                    "fields": [{"label": f, "type": "text"} for f in fields],
                    "raw_text": cleaned
                })
    return structured_forms

if __name__ == "__main__":
    forms = process_forms("downloaded_forms")
    with open("structured_forms.json", "w", encoding="utf-8") as f:
        json.dump(forms, f, indent=4, ensure_ascii=False)
    print("Forms structured and saved to structured_forms.json")
