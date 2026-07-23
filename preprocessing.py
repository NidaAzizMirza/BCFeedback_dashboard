import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import demoji                       # for emoji handling
import string

def run_preprocessing(file_path, sheet_name="Sheet"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = preprocessing_first(df)
    return df


domain_words = {
    "curtilage", "levy", "pdf", "pdfs", "planning",
    "application", "submission", "upload", "uploads",
    "portal", "document", "documents", "fee", "fees",
    "validation", "dwelling", "BNG", "bng", "10mb", "10 mb", "mb", "kb", "gb", "lpa"
}

spell = SpellChecker()
spell.word_frequency.load_words(domain_words)



# file_path = "/Users/nidaaziz/Desktop/TQ/Feedback tool/Apr20th.xlsx"


def clean_whitespace(text):
    text = re.sub(r'\s+', ' ', text)    # replace any whitespace sequence with a single space
    return text.strip()

def clean_html(text):
    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text(separator=" ", strip=True)


def correct_spelling(text):
    words = text.split()
    corrected = []

    for w in words:
        w_clean = w.lower()

        # Skip very short words (performance boost)
        if len(w_clean) <= 2:
            corrected.append(w)
            continue

        # Skip domain-specific words
        if w_clean in domain_words:
            corrected.append(w)
            continue

        # Skip correct words
        if w_clean in spell.word_frequency:
            corrected.append(w)
        else:
            corrected.append(spell.correction(w) or w)

    return ' '.join(corrected)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))



def pre_preprocessing(data, has_extra_header_row=True):
    # Use hardcoded column name — preprocessing.py doesn't import cfg
    respondent_col = "Respondent ID"
    print(f"Row 0 Respondent ID: {data.iloc[0][respondent_col] if len(data) > 0 else 'empty'}")
    print(f"Row 1 Respondent ID: {data.iloc[1][respondent_col] if len(data) > 1 else 'empty'}")

    # SurveyMonkey's Excel export has a spurious second header row
    # (sub-labels), which this drops. Data pulled directly from the API
    # is already clean, so has_extra_header_row=False skips this.
    if has_extra_header_row:
        data = data.iloc[1:].reset_index(drop=True)
    print(data.columns.tolist())
    #change name of Ratings column to Rating
    data.rename(columns={"Rate your experience": "Rating"}, inplace=True)
    rating_map = {"Excellent": 5, "Poor": 1}
    data["Rating"] = data["Rating"].map(rating_map).fillna(
        pd.to_numeric(data["Rating"], errors="coerce")
    )

    print(f"After rating map: {len(data)} rows, non-null ratings: {data['Rating'].notna().sum()}")

    data = data.rename(columns={
        "Is there anything you would like to tell us about your experience?": "Feedback"
    })
    print(f"Feedback non-null: {data['Feedback'].notna().sum()}")
    return data


# def preprocessing_first(df):
#     data = pre_preprocessing(df)
#     data = data.dropna(subset=['Feedback'])
#     data = data[data['Feedback'].str.strip() != '']
#     data['Feedback'] = data['Feedback'].str.lower()
#     data = data.reset_index(drop=True)
#
#     data['Feedback_clean'] = data['Feedback'].apply(clean_whitespace)
#     data['Feedback_clean'] = data['Feedback_clean'].apply(clean_html)
#
#     data['Feedback_clean'] = data['Feedback_clean'].apply(
#         lambda x: demoji.replace_with_desc(x, sep=' ')
#     )
#
#     data['Feedback_clean'] = data['Feedback_clean'].apply(remove_punctuation)
#     # Optional slow step
#     data['Feedback_clean'] = data['Feedback_clean'].apply(correct_spelling)
#
#     return data

def preprocessing_first(df, has_extra_header_row=True):
    data = pre_preprocessing(df, has_extra_header_row=has_extra_header_row)
    data = data.dropna(subset=['Feedback'])

    # Some respondents leave a purely numeric comment (e.g. just typing a
    # number), and Excel/pandas reads that cell as a float, not a string.
    # dropna() above only catches truly missing values — a real float like
    # 500.0 sails through it. Then `.str.strip()` on that float silently
    # returns NaN (not an error), and `NaN != ''` evaluates to True, so the
    # row survives the empty-string filter below and crashes several lines
    # later when clean_whitespace() calls re.sub() on a non-string value.
    # Casting to str here — after the real-NaN drop, before any .str use —
    # closes that gap for good.
    data['Feedback'] = data['Feedback'].astype(str)

    data = data[data['Feedback'].str.strip() != '']
    data['Feedback'] = data['Feedback'].str.lower()
    data = data.reset_index(drop=True)

    data['Feedback_clean'] = data['Feedback'].apply(clean_whitespace)
    data['Feedback_clean'] = data['Feedback_clean'].apply(clean_html)
    data['Feedback_clean'] = data['Feedback_clean'].apply(
        lambda x: demoji.replace_with_desc(x, sep=' ')
    )
    data['Feedback_clean'] = data['Feedback_clean'].apply(remove_punctuation)
    # data['Feedback_clean'] = data['Feedback_clean'].apply(correct_spelling)
    # ↑ commented out — causing data loss in pipeline context

    # Print output
    # print("\nProcessed Data Preview:")
    # print(data.head(500))

    return data

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# INPUT_FILE        = os.path.join(BASE_DIR, "inputs", "new_export.xlsx")
# INPUT_SHEET       = "Sheet"
# run_preprocessing(INPUT_FILE)