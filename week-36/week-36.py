import regex as re
import pandas as pd


splits = {'train': 'train.parquet', 'validation': 'validation.parquet'}
df_train = pd.read_parquet("hf://datasets/coastalcph/tydi_xor_rc/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/coastalcph/tydi_xor_rc/" + splits["validation"])

#TO DO
#
#GENERAL
#✅ Shape
#✅ Word Count
#✅ Token Count
#Anything else?
#
#SPECIFIC (For each language)🙈🙈🙈🙈
# 5 Most common words + English translation
# Analyze type of words
# Rule based classifier (answerable or not)
# Performance Evaluation (answerable field) 

l = ["ar", "ko", "te"]
df_train = df_train[df_train["lang"].isin(l)]
df_val = df_val[df_val["lang"].isin(l)]


if False:
    print(f"Train shape: {df_train.shape}")
    print(f"Validation shape: {df_val.shape}")

    train_lan = df_train['lang'].value_counts()
    val_len = df_val['lang'].value_counts()
    print(train_lan)
    print(val_len)  

######### Word Count #########
ar, ko, te = df_train[df_train["lang"] == "ar"], df_train[df_train["lang"] == "ko"], df_train[df_train["lang"] == "te"]

def word_list(df):
    words = [re.findall(r'\w+', quest) for quest in df["question"]]
    
    return [w for q in words for w in q]

ar_words = word_list(ar)
ko_words = word_list(ko)
te_words = word_list(te)
print(f"Word counts for Arabic: {len(ar_words)}")
print(f"Word counts for Korean: {len(ko_words)}")
print(f"Word counts for Telugu: {len(te_words)}")
    
######### Token🙈Count #########
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def token_count(text):
    return len(tokenizer.tokenize(text))

results = []
for lang_name, df in [("ar", ar), ("ko", ko), ("te", te)]:
    results.append({
        "Language": lang_name,
        "Token Count": df["question"].apply(token_count).sum(),
    })

token_counts = pd.DataFrame(results)
print(token_counts)

# translate using nllb-200-distilled-600M🙈🙈🙈
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

ar = ar.sample(5)
print(ar["question"].tolist())
tokenizer.src_lang = "ara_Arab"
inputs = tokenizer(ar["question"].tolist(), return_tensors="pt", padding=True)
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# 5 Most common words 🙈
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display


ar_text = " ".join(ar_words)
reshaped_text = arabic_reshaper.reshape(ar_text)
bidi_text = get_display(reshaped_text)
ar_wordcloud = WordCloud(font_path='arial.ttf', background_color='white').generate(bidi_text)

ko_text = " ".join(ko_words)
ko_wordcloud = WordCloud(font_path='malgun.ttf', background_color='white').generate(ko_text)

te_text = " ".join(te_words)
te_wordcloud = WordCloud(font_path='gautami.ttf', background_color='white').generate(te_text)

plt.figure(figsize=(15, 5), dpi=600)

plt.subplot(1, 3, 1)
plt.imshow(ar_wordcloud, interpolation='bilinear')
plt.title('Arabic')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ko_wordcloud, interpolation='bilinear')
plt.title('Korean')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(te_wordcloud, interpolation='bilinear')
plt.title('Telugu')
plt.axis('off')

plt.show()

# Translate to 🙈 english


# Analyze the type of words 🙈 (wordcloud, pos🙈tagging, etc)


# MODEL

# import🙈torch
# from🙈transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 🙈model = AutoModelForQuestionAnswering.from_pretrained("🙈facebook/nllb-200-distilled-600M")
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-🙈200-distilled-600M")