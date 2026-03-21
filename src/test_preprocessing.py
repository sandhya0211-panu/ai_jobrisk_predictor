from preprocessing import load_data, preprocess_data

df = load_data("data/ai_job_replacement_2020_2026_v2.csv")

df = preprocess_data(df)

print(df.shape)

print(df.head())