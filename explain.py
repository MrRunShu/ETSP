import shap

def explain_with_shap(model, tokenizer, sample_text):
    explainer = shap.Explainer(model, tokenizer)
    inputs = tokenizer(sample_text, return_tensors="pt")
    shap_values = explainer(inputs)
    shap.plots.text(shap_values)
