import os

def generate_sum(model, modelSection, modelinputType, evalSection, evalInputType):
    return os.system(f""" python3 generate.py --model {model} \
    --modelSection {modelSection} --modelinputType {modelinputType} \
    --evalSection {evalSection} --evalInputType {evalInputType} \
    """)

def main():
    models = [
        # 'bart-large', 
        'bart-large-cnn'
    ]
    experiments = [
        # ['abstract_kg', 'abstract_kg'],
        ['summary_kg', 'abstract_kg'],
        ['summary_kg', 'summary_kg'],
    ]
    exclude = ['bart-large_summary_kg_abstract_kg',
              'bart-large_summary_kg_summary_kg',
              'bart-large-cnn_abstract_kg_abstract_kg',]
    for model in models:
        for exp in experiments:
            modelInput = exp[0].split("_")
            evalInput = exp[1].split("_")
            if f"{model}_{exp[0]}_{exp[1]}" in exclude: continue
            generate_sum(
                model=model, 
                modelSection=modelInput[0], 
                modelinputType=modelInput[1], 
                evalSection=evalInput[0],  
                evalInputType=evalInput[1], 
            )
            
if __name__ == "__main__":
    main()