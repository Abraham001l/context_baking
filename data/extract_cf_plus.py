import json

def generate_counterfact_plus():
    # 1. Load the base downloaded dataset
    with open('counterfact.json', 'r') as f:
        base_data = json.load(f)

    extracted_data = []

    for record in base_data:
        # Extract the core edit components
        subject = record['requested_rewrite']['subject']
        prompt_template = record['requested_rewrite']['prompt']
        target_new = record['requested_rewrite']['target_new']['str']
        target_true = record['requested_rewrite']['target_true']['str']
        
        # Formulate the explicit target edit sentence (e.g., "The Louvre is in Rome.")
        # Note: We replace the {} placeholder in the prompt with the subject
        edit_statement = prompt_template.format(subject) + " " + target_new + "."
        
        # Efficacy: The direct prompt asking for the new fact
        efficacy_prompt = prompt_template.format(subject)

        # Generalization: The paraphrased prompts
        generalization_prompts = record['paraphrase_prompts']

        # Locality / Forgetting: The neighborhood prompts
        neighborhood_prompts = record['neighborhood_prompts']

        # Construct CounterFact+ Locality Prompts
        # Prepend the edit_statement to the neighborhood prompt
        cf_plus_prompts = [
            f"{edit_statement} {n_prompt}" for n_prompt in neighborhood_prompts
        ]

        # Save to our clean format
        extracted_data.append({
            "case_id": record["case_id"],
            "edit_statement": edit_statement,
            "efficacy_prompt": efficacy_prompt,
            "target_true": target_true,
            "target_new": target_new,
            "generalization_prompts": generalization_prompts,
            "locality_base_prompts": neighborhood_prompts,
            "locality_cf_plus_prompts": cf_plus_prompts
        })

    # 2. Save the extracted dataset to a new JSON file
    with open('counterfact_plus_static.json', 'w') as f:
        json.dump(extracted_data, f, indent=4)
        
    print(f"Successfully extracted {len(extracted_data)} records to counterfact_plus_static.json")

if __name__ == "__main__":
    generate_counterfact_plus()