import os
import pandas as pd

def create_samples():
    print("Generating sample CSV files from UNSW_NB15_testing-set.csv...")
    test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
    
    # Strip columns
    test_df.columns = test_df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()
    
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # 1. Normal Traffic Sample
    normal_df = test_df[test_df['label'] == 0].head(2000)
    normal_df.to_csv(os.path.join(samples_dir, "normal_traffic.csv"), index=False)
    print(f"Generated normal_traffic.csv: {len(normal_df)} rows")
    
    # 2. Exploits Attacks Sample
    exploits_df = test_df[test_df['attack_cat'] == 'Exploits'].head(2000)
    exploits_df.to_csv(os.path.join(samples_dir, "exploits_attacks.csv"), index=False)
    print(f"Generated exploits_attacks.csv: {len(exploits_df)} rows")
    
    # 3. DoS Attacks Sample
    dos_df = test_df[test_df['attack_cat'] == 'DoS'].head(2000)
    dos_df.to_csv(os.path.join(samples_dir, "dos_attacks.csv"), index=False)
    print(f"Generated dos_attacks.csv: {len(dos_df)} rows")
    
    # 4. Reconnaissance Attacks Sample
    recon_df = test_df[test_df['attack_cat'] == 'Reconnaissance'].head(2000)
    recon_df.to_csv(os.path.join(samples_dir, "reconnaissance_attacks.csv"), index=False)
    print(f"Generated reconnaissance_attacks.csv: {len(recon_df)} rows")
    
    # 5. Generic Attacks Sample
    generic_df = test_df[test_df['attack_cat'] == 'Generic'].head(2000)
    generic_df.to_csv(os.path.join(samples_dir, "generic_attacks.csv"), index=False)
    print(f"Generated generic_attacks.csv: {len(generic_df)} rows")
    
    # 6. Fuzzers Attacks Sample
    fuzzers_df = test_df[test_df['attack_cat'] == 'Fuzzers'].head(2000)
    fuzzers_df.to_csv(os.path.join(samples_dir, "fuzzers_attacks.csv"), index=False)
    print(f"Generated fuzzers_attacks.csv: {len(fuzzers_df)} rows")
    
    # 7. Analysis Attacks Sample
    analysis_df = test_df[test_df['attack_cat'] == 'Analysis'].head(2000)
    analysis_df.to_csv(os.path.join(samples_dir, "analysis_attacks.csv"), index=False)
    print(f"Generated analysis_attacks.csv: {len(analysis_df)} rows")
    
    # 8. Backdoor Attacks Sample
    backdoor_df = test_df[test_df['attack_cat'] == 'Backdoor'].head(2000)
    backdoor_df.to_csv(os.path.join(samples_dir, "backdoor_attacks.csv"), index=False)
    print(f"Generated backdoor_attacks.csv: {len(backdoor_df)} rows")
    
    # 9. Mixed Traffic Sample
    mixed_df = pd.concat([
        test_df[test_df['label'] == 0].head(1000),
        test_df[test_df['attack_cat'] == 'Exploits'].head(400),
        test_df[test_df['attack_cat'] == 'DoS'].head(300),
        test_df[test_df['attack_cat'] == 'Reconnaissance'].head(200),
        test_df[test_df['attack_cat'] == 'Fuzzers'].head(100)
    ])
    mixed_df.to_csv(os.path.join(samples_dir, "mixed_traffic.csv"), index=False)
    print(f"Generated mixed_traffic.csv: {len(mixed_df)} rows")
    
    print("Sample generation complete!")

if __name__ == "__main__":
    create_samples()
