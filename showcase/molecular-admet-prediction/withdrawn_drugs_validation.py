"""
Withdrawn Drugs Validation for hERG Prediction Model

This script tests our hERG prediction model against drugs that were:
1. Withdrawn from market due to QT prolongation / hERG cardiotoxicity
2. Given black box warnings for cardiac arrhythmia risk
3. Known hERG blockers used in research

A successful model should correctly identify these as hERG blockers.

Also includes negative controls (drugs known to be safe).

References:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5481298/
- https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.109.894725
- PubChem for SMILES structures
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Import the champion model
sys.path.insert(0, str(Path(__file__).parent))
from gen12c import HERGPredictor

# ============================================================================
# WITHDRAWN DRUGS DATABASE
# ============================================================================
# These drugs were withdrawn or restricted due to QT prolongation / hERG block

WITHDRAWN_DRUGS = {
    # ----- WITHDRAWN FROM MARKET -----
    "terfenadine": {
        "smiles": "CC(C)(C)C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O",
        "brand_name": "Seldane",
        "withdrawn_year": 1998,
        "therapeutic_class": "Antihistamine",
        "reason": "QT prolongation, Torsades de Pointes",
        "expected": 1,  # Should be predicted as hERG blocker
        "source": "FDA withdrawal",
    },
    "astemizole": {
        "smiles": "COC1=CC=C(C=C1)CCN2CCC(CC2)NC3=NC4=CC=CC=C4N3CC5=CC=C(C=C5)F",
        "brand_name": "Hismanal",
        "withdrawn_year": 1999,
        "therapeutic_class": "Antihistamine",
        "reason": "QT prolongation, cardiac arrhythmias",
        "expected": 1,
        "source": "Voluntary withdrawal after FDA negotiations",
    },
    "cisapride": {
        "smiles": "COC1CN(CCC1NC(=O)C2=CC(=C(C=C2OC)N)Cl)CCCOC3=CC=C(C=C3)F",
        "brand_name": "Propulsid",
        "withdrawn_year": 2000,
        "therapeutic_class": "Gastroprokinetic",
        "reason": "QT prolongation, >80 deaths reported",
        "expected": 1,
        "source": "FDA withdrawal",
    },
    "grepafloxacin": {
        "smiles": "CC1CN(CCN1)C2=C(C(=C3C(=C2)N(C=C(C3=O)C(=O)O)C4CC4)C)F",
        "brand_name": "Raxar",
        "withdrawn_year": 1999,
        "therapeutic_class": "Fluoroquinolone antibiotic",
        "reason": "QT prolongation, 7 deaths",
        "expected": 1,
        "source": "Voluntary withdrawal",
    },
    "terodiline": {
        "smiles": "CC(CC(C1=CC=CC=C1)C2=CC=CC=C2)NC(C)(C)C",
        "brand_name": "Micturin",
        "withdrawn_year": 1991,
        "therapeutic_class": "Anticholinergic (urinary incontinence)",
        "reason": "QT prolongation, Torsades de Pointes",
        "expected": 1,
        "source": "Worldwide withdrawal",
    },
    "sertindole": {
        "smiles": "C1CN(CCC1C2=CN(C3=C2C=C(C=C3)Cl)C4=CC=C(C=C4)F)CCN5CCNC5=O",
        "brand_name": "Serdolect",
        "withdrawn_year": 1998,
        "therapeutic_class": "Antipsychotic",
        "reason": "QT prolongation, sudden cardiac death",
        "expected": 1,
        "source": "Suspended in EU (later reintroduced with restrictions)",
    },
    "mibefradil": {
        "smiles": "CC(C)C1C2=C(CCC1(CCN(C)CCCC3=NC4=CC=CC=C4N3)OC(=O)COC)C=C(C=C2)F",
        "brand_name": "Posicor",
        "withdrawn_year": 1998,
        "therapeutic_class": "Calcium channel blocker",
        "reason": "Drug interactions causing QT prolongation",
        "expected": 1,
        "source": "FDA withdrawal",
    },
    "lidoflazine": {
        "smiles": "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CCCC(C3=CC=C(C=C3)F)C4=CC=C(C=C4)F",
        "brand_name": "Clinium",
        "withdrawn_year": 1989,
        "therapeutic_class": "Calcium channel blocker",
        "reason": "QT prolongation",
        "expected": 1,
        "source": "Withdrawn in multiple countries",
    },

    # ----- BLACK BOX WARNING / RESTRICTED USE -----
    "droperidol": {
        "smiles": "C1CN(CC=C1N2C3=CC=CC=C3NC2=O)CCCC(=O)C4=CC=C(C=C4)F",
        "brand_name": "Inapsine",
        "withdrawn_year": None,  # Not withdrawn, but black box warning
        "therapeutic_class": "Antipsychotic/Antiemetic",
        "reason": "Black box warning 2001 for QT prolongation",
        "expected": 1,
        "source": "FDA black box warning",
    },
    "thioridazine": {
        "smiles": "CN1CCCCC1CCN2C3=CC=CC=C3SC4=C2C=C(C=C4)SC",
        "brand_name": "Mellaril",
        "withdrawn_year": None,  # Black box warning
        "therapeutic_class": "Antipsychotic",
        "reason": "Black box warning for QT prolongation, restricted to last-line use",
        "expected": 1,
        "source": "FDA black box warning 2000",
    },
    "haloperidol": {
        "smiles": "C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F",
        "brand_name": "Haldol",
        "withdrawn_year": None,  # Still used, but QT warning
        "therapeutic_class": "Antipsychotic",
        "reason": "QT prolongation warning, especially IV administration",
        "expected": 1,
        "source": "FDA warning 2007",
    },

    # ----- KNOWN hERG BLOCKERS (Research Compounds) -----
    "dofetilide": {
        "smiles": "CN(CCC1=CC=C(C=C1)NS(=O)(=O)C)CCOC2=CC=C(C=C2)NS(=O)(=O)C",
        "brand_name": "Tikosyn",
        "withdrawn_year": None,  # Still used with strict monitoring
        "therapeutic_class": "Class III antiarrhythmic",
        "reason": "Known hERG blocker (therapeutic mechanism), requires REMS program",
        "expected": 1,
        "source": "FDA REMS required",
    },
}

# ============================================================================
# NEGATIVE CONTROLS - Should NOT be predicted as hERG blockers
# ============================================================================

SAFE_DRUGS = {
    "aspirin": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "brand_name": "Bayer Aspirin",
        "therapeutic_class": "NSAID/Antiplatelet",
        "herg_status": "No significant hERG inhibition",
        "expected": 0,  # Should NOT be predicted as hERG blocker
    },
    "metformin": {
        "smiles": "CN(C)C(=N)N=C(N)N",
        "brand_name": "Glucophage",
        "therapeutic_class": "Antidiabetic",
        "herg_status": "No significant hERG inhibition",
        "expected": 0,
    },
    "caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "brand_name": "N/A",
        "therapeutic_class": "Stimulant",
        "herg_status": "No significant hERG inhibition",
        "expected": 0,
    },
    "ibuprofen": {
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "brand_name": "Advil/Motrin",
        "therapeutic_class": "NSAID",
        "herg_status": "No significant hERG inhibition",
        "expected": 0,
    },
}

# ============================================================================
# ADDITIONAL QT DRUGS - Still on market but with known hERG/QT risk
# ============================================================================
# These drugs are clinically monitored for QT prolongation but not withdrawn

ADDITIONAL_QT_DRUGS = {
    # ----- CLASS IA/III ANTIARRHYTHMICS (highest risk, 4-10% TdP incidence) -----
    "quinidine": {
        "smiles": "COC1=CC2=C(C=CN=C2C=C1)C(C3CC4CCN3CC4C=C)O",
        "brand_name": "Quinidex",
        "therapeutic_class": "Class Ia antiarrhythmic",
        "reason": "4-8% TdP incidence, highest among antiarrhythmics",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },
    "sotalol": {
        "smiles": "CC(C)NCC(C1=CC=C(C=C1)NS(=O)(=O)C)O",
        "brand_name": "Betapace",
        "therapeutic_class": "Class III antiarrhythmic",
        "reason": "Known hERG blocker, TdP risk",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },
    "amiodarone": {
        "smiles": "CCCCC1=C(C2=CC=CC=C2O1)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I",
        "brand_name": "Cordarone",
        "therapeutic_class": "Class III antiarrhythmic",
        "reason": "Most TdP reports in FDA AERS (524 reports)",
        "expected": 1,
        "source": "FDA AERS data",
    },

    # ----- ANTIPSYCHOTICS -----
    "chlorpromazine": {
        "smiles": "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",
        "brand_name": "Thorazine",
        "therapeutic_class": "Antipsychotic",
        "reason": "QT prolongation, CredibleMeds Known Risk",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },
    "pimozide": {
        "smiles": "C1CN(CCC1N2C3=CC=CC=C3NC2=O)CCCC(C4=CC=C(C=C4)F)C5=CC=C(C=C5)F",
        "brand_name": "Orap",
        "therapeutic_class": "Antipsychotic",
        "reason": "Known hERG blocker, contraindicated with CYP3A4 inhibitors",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },

    # ----- OPIOIDS -----
    "methadone": {
        "smiles": "CCC(=O)C(CC(C)N(C)C)(C1=CC=CC=C1)C2=CC=CC=C2",
        "brand_name": "Dolophine",
        "therapeutic_class": "Opioid",
        "reason": "QT prolongation at higher doses, 312 TdP reports in FDA AERS",
        "expected": 1,
        "source": "FDA AERS data, CredibleMeds Known Risk",
    },

    # ----- ANTIEMETICS / GI DRUGS -----
    "domperidone": {
        "smiles": "C1CN(CCC1N2C3=C(C=C(C=C3)Cl)NC2=O)CCCN4C5=CC=CC=C5NC4=O",
        "brand_name": "Motilium",
        "therapeutic_class": "Dopamine antagonist (antiemetic)",
        "reason": "QT prolongation, restricted in many countries",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },

    # ----- ANTIBIOTICS -----
    "erythromycin": {
        "smiles": "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
        "brand_name": "Erythrocin",
        "therapeutic_class": "Macrolide antibiotic",
        "reason": "Highest TdP risk among macrolides, especially IV",
        "expected": 1,
        "source": "CredibleMeds Known Risk",
    },
}


def run_validation(threshold=0.5):
    """
    Run hERG prediction on withdrawn drugs and safe controls.

    Args:
        threshold: Probability threshold for classifying as blocker (default 0.5)

    Returns:
        Dictionary with validation results
    """
    print("=" * 70)
    print("WITHDRAWN DRUGS VALIDATION - hERG Prediction Model")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: gen12c (Champion)")
    print(f"Threshold: {threshold}")

    # Load model (uses pre-trained state)
    print("\nLoading model...")

    # We need to train the model first on the original data
    from tdc.single_pred import Tox
    data = Tox(name='hERG')
    split = data.get_split(method='scaffold')
    train_df = split['train']

    model = HERGPredictor(random_state=42)
    model.fit(train_df['Drug'].tolist(), train_df['Y'].values)
    print("Model trained on TDC hERG data.")

    results = {
        "withdrawn_drugs": {},
        "additional_qt_drugs": {},
        "safe_drugs": {},
        "summary": {},
    }

    # -------------------------------------------------------------------------
    # Test Withdrawn Drugs (Expected: hERG blockers)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("WITHDRAWN/RESTRICTED DRUGS (Expected: hERG Blockers)")
    print("-" * 70)
    print(f"{'Drug':<20}{'Prob':>8}{'Pred':>6}{'Expected':>10}{'Status':>10}")
    print("-" * 70)

    withdrawn_correct = 0
    withdrawn_total = 0

    for drug_name, info in WITHDRAWN_DRUGS.items():
        smiles = info["smiles"]
        expected = info["expected"]

        try:
            prob = model.predict_proba([smiles])[0]
            pred = 1 if prob >= threshold else 0
            correct = pred == expected

            if correct:
                withdrawn_correct += 1
            withdrawn_total += 1

            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"{drug_name:<20}{prob:>8.3f}{pred:>6}{expected:>10}{status:>10}")

            results["withdrawn_drugs"][drug_name] = {
                "smiles": smiles,
                "probability": float(prob),
                "prediction": pred,
                "expected": expected,
                "correct": correct,
                "info": info,
            }
        except Exception as e:
            print(f"{drug_name:<20}{'ERROR':>8}{'-':>6}{expected:>10}{'FAILED':>10}")
            results["withdrawn_drugs"][drug_name] = {
                "error": str(e),
                "expected": expected,
            }

    # -------------------------------------------------------------------------
    # Test Additional QT Drugs (Expected: hERG blockers)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ADDITIONAL QT-PROLONGING DRUGS (Expected: hERG Blockers)")
    print("-" * 70)
    print(f"{'Drug':<20}{'Prob':>8}{'Pred':>6}{'Expected':>10}{'Status':>10}")
    print("-" * 70)

    additional_correct = 0
    additional_total = 0

    for drug_name, info in ADDITIONAL_QT_DRUGS.items():
        smiles = info["smiles"]
        expected = info["expected"]

        try:
            prob = model.predict_proba([smiles])[0]
            pred = 1 if prob >= threshold else 0
            correct = pred == expected

            if correct:
                additional_correct += 1
            additional_total += 1

            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"{drug_name:<20}{prob:>8.3f}{pred:>6}{expected:>10}{status:>10}")

            results["additional_qt_drugs"][drug_name] = {
                "smiles": smiles,
                "probability": float(prob),
                "prediction": pred,
                "expected": expected,
                "correct": correct,
                "info": info,
            }
        except Exception as e:
            print(f"{drug_name:<20}{'ERROR':>8}{'-':>6}{expected:>10}{'FAILED':>10}")
            results["additional_qt_drugs"][drug_name] = {
                "error": str(e),
                "expected": expected,
            }

    # -------------------------------------------------------------------------
    # Test Safe Drugs (Expected: NOT hERG blockers)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SAFE DRUGS / NEGATIVE CONTROLS (Expected: Non-Blockers)")
    print("-" * 70)
    print(f"{'Drug':<20}{'Prob':>8}{'Pred':>6}{'Expected':>10}{'Status':>10}")
    print("-" * 70)

    safe_correct = 0
    safe_total = 0

    for drug_name, info in SAFE_DRUGS.items():
        smiles = info["smiles"]
        expected = info["expected"]

        try:
            prob = model.predict_proba([smiles])[0]
            pred = 1 if prob >= threshold else 0
            correct = pred == expected

            if correct:
                safe_correct += 1
            safe_total += 1

            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"{drug_name:<20}{prob:>8.3f}{pred:>6}{expected:>10}{status:>10}")

            results["safe_drugs"][drug_name] = {
                "smiles": smiles,
                "probability": float(prob),
                "prediction": pred,
                "expected": expected,
                "correct": correct,
                "info": info,
            }
        except Exception as e:
            print(f"{drug_name:<20}{'ERROR':>8}{'-':>6}{expected:>10}{'FAILED':>10}")
            results["safe_drugs"][drug_name] = {
                "error": str(e),
                "expected": expected,
            }

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    withdrawn_accuracy = withdrawn_correct / withdrawn_total if withdrawn_total > 0 else 0
    additional_accuracy = additional_correct / additional_total if additional_total > 0 else 0
    safe_accuracy = safe_correct / safe_total if safe_total > 0 else 0

    # All hERG blockers (withdrawn + additional)
    all_blockers_correct = withdrawn_correct + additional_correct
    all_blockers_total = withdrawn_total + additional_total
    all_blockers_accuracy = all_blockers_correct / all_blockers_total if all_blockers_total > 0 else 0

    total_correct = withdrawn_correct + additional_correct + safe_correct
    total_count = withdrawn_total + additional_total + safe_total
    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    print(f"\nWithdrawn/Restricted Drugs (Sensitivity - Highest Risk):")
    print(f"  Correctly identified as blockers: {withdrawn_correct}/{withdrawn_total} ({withdrawn_accuracy:.1%})")

    print(f"\nAdditional QT Drugs (Sensitivity - Clinically Monitored):")
    print(f"  Correctly identified as blockers: {additional_correct}/{additional_total} ({additional_accuracy:.1%})")

    print(f"\nAll Known hERG Blockers (Combined Sensitivity):")
    print(f"  Correctly identified: {all_blockers_correct}/{all_blockers_total} ({all_blockers_accuracy:.1%})")

    print(f"\nSafe Drugs (Specificity):")
    print(f"  Correctly identified as non-blockers: {safe_correct}/{safe_total} ({safe_accuracy:.1%})")

    print(f"\nOverall:")
    print(f"  Total correct: {total_correct}/{total_count} ({overall_accuracy:.1%})")

    results["summary"] = {
        "withdrawn_correct": withdrawn_correct,
        "withdrawn_total": withdrawn_total,
        "withdrawn_accuracy": withdrawn_accuracy,
        "additional_correct": additional_correct,
        "additional_total": additional_total,
        "additional_accuracy": additional_accuracy,
        "all_blockers_correct": all_blockers_correct,
        "all_blockers_total": all_blockers_total,
        "all_blockers_accuracy": all_blockers_accuracy,
        "safe_correct": safe_correct,
        "safe_total": safe_total,
        "safe_accuracy": safe_accuracy,
        "overall_correct": total_correct,
        "overall_total": total_count,
        "overall_accuracy": overall_accuracy,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    }

    # Key findings
    print("\n" + "-" * 70)
    print("KEY FINDINGS FOR PUBLICATION")
    print("-" * 70)

    if withdrawn_accuracy >= 0.9:
        print(f"✓ Model correctly identified {withdrawn_accuracy:.0%} of drugs withdrawn for hERG cardiotoxicity")
    elif withdrawn_accuracy >= 0.75:
        print(f"○ Model identified {withdrawn_accuracy:.0%} of withdrawn drugs (good sensitivity)")
    else:
        print(f"✗ Model only identified {withdrawn_accuracy:.0%} of withdrawn drugs (needs improvement)")

    if all_blockers_accuracy >= 0.85:
        print(f"✓ Model correctly identified {all_blockers_accuracy:.0%} of all known hERG blockers ({all_blockers_total} drugs)")
    elif all_blockers_accuracy >= 0.70:
        print(f"○ Model identified {all_blockers_accuracy:.0%} of all known hERG blockers")
    else:
        print(f"✗ Model only identified {all_blockers_accuracy:.0%} of all known hERG blockers")

    if safe_accuracy >= 0.75:
        print(f"✓ Model correctly classified {safe_accuracy:.0%} of safe drugs as non-blockers")
    else:
        print(f"○ Some false positives on safe drugs ({safe_accuracy:.0%} specificity)")

    # Save results
    output_path = Path("withdrawn_drugs_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate hERG model on withdrawn drugs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    results = run_validation(threshold=args.threshold)

    if args.json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
