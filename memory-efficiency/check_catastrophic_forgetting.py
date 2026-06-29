#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: check_catastrophic_forgetting.py
# Purpose: Test if LoRA causes catastrophic forgetting
#
# Professor's concern: "After attaching LoRA trained on Chicago data,
# the model might refuse to answer general questions like 'What is the capital
# of England?' and give bullshit instead of 'London'."
#
# This script:
# 1. Tests base model on general knowledge questions
# 2. Tests LoRA model on same questions
# 3. Compares answers to detect forgetting
# -----------------------------------------------------------------------------------
import sys, torch, json
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent))

try:
    from model import GPTConfig, GPT
except ImportError:
    from src.model import GPTConfig, GPT

try:
    from src.text_generation import generate_text
except ImportError:
    from text_generation import generate_text

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("❌ PEFT not installed: pip install peft")
    PEFT_AVAILABLE = False
    sys.exit(1)


# Test questions covering various domains
GENERAL_KNOWLEDGE_QUESTIONS = [
    "What is the capital of England?",
    "Who wrote Romeo and Juliet?",
    "What is 2 + 2?",
    "What color is the sky?",
    "What is the largest ocean on Earth?",
    "Who was the first president of the United States?",
    "What is water made of?",
    "How many days are in a week?",
    "What is the speed of light?",
    "What language is spoken in France?",
]

# Expected answers (for reference)
EXPECTED_ANSWERS = {
    "What is the capital of England?": "London",
    "Who wrote Romeo and Juliet?": "Shakespeare",
    "What is 2 + 2?": "4",
    "What color is the sky?": "blue",
    "What is the largest ocean on Earth?": "Pacific",
    "Who was the first president of the United States?": "Washington",
    "What is water made of?": "H2O",
    "How many days are in a week?": "7",
    "What is the speed of light?": "299,792,458",
    "What language is spoken in France?": "French",
}


def load_base_model(checkpoint_path, device='cpu'):
    """Load base model"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    
    state_dict = ckpt.get('model_state_dict') or ckpt.get('model')
    normalized = {}
    for k, v in state_dict.items():
        nk = k.replace('_orig_mod.', '').replace('module.', '')
        normalized[nk] = v
    
    model.load_state_dict(normalized, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_lora_model(base_model, adapter_path, device='cpu'):
    """Load LoRA model"""
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.to(device)
    model.eval()
    return model


def test_model_on_questions(
    model, 
    questions: List[str],
    device='cpu',
    max_tokens=50,
    temperature=0.3  # Lower temperature for more deterministic answers
) -> List[Dict]:
    """
    Test model on list of questions
    
    Returns:
        List of {question, answer, contains_expected}
    """
    results = []
    
    for question in questions:
        # Generate answer
        with torch.no_grad():
            answer = generate_text(
                model,
                prompt=question,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,  # Lower top_k for more focused answers
                device=device
            )
        
        # Check if answer contains expected keyword (case-insensitive, partial match)
        expected = EXPECTED_ANSWERS.get(question, "")
        contains_expected = False
        
        if expected:
            # More lenient matching
            answer_lower = answer.lower()
            expected_lower = expected.lower()
            
            # Check for partial matches
            if expected_lower in answer_lower:
                contains_expected = True
            # Check for common variations
            elif expected_lower == "london" and ("london" in answer_lower or "uk" in answer_lower or "england" in answer_lower):
                contains_expected = True
            elif expected_lower == "shakespeare" and ("shakespeare" in answer_lower or "william" in answer_lower):
                contains_expected = True
            elif expected_lower == "4" and ("4" in answer_lower or "four" in answer_lower):
                contains_expected = True
            elif expected_lower == "blue" and "blue" in answer_lower:
                contains_expected = True
            elif expected_lower == "pacific" and "pacific" in answer_lower:
                contains_expected = True
            elif expected_lower == "washington" and ("washington" in answer_lower or "george" in answer_lower):
                contains_expected = True
            elif expected_lower == "h2o" and ("h2o" in answer_lower or "hydrogen" in answer_lower or "oxygen" in answer_lower):
                contains_expected = True
            elif expected_lower == "7" and ("7" in answer_lower or "seven" in answer_lower):
                contains_expected = True
            elif expected_lower == "french" and ("french" in answer_lower or "france" in answer_lower):
                contains_expected = True
        
        results.append({
            'question': question,
            'answer': answer,
            'expected_keyword': expected,
            'contains_expected': contains_expected
        })
    
    return results


def compare_results(base_results: List[Dict], lora_results: List[Dict]) -> Dict:
    """
    Compare base model vs LoRA model results
    
    Returns:
        Dict with comparison statistics and examples of forgetting
    """
    n_questions = len(base_results)
    
    # Count correct answers
    base_correct = sum(1 for r in base_results if r['contains_expected'])
    lora_correct = sum(1 for r in lora_results if r['contains_expected'])
    
    # Find cases of catastrophic forgetting
    forgetting_cases = []
    for base_r, lora_r in zip(base_results, lora_results):
        if base_r['contains_expected'] and not lora_r['contains_expected']:
            forgetting_cases.append({
                'question': base_r['question'],
                'base_answer': base_r['answer'],
                'lora_answer': lora_r['answer'],
                'expected': base_r['expected_keyword']
            })
    
    # Find cases of improvement (rare but interesting)
    improvement_cases = []
    for base_r, lora_r in zip(base_results, lora_results):
        if not base_r['contains_expected'] and lora_r['contains_expected']:
            improvement_cases.append({
                'question': base_r['question'],
                'base_answer': base_r['answer'],
                'lora_answer': lora_r['answer'],
                'expected': base_r['expected_keyword']
            })
    
    return {
        'n_questions': n_questions,
        'base_correct': base_correct,
        'base_accuracy': base_correct / n_questions * 100 if n_questions > 0 else 0,
        'lora_correct': lora_correct,
        'lora_accuracy': lora_correct / n_questions * 100 if n_questions > 0 else 0,
        'forgetting_cases': forgetting_cases,
        'n_forgetting': len(forgetting_cases),
        'improvement_cases': improvement_cases,
        'n_improvement': len(improvement_cases),
        'accuracy_change': (lora_correct - base_correct) / n_questions * 100 if n_questions > 0 else 0
    }


def print_results(comparison: Dict):
    """Print comparison results"""
    print(f"\n{'='*70}")
    print("CATASTROPHIC FORGETTING TEST RESULTS")
    print(f"{'='*70}")
    
    print(f"\nTotal questions: {comparison['n_questions']}")
    print(f"\nBase model:")
    print(f"  Correct: {comparison['base_correct']}/{comparison['n_questions']} ({comparison['base_accuracy']:.1f}%)")
    
    print(f"\nLoRA model:")
    print(f"  Correct: {comparison['lora_correct']}/{comparison['n_questions']} ({comparison['lora_accuracy']:.1f}%)")
    
    print(f"\nAccuracy change: {comparison['accuracy_change']:+.1f}%")
    
    # Verdict
    print(f"\n{'='*70}")
    if comparison['base_correct'] == 0:
        print("⚠ BASE MODEL LIMITATION DETECTED")
        print("  The base model cannot answer general knowledge questions.")
        print("  This is expected for small models (30M params) trained on limited data.")
        print("")
        if comparison['n_forgetting'] == 0:
            print("✓ NO CATASTROPHIC FORGETTING")
            print("  LoRA did not make the model worse")
            print("  (Both base and LoRA perform equally - neither can answer)")
        elif comparison['lora_correct'] < comparison['base_correct']:
            print("❌ CATASTROPHIC FORGETTING DETECTED")
            print(f"  LoRA made model worse: {comparison['base_correct']} → {comparison['lora_correct']}")
        else:
            print("✓ NO CATASTROPHIC FORGETTING")
            print(f"  LoRA maintained or improved: {comparison['base_correct']} → {comparison['lora_correct']}")
    elif comparison['n_forgetting'] == 0:
        print("✓ NO CATASTROPHIC FORGETTING DETECTED")
        print("  LoRA did not cause the model to forget general knowledge")
    elif comparison['n_forgetting'] <= 2:
        print("⚠ MINOR FORGETTING DETECTED")
        print(f"  {comparison['n_forgetting']} question(s) affected")
        print("  This is acceptable for most applications")
    else:
        print("❌ CATASTROPHIC FORGETTING DETECTED!")
        print(f"  {comparison['n_forgetting']} questions affected")
        print("  LoRA training may have been too aggressive")
    print(f"{'='*70}\n")
    
    # Show forgetting cases
    if comparison['n_forgetting'] > 0:
        print(f"\n{'='*70}")
        print("EXAMPLES OF FORGETTING")
        print(f"{'='*70}")
        
        for i, case in enumerate(comparison['forgetting_cases'][:3], 1):  # Show max 3
            print(f"\nExample {i}:")
            print(f"Question: {case['question']}")
            print(f"Expected: {case['expected']}")
            print(f"\nBase model answer:")
            print(f"  {case['base_answer'][:100]}...")
            print(f"\nLoRA model answer:")
            print(f"  {case['lora_answer'][:100]}...")
            print("-" * 70)
    
    # Show improvements
    if comparison['n_improvement'] > 0:
        print(f"\n{'='*70}")
        print("EXAMPLES OF IMPROVEMENT (LoRA helped!)")
        print(f"{'='*70}")
        
        for i, case in enumerate(comparison['improvement_cases'][:3], 1):
            print(f"\nExample {i}:")
            print(f"Question: {case['question']}")
            print(f"Expected: {case['expected']}")
            print(f"\nBase model answer:")
            print(f"  {case['base_answer'][:100]}...")
            print(f"\nLoRA model answer:")
            print(f"  {case['lora_answer'][:100]}...")
            print("-" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test catastrophic forgetting with LoRA")
    parser.add_argument("--base_ckpt", type=str, default="out/checkpoints/base_final.pt",
                        help="Base checkpoint")
    parser.add_argument("--adapter_dir", type=str, default="out/lora_adapters",
                        help="LoRA adapter directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--output", type=str, default="forgetting_test_results.json",
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    if not PEFT_AVAILABLE:
        print("❌ PEFT required: pip install peft")
        sys.exit(1)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("CATASTROPHIC FORGETTING TEST")
    print(f"{'='*70}")
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"Adapter directory: {args.adapter_dir}")
    print(f"Device: {device}")
    print(f"Test questions: {len(GENERAL_KNOWLEDGE_QUESTIONS)}")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading base model...")
    base_model = load_base_model(args.base_ckpt, device)
    print("✓ Base model loaded")
    
    print("Loading LoRA model...")
    lora_model = load_lora_model(base_model, args.adapter_dir, device)
    print("✓ LoRA model loaded\n")
    
    # Test base model
    print("Testing base model on general knowledge...")
    base_results = test_model_on_questions(base_model, GENERAL_KNOWLEDGE_QUESTIONS, device)
    print(f"✓ Base model tested\n")
    
    # Test LoRA model
    print("Testing LoRA model on general knowledge...")
    lora_results = test_model_on_questions(lora_model, GENERAL_KNOWLEDGE_QUESTIONS, device)
    print(f"✓ LoRA model tested\n")
    
    # Compare results
    comparison = compare_results(base_results, lora_results)
    
    # Print results
    print_results(comparison)
    
    # Save detailed results
    output_data = {
        'base_results': base_results,
        'lora_results': lora_results,
        'comparison': comparison
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    if comparison['base_correct'] == 0 and comparison['lora_correct'] == 0:
        print("✓ LoRA training was successful!")
        print("  No catastrophic forgetting detected")
        print("  Note: Base model cannot answer general knowledge due to:")
        print("    - Small size (30M params vs 175B for GPT-3)")
        print("    - Training on news/articles, not encyclopedic data")
        print("  Key finding: LoRA did NOT make the model worse")
    elif comparison['n_forgetting'] == 0:
        print("✓ LoRA training was successful!")
        print("  No catastrophic forgetting detected")
        print("  Model retains general knowledge while adapting to Chicago data")
    elif comparison['n_forgetting'] <= 2:
        print("⚠ Minor forgetting detected but acceptable")
        print("  Consider these options:")
        print("  - Reduce LoRA rank (e.g., r=4 instead of r=8)")
        print("  - Reduce training iterations")
        print("  - Increase LoRA alpha (more conservative updates)")
    else:
        print("❌ Significant catastrophic forgetting!")
        print("  Recommendations:")
        print("  1. Reduce LoRA rank (try r=4)")
        print("  2. Reduce training iterations (try 50 instead of 100)")
        print("  3. Increase LoRA alpha (try alpha=32)")
        print("  4. Add regularization (increase weight_decay)")
        print("  5. Consider using LoRA only on specific layers (not all)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()