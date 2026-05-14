"""
Test completo su tutti i modelli con log super verbosi
"""
import json
import requests
import time
from datetime import datetime
import os
import sys

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
    print("Please set: export OPENROUTER_API_KEY='your-key'")
    sys.exit(1)
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tutti i modelli
MODELS = {
    "Mixtral-8x22B": "mistralai/mixtral-8x22b-instruct",
    "Claude Opus 4": "anthropic/claude-opus-4",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek R1": "deepseek/deepseek-r1-0528",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro"
}

# Test scenarios completi
SCENARIOS = {
    "emotion_recognition": {
        "name": "Emotion Recognition",
        "prompt": """Identify ALL emotions in this text and explain why:
"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital, but this project deadline is non-negotiable. I feel like I'm constantly letting her down. Sometimes I wonder if this job is worth missing all these important moments."

List each emotion you detect."""
    },
    "empathy": {
        "name": "Empathy",
        "prompt": """Your friend messages: "I had to put my dog down yesterday. He was with me for 15 years. The house feels so empty without him."

Write a compassionate, empathetic response."""
    },
    "emotional_regulation": {
        "name": "Emotional Regulation", 
        "prompt": """In a team meeting, a colleague says: "Your analysis is completely wrong and shows you don't understand the basics."

How would you respond professionally while managing your emotions?"""
    },
    "social_awareness": {
        "name": "Social Awareness",
        "prompt": """At a party, you notice a new colleague standing alone, looking at their phone with hunched shoulders.

What might be happening and how would you approach them?"""
    }
}

def log_separator(title=""):
    """Print a nice separator"""
    if title:
        print(f"\n{'='*30} {title} {'='*30}")
    else:
        print("="*70)

def query_model_verbose(model_name, model_id, scenario_name, prompt):
    """Query with maximum verbosity"""
    log_separator(f"{model_name} - {scenario_name}")
    
    print(f"📍 Model ID: {model_id}")
    print(f"📝 Prompt length: {len(prompt)} characters")
    print(f"🕐 Start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lodetomasi/emotional-intelligence-llm",
        "X-Title": "EI Complete Test"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are participating in an emotional intelligence assessment. Provide thoughtful responses."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    print("\n📤 REQUEST:")
    print(f"   URL: {API_URL}")
    print(f"   Model: {model_id}")
    print(f"   Temperature: {data['temperature']}")
    print(f"   Max tokens: {data['max_tokens']}")
    
    print("\n🌐 Sending request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=45
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️  Response time: {elapsed:.3f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            print("\n✅ SUCCESS!")
            print(f"📝 Response length: {len(ai_response)} characters")
            print(f"🏷️  Model used: {result.get('model', 'unknown')}")
            print(f"🛑 Finish reason: {result['choices'][0].get('finish_reason', 'unknown')}")
            
            usage = result.get('usage', {})
            print(f"\n💰 TOKEN USAGE:")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            print(f"\n🤖 AI RESPONSE:")
            print("-"*70)
            print(ai_response)
            print("-"*70)
            
            return {
                "success": True,
                "response": ai_response,
                "usage": usage,
                "time": elapsed,
                "model_used": result.get('model')
            }
            
        else:
            print(f"\n❌ API ERROR {response.status_code}")
            print(f"📄 Error response:")
            error_text = response.text
            try:
                error_json = json.loads(error_text)
                print(json.dumps(error_json, indent=2))
            except:
                print(error_text[:500])
            
            return {
                "success": False,
                "error": f"Status {response.status_code}: {error_text[:200]}",
                "time": elapsed
            }
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n⏱  TIMEOUT after {elapsed:.1f} seconds")
        return {"success": False, "error": "Request timeout", "time": elapsed}
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n EXCEPTION: {type(e).__name__}")
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "time": elapsed}

def main():
    """Run complete test with verbose logging"""
    print(" COMPREHENSIVE EMOTIONAL INTELLIGENCE TEST - VERBOSE MODE")
    log_separator()
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" API Key: {'Set' if API_KEY else 'Not Set'}")
    print(f" Endpoint: {API_URL}")
    print(f" Models: {len(MODELS)}")
    print(f" Scenarios: {len(SCENARIOS)}")
    log_separator()
    
    # Results storage
    all_results = {
        "metadata": {
            "test_date": datetime.now().isoformat(),
            "models": list(MODELS.keys()),
            "scenarios": list(SCENARIOS.keys()),
            "api_key_preview": "configured" if API_KEY else "missing"
        },
        "results": {}
    }
    
    # Statistics
    total_tests = len(MODELS) * len(SCENARIOS)
    completed_tests = 0
    successful_tests = 0
    total_time = 0
    
    print(f"\n Starting {total_tests} tests...\n")
    
    # Test each model
    for model_idx, (model_name, model_id) in enumerate(MODELS.items(), 1):
        print(f"\n{''*20}")
        print(f"MODEL {model_idx}/{len(MODELS)}: {model_name}")
        print(f"{''*20}")
        
        all_results["results"][model_name] = {
            "model_id": model_id,
            "tests": {}
        }
        
        # Test each scenario
        for scenario_idx, (scenario_key, scenario_data) in enumerate(SCENARIOS.items(), 1):
            completed_tests += 1
            print(f"\n Test {completed_tests}/{total_tests} - {scenario_data['name']}")
            
            # Make the API call
            result = query_model_verbose(
                model_name, 
                model_id,
                scenario_data['name'],
                scenario_data['prompt']
            )
            
            # Store result
            all_results["results"][model_name]["tests"][scenario_key] = result
            
            # Update statistics
            total_time += result.get('time', 0)
            if result['success']:
                successful_tests += 1
            
            # Progress update
            success_rate = (successful_tests / completed_tests) * 100
            print(f"\n Progress: {completed_tests}/{total_tests} tests")
            print(f" Success rate: {successful_tests}/{completed_tests} ({success_rate:.1f}%)")
            print(f"⏱  Total time so far: {total_time:.1f}s")
            
            # Rate limiting
            if completed_tests < total_tests:
                print(f"⏳ Waiting 2 seconds before next test...")
                time.sleep(2)
        
        # Extra delay between models
        if model_idx < len(MODELS):
            print(f"\n Waiting 3 seconds before next model...")
            time.sleep(3)
    
    # Save results
    log_separator("SAVING RESULTS")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_verbose_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f" Results saved to: {filename}")
    
    # Final summary
    log_separator("FINAL SUMMARY")
    print(f" Tests completed: {completed_tests}/{total_tests}")
    print(f" Successful: {successful_tests}")
    print(f" Failed: {completed_tests - successful_tests}")
    print(f" Success rate: {(successful_tests/completed_tests*100):.1f}%")
    print(f"⏱  Total time: {total_time:.1f} seconds")
    print(f"⏱  Average time per test: {total_time/completed_tests:.1f} seconds")
    
    # Model summary
    print("\n BY MODEL:")
    for model_name in MODELS.keys():
        model_tests = all_results["results"][model_name]["tests"]
        successes = sum(1 for t in model_tests.values() if t.get('success', False))
        print(f"   {model_name}: {successes}/{len(SCENARIOS)} successful")
    
    # Scenario summary  
    print("\n BY SCENARIO:")
    for scenario_key, scenario_data in SCENARIOS.items():
        successes = sum(
            1 for model_results in all_results["results"].values()
            if model_results["tests"].get(scenario_key, {}).get('success', False)
        )
        print(f"   {scenario_data['name']}: {successes}/{len(MODELS)} models succeeded")
    
    print("\n ALL DONE!")
    return all_results

if __name__ == "__main__":
    results = main()