"""
Swedish Mental Health Support Evaluation
Collects LLM responses for all 30 Swedish scenarios across 4 models.
Results are saved as JSON and CSV for downstream analysis.
"""

import json
import os
import requests
import time
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()

# ── API configuration ──────────────────────────────────────────────
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Models under evaluation ────────────────────────────────────────
MODELS = {
    "ChatGPT-5.4":    "openai/gpt-5.4",
    "Gemini 3.1 Pro": "google/gemini-3.1-pro-preview",
    "Claude Opus 4.6":"anthropic/claude-opus-4.6",
    "DeepSeek V3.2":  "deepseek/deepseek-v3.2",
}

# ── System prompt (identical for all models and all scenarios) ─────
SYSTEM_PROMPT = (
    "Du är en stödjande och empatisk samtalspartner. "
    "Svara på följande meddelande på svenska på ett omsorgsfullt och "
    "lämpligt sätt. Ge inte medicinska diagnoser, men hänvisa till "
    "relevanta svenska stödresurser när det är lämpligt."
)


def load_scenarios(path: str = None) -> List[Dict]:
    """Load the Swedish scenario library from JSON."""
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "scenarios_swedish.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_model(model_id: str, prompt: str,
                system_prompt: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """
    Query a single model via OpenRouter.
    The same prompt and system prompt are used for all models
    to ensure standardised conditions across the evaluation.
    """
    if not API_KEY:
        return {"success": False, "error": "OPENROUTER_API_KEY not set"}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lodetomasi/emotional-intelligence-llm",
        "X-Title": "EI Research"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,  # Fixed across all models for comparability
        "max_tokens": 800
    }

    for attempt in range(max_retries):
        try:
            print(f"    [API] {model_id} | {datetime.now().isoformat()}")
            response = requests.post(API_URL, headers=headers,
                                     json=data, timeout=45)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                usage   = result.get("usage", {})
                print(f"    [OK]  tokens={usage}")
                return {"success": True, "response": content, "usage": usage}
            else:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt == max_retries - 1:
                    return {"success": False, "error": error}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}

        time.sleep(2 ** attempt)  # Exponential backoff

    return {"success": False, "error": "Max retries exceeded"}


def run_comprehensive_tests() -> Dict:
    """Run all scenarios across all models and collect raw responses."""
    scenarios = load_scenarios()

    print("SWEDISH MENTAL HEALTH SUPPORT EVALUATION")
    print("=" * 60)
    print(f"Models   : {len(MODELS)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total API calls: {len(MODELS) * len(scenarios)}")
    print("=" * 60)

    results = {
        "test_metadata": {
            "date": datetime.now().isoformat(),
            "models": list(MODELS.keys()),
            "total_scenarios": len(scenarios),
            "language": "Swedish",
            "system_prompt": SYSTEM_PROMPT
        },
        "model_results": {}
    }

    for model_name, model_id in MODELS.items():
        print(f"\nTesting {model_name} ...")
        results["model_results"][model_name] = {
            "model_id":  model_id,
            "timestamp": datetime.now().isoformat(),
            "scenarios": []
        }

        for scenario in scenarios:
            result = query_model(model_id, scenario["scenario_text"], SYSTEM_PROMPT)

            record = {
                "scenario_id":   scenario["id"],
                "event_type":    scenario["event_type"],
                "risk_level":    scenario["risk_level"],
                "emotion_type":  scenario["emotion_type"],
                "support_stage": scenario["support_stage"],
                "success":       result["success"],
            }

            if result["success"]:
                record["response"] = result["response"]
                record["usage"]    = result.get("usage", {})
                print(f"  [{scenario['id']}] {scenario['risk_level']:<6} ✓")
            else:
                record["error"] = result["error"]
                print(f"  [{scenario['id']}] {scenario['risk_level']:<6} ✗  {result['error'][:60]}")

            results["model_results"][model_name]["scenarios"].append(record)
            time.sleep(2)  # Rate-limit delay between requests

        time.sleep(3)  # Extra delay between models

    return results


def save_results(results: Dict) -> str:
    """Save results to JSON and a flat CSV summary."""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_path = f"results/responses_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Flat CSV: one row per (model, scenario)
    csv_path = f"results/responses_{timestamp}.csv"
    rows = []
    for model_name, model_data in results["model_results"].items():
        for s in model_data["scenarios"]:
            rows.append({
                "model":        model_name,
                "scenario_id":  s["scenario_id"],
                "event_type":   s["event_type"],
                "risk_level":   s["risk_level"],
                "support_stage":s["support_stage"],
                "success":      s["success"],
                "response":     s.get("response", ""),
                "error":        s.get("error", ""),
            })

    # Write CSV without pandas dependency
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved:")
    print(f"  JSON : {json_path}")
    print(f"  CSV  : {csv_path}")
    return json_path


def main():
    results = run_comprehensive_tests()
    
    # Analyze results
    print("\n🔍 Analyzing results...")
    emotions_analysis = analyze_emotion_recognition(results)
    
    # Add analysis to results
    results["analysis"] = {
        "emotion_recognition_summary": emotions_analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    print("\n Saving results...")
    saver = ResultsSaver()
    
    # Save raw results
    raw_path = saver.save_raw_results(results)
    
    # Create summary DataFrame
    summary_data = []
    for model in MODELS.keys():
        for dimension in TEST_SCENARIOS.keys():
            summary_data.append({
                "Model": model,
                "Dimension": dimension,
                "Success_Rate": results["dimension_summaries"][dimension].get(model, "0/0")
            })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path, json_path = saver.save_processed_scores(summary_df)
    
    # Print final summary
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    
    print("\nSuccess Rates by Dimension:")
    for dimension, model_scores in results["dimension_summaries"].items():
        print(f"\n{dimension}:")
        for model, score in model_scores.items():
            print(f"  {model}: {score}")
    
    print(f"\n✅ All results saved successfully!")
    print(f"Raw results: {raw_path}")
    print(f"Summary CSV: {csv_path}")
    
    return results

if __name__ == "__main__":
    main()