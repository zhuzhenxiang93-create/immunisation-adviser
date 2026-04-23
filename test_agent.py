import sys, json
sys.path.insert(0, '.')
from agent.query_handler import run_query

queries = [
    "When should the MMR vaccine be given to a 12-month-old child in New Zealand?",
    "Is the influenza vaccine safe for a patient with egg allergy?",
    "How should varicella vaccine be stored?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    result = run_query(q)
    print(result['formatted'])
    print(f"\n[chunks retrieved: {result['output']['audit']['chunks_retrieved']}, confidence: {result['output']['confidence']}]")
