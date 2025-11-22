"""
Evaluation Suite for Multi-Modal RAG System
"""

import time
import json
import os
from typing import List, Dict, Any
import numpy as np
from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

class RAGEvaluator:
    def __init__(self, vector_store: VectorStore, qa_system):
        self.vector_store = vector_store
        self.qa_system = qa_system
        self.results = {}
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create evaluation queries."""
        return [
            {
                'query': 'What is the GDP growth rate mentioned in the document?',
                'type': 'factual',
                'expected_modality': 'text',
                'complexity': 'simple'
            },
            {
                'query': 'What are the key economic indicators shown in the tables?',
                'type': 'factual',
                'expected_modality': 'table',
                'complexity': 'medium'
            },
            {
                'query': 'Describe the trends shown in the charts or graphs.',
                'type': 'analytical',
                'expected_modality': 'image',
                'complexity': 'medium'
            },
            {
                'query': 'What are the IMF recommendations for Qatar?',
                'type': 'factual',
                'expected_modality': 'text',
                'complexity': 'simple'
            },
            {
                'query': 'Compare the fiscal indicators across different years.',
                'type': 'comparative',
                'expected_modality': 'table',
                'complexity': 'complex'
            },
            {
                'query': 'What is the inflation rate and banking sector health?',
                'type': 'multi-part',
                'expected_modality': 'text',
                'complexity': 'medium'
            },
            {
                'query': 'Summarize the key risks mentioned in the document.',
                'type': 'summarization',
                'expected_modality': 'text',
                'complexity': 'complex'
            },
            {
                'query': 'What information is provided about the non-hydrocarbon sector?',
                'type': 'factual',
                'expected_modality': 'text',
                'complexity': 'medium'
            }
        ]
    
    def evaluate_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Evaluate retrieval quality."""
        start_time = time.time()
        results = self.vector_store.search(query, k=k)
        retrieval_time = time.time() - start_time
        
        modalities = [r['chunk']['type'] for r in results]
        modality_distribution = {
            'text': modalities.count('text'),
            'table': modalities.count('table'),
            'image': modalities.count('image')
        }
        
        counts = list(modality_distribution.values())
        total = sum(counts)
        if total > 0:
            probs = [c/total for c in counts if c > 0]
            diversity = -sum(p * np.log(p) for p in probs)
        else:
            diversity = 0
        
        return {
            'retrieval_time': retrieval_time,
            'num_results': len(results),
            'modality_distribution': modality_distribution,
            'diversity_score': diversity,
            'avg_relevance': np.mean([r['score'] for r in results]) if results else 0,
            'results': results
        }
    
    def evaluate_answer_generation(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Evaluate answer generation."""
        start_time = time.time()
        result = self.qa_system.generate_answer_with_citations(query, search_results)
        generation_time = time.time() - start_time
        
        answer = result['answer']
        answer_length = len(answer)
        has_citations = len(result['citations']) > 0
        citation_coverage = len(result['citations']) / len(search_results) if search_results else 0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words & answer_words) / len(query_words) if query_words else 0
        
        return {
            'generation_time': generation_time,
            'answer_length': answer_length,
            'has_citations': has_citations,
            'num_citations': len(result['citations']),
            'citation_coverage': citation_coverage,
            'query_term_coverage': query_coverage,
            'answer': answer,
            'citations': result['citations']
        }
    
    def evaluate_single_query(self, test_case: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        """Run complete evaluation for a single query."""
        query = test_case['query']
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Type: {test_case['type']} | Complexity: {test_case['complexity']}")
        print(f"{'='*80}")
        
        retrieval_eval = self.evaluate_retrieval(query, k=k)
        print(f"\n📊 Retrieval Metrics:")
        print(f"   ⏱️  Time: {retrieval_eval['retrieval_time']:.3f}s")
        print(f"   📝 Results: {retrieval_eval['num_results']}")
        print(f"   🎯 Avg Relevance: {retrieval_eval['avg_relevance']:.3f}")
        print(f"   🌈 Diversity: {retrieval_eval['diversity_score']:.3f}")
        
        answer_eval = self.evaluate_answer_generation(query, retrieval_eval['results'])
        print(f"\n💬 Answer Metrics:")
        print(f"   ⏱️  Time: {answer_eval['generation_time']:.3f}s")
        print(f"   📏 Length: {answer_eval['answer_length']} chars")
        print(f"   📚 Citations: {answer_eval['num_citations']}")
        print(f"\n   Answer: {answer_eval['answer'][:200]}...")
        
        total_time = retrieval_eval['retrieval_time'] + answer_eval['generation_time']
        expected_mod = test_case['expected_modality']
        modality_found = retrieval_eval['modality_distribution'].get(expected_mod, 0) > 0
        
        return {
            'query': query,
            'test_type': test_case['type'],
            'complexity': test_case['complexity'],
            'retrieval': retrieval_eval,
            'answer': answer_eval,
            'total_time': total_time,
            'expected_modality_found': modality_found,
            'overall_score': self._calculate_overall_score(retrieval_eval, answer_eval, modality_found)
        }
    
    def _calculate_overall_score(self, retrieval_eval: Dict, answer_eval: Dict, modality_found: bool) -> float:
        """Calculate overall score."""
        retrieval_score = min(retrieval_eval['avg_relevance'] / 10, 1.0)
        diversity_score = min(retrieval_eval['diversity_score'] / 2, 1.0)
        citation_score = answer_eval['citation_coverage']
        coverage_score = answer_eval['query_term_coverage']
        modality_score = 1.0 if modality_found else 0.5
        
        score = (
            0.3 * retrieval_score +
            0.2 * diversity_score +
            0.2 * citation_score +
            0.2 * coverage_score +
            0.1 * modality_score
        )
        
        return score
    
    def run_full_evaluation(self, k: int = 5) -> Dict[str, Any]:
        """Run evaluation on all test queries."""
        print("\n" + "="*80)
        print("🚀 STARTING EVALUATION")
        print("="*80)
        
        test_queries = self.create_test_queries()
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n[Test {i}/{len(test_queries)}]")
            result = self.evaluate_single_query(test_case, k=k)
            results.append(result)
            time.sleep(0.5)
        
        self.results = self._aggregate_results(results)
        self._print_summary()
        
        return self.results
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results."""
        return {
            'num_queries': len(results),
            'avg_retrieval_time': np.mean([r['retrieval']['retrieval_time'] for r in results]),
            'avg_generation_time': np.mean([r['answer']['generation_time'] for r in results]),
            'avg_total_time': np.mean([r['total_time'] for r in results]),
            'avg_relevance_score': np.mean([r['retrieval']['avg_relevance'] for r in results]),
            'avg_diversity_score': np.mean([r['retrieval']['diversity_score'] for r in results]),
            'avg_citation_coverage': np.mean([r['answer']['citation_coverage'] for r in results]),
            'avg_overall_score': np.mean([r['overall_score'] for r in results]),
            'modality_accuracy': sum(r['expected_modality_found'] for r in results) / len(results),
            'detailed_results': results
        }
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        r = self.results
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Avg Retrieval Time: {r['avg_retrieval_time']:.3f}s")
        print(f"   Avg Generation Time: {r['avg_generation_time']:.3f}s")
        print(f"   Avg Total Time: {r['avg_total_time']:.3f}s")
        
        print(f"\n🎯 QUALITY:")
        print(f"   Avg Relevance: {r['avg_relevance_score']:.3f}")
        print(f"   Avg Diversity: {r['avg_diversity_score']:.3f}")
        print(f"   Citation Coverage: {r['avg_citation_coverage']:.2%}")
        print(f"   Modality Accuracy: {r['modality_accuracy']:.2%}")
        print(f"   Overall Score: {r['avg_overall_score']:.3f}")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath: str = 'evaluation_results.json'):
        """Save results to file."""
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results_serializable = convert_types(self.results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filepath}")


def main():
    print("="*80)
    print("🔬 MULTI-MODAL RAG EVALUATION")
    print("="*80)
    
    # Check multiple possible FAISS locations
    faiss_paths = [
        f"{config.VECTOR_STORE_PATH}.faiss",
        os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    ]
    
    faiss_exists = any(os.path.exists(path) for path in faiss_paths)
    
    if not faiss_exists:
        print("\n❌ Error: Vector store not found!")
        print("Please run: python run_pipeline.py")
        return
    
    print("\n📚 Loading vector store...")
    vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
    vector_store.load(config.VECTOR_STORE_PATH)
    print(f"✓ Loaded {len(vector_store.chunks)} chunks")
    
    print("\n🤖 Loading QA system...")
    try:
        qa_system = LLMQA(model_name=config.LLM_MODEL)
    except:
        print("⚠️  Using SimpleQA")
        qa_system = SimpleQA()
    
    evaluator = RAGEvaluator(vector_store, qa_system)
    results = evaluator.run_full_evaluation(k=5)
    evaluator.save_results('evaluation_results.json')
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()