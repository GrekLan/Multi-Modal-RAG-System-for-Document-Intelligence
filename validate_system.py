"""
System Validation Script
Run before demo to ensure everything works.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(passed, message):
    icon = "✅" if passed else "❌"
    status = "PASS" if passed else "FAIL"
    print(f"{icon} [{status}] {message}")
    return passed

def validate_system_dependencies():
    print_header("1. System Dependencies")
    all_passed = True
    
    # Python version
    python_version = sys.version_info
    passed = python_version >= (3, 8)
    all_passed &= print_check(
        passed,
        f"Python: {python_version.major}.{python_version.minor} (Required: 3.8+)"
    )
    
    # Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print_check(True, f"Tesseract OCR: {version}")
    except Exception as e:
        all_passed &= print_check(False, "Tesseract OCR: Not found")
        print(f"   Install: https://github.com/UB-Mannheim/tesseract/wiki")
    
    return all_passed

def validate_python_packages():
    print_header("2. Python Packages")
    all_passed = True
    
    packages = [
        ('streamlit', 'streamlit'),
        ('fitz', 'pymupdf'),
        ('pytesseract', 'pytesseract'),
        ('PIL', 'Pillow'),
        ('langchain', 'langchain'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('torch', 'torch'),
        ('numpy', 'numpy')
    ]
    
    for module_name, package_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print_check(True, f"{package_name}: {version}")
        except ImportError:
            all_passed &= print_check(False, f"{package_name}: Not installed")
            print(f"   Install: pip install {package_name}")
    
    return all_passed

def validate_directory_structure():
    print_header("3. Directory Structure")
    all_passed = True
    
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/vector_store',
        'data/images'
    ]
    
    for dir_path in directories:
        exists = os.path.exists(dir_path)
        all_passed &= print_check(exists, dir_path)
        if not exists:
            print(f"   Run: python config.py")
    
    return all_passed

def validate_document_processing():
    print_header("4. Document Processing")
    all_passed = True
    
    import config
    
    pdf_exists = os.path.exists(config.PDF_PATH)
    all_passed &= print_check(pdf_exists, f"PDF: {config.PDF_PATH}")
    
    if pdf_exists:
        size_mb = os.path.getsize(config.PDF_PATH) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
    
    chunks_exist = os.path.exists(config.CHUNKS_PATH)
    all_passed &= print_check(chunks_exist, f"Chunks: {config.CHUNKS_PATH}")
    
    if chunks_exist:
        import json
        with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"   Total: {len(chunks)} chunks")
    else:
        print(f"   Run: python process_document.py")
    
    return all_passed

def validate_vector_store():
    print_header("5. Vector Store")
    all_passed = True
    
    import config
    
    # Check multiple possible locations
    faiss_path = f"{config.VECTOR_STORE_PATH}.faiss"
    pkl_path = f"{config.VECTOR_STORE_PATH}_chunks.pkl"
    
    # Also check without extension (some systems)
    alt_faiss_path = os.path.join(config.VECTOR_STORE_DIR, "index.faiss")
    
    faiss_exists = os.path.exists(faiss_path) or os.path.exists(alt_faiss_path)
    pkl_exists = os.path.exists(pkl_path)
    
    all_passed &= print_check(faiss_exists, f"FAISS Index")
    all_passed &= print_check(pkl_exists, f"Chunks Pickle")
    
    if faiss_exists and pkl_exists:
        try:
            from vector_store import VectorStore
            vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
            vector_store.load(config.VECTOR_STORE_PATH)
            print_check(True, f"Loaded {len(vector_store.chunks)} vectors")
        except Exception as e:
            all_passed &= print_check(False, f"Error loading: {str(e)[:50]}")
    else:
        print(f"   Run: python create_embeddings.py")
        if not faiss_exists:
            print(f"   Missing: {faiss_path}")
        if not pkl_exists:
            print(f"   Missing: {pkl_path}")
    
    return all_passed

def validate_llm():
    print_header("6. Language Model")
    all_passed = True
    
    import config
    
    try:
        from llm_qa import LLMQA
        qa_system = LLMQA(model_name=config.LLM_MODEL)
        print_check(True, "FLAN-T5 loaded")
    except Exception as e:
        all_passed &= print_check(False, f"LLM error: {e}")
        print("   Will use SimpleQA fallback")
    
    return all_passed

def test_end_to_end():
    print_header("7. End-to-End Test")
    all_passed = True
    
    try:
        import config
        from vector_store import VectorStore
        from llm_qa import LLMQA, SimpleQA
        
        print("   Loading system...")
        vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
        vector_store.load(config.VECTOR_STORE_PATH)
        
        try:
            qa_system = LLMQA(model_name=config.LLM_MODEL)
        except:
            qa_system = SimpleQA()
        
        test_query = "What is the main topic?"
        print(f"   Testing: '{test_query}'")
        
        search_results = vector_store.search(test_query, k=3)
        print_check(len(search_results) > 0, f"Retrieval: {len(search_results)} results")
        
        result = qa_system.generate_answer_with_citations(test_query, search_results)
        has_answer = len(result['answer']) > 0
        
        all_passed &= print_check(has_answer, f"Answer: {len(result['answer'])} chars")
        
        if has_answer:
            print(f"\n   Sample: {result['answer'][:150]}...")
        
    except Exception as e:
        all_passed &= print_check(False, f"Test failed: {e}")
    
    return all_passed

def print_summary(results):
    print_header("SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%\n")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. streamlit run app.py")
        print("  2. Test queries")
        print("  3. Record demo")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nFailed:")
        for check_name, passed in results.items():
            if not passed:
                print(f"  ❌ {check_name}")
        
        print("\nFixes:")
        print("  pip install -r requirements.txt")
        print("  python run_pipeline.py")

def main():
    print("\n" + "="*70)
    print("  SYSTEM VALIDATION")
    print("="*70)
    
    results = {}
    results['System Dependencies'] = validate_system_dependencies()
    results['Python Packages'] = validate_python_packages()
    results['Directory Structure'] = validate_directory_structure()
    results['Document Processing'] = validate_document_processing()
    results['Vector Store'] = validate_vector_store()
    results['Language Model'] = validate_llm()
    results['End-to-End Test'] = test_end_to_end()
    
    print_summary(results)
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()