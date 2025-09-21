from importlib import util

spec = util.spec_from_file_location("ingest", "src/ingest_and_chunk.py")
ing = util.module_from_spec(spec)
spec.loader.exec_module(ing)

docs = ing.ingest_folder("data")
print("Found", len(docs), "chunks")
print("Sources:", sorted(set(d['source'] for d in docs)))
