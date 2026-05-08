#!/usr/bin/env python3
import ast
import sys
import importlib.util
import json
import re
from pathlib import Path

def is_stdlib(module_name: str) -> bool:
    """Check if a module is part of the Python Standard Library."""
    # Handle submodules (e.g., urllib.request -> urllib)
    base_name = module_name.split('.')[0]
    return base_name in sys.stdlib_module_names

def run_deep_audit(root_dir="."):
    files = list(Path(root_dir).rglob("*.py"))
    
    # Exclude virtual environments or hidden folders
    files = [f for f in files if "venv" not in f.parts and not f.name.startswith(".")]
    
    results = {}
    all_endpoints = {}
    all_models = {}
    all_local_modules = [f.stem for f in files]
    
    # We will track every string that looks like a URL route
    # to cross-reference against actual endpoints.
    global_api_calls = []

    for f in files:
        file_result = {
            "status": "ok",
            "imports_stdlib": [],
            "imports_local": [],
            "imports_3rd_party_found": [],
            "imports_3rd_party_missing": [],
            "endpoints": [],
            "models": [],
            "api_calls_made": []
        }
        
        try:
            source = f.read_text(encoding="utf-8")
            tree = ast.parse(source)
            
            # --- DEEP ANALYSIS: Disconnected API Call Tracing ---
            # Find any string that looks like an API route (e.g., '/command', '/health')
            found_urls = re.findall(r'[\'"](/[\w/-]+)[\'"]', source)
            file_result["api_calls_made"] = list(set(found_urls))
            global_api_calls.extend(found_urls)

            for node in ast.walk(tree):
                
                # --- FIX FLAW 1: Detect both sync AND async functions ---
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in node.decorator_list:
                        d_str = ast.unparse(decorator)
                        if any(method in d_str for method in ["get", "post", "put", "delete", "websocket"]):
                            ep_info = {
                                "name": node.name, 
                                "decorator": d_str, 
                                "is_async": isinstance(node, ast.AsyncFunctionDef)
                            }
                            file_result["endpoints"].append(ep_info)
                            all_endpoints[node.name] = {"file": f.name, **ep_info}

                # --- Extract Pydantic Gatekeepers ---
                if isinstance(node, ast.ClassDef):
                    bases = [ast.unparse(b) for b in node.bases]
                    if any("BaseModel" in b for b in bases):
                        fields = [ast.unparse(item.target) for item in node.body if isinstance(item, ast.AnnAssign)]
                        model_info = {"class": node.name, "fields": fields}
                        file_result["models"].append(model_info)
                        all_models[node.name] = {"file": f.name, **model_info}

                # --- Extract Imports ---
                imports_found = []
                if isinstance(node, ast.Import):
                    imports_found.extend([n.name for n in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_found.append(node.module)
                
                for imp in set(imports_found):
                    if not imp or imp.startswith("_"): continue
                    
                    base_imp = imp.split('.')[0]
                    
                    # --- FIX FLAW 2: Smart Categorization ---
                    if is_stdlib(base_imp):
                        file_result["imports_stdlib"].append(imp)
                    elif base_imp in all_local_modules:
                        file_result["imports_local"].append(imp)
                    else:
                        # It is a 3rd party dependency. Let's strictly check if it's installed.
                        try:
                            spec = importlib.util.find_spec(base_imp)
                            # find_spec returns None if missing, instead of crashing
                            if spec is None:
                                file_result["imports_3rd_party_missing"].append(base_imp)
                            else:
                                file_result["imports_3rd_party_found"].append(base_imp)
                        except Exception:
                            # If it crashes, it's genuinely missing or broken
                            file_result["imports_3rd_party_missing"].append(base_imp)
                            
        except SyntaxError as e:
            file_result["status"] = f"SYNTAX ERROR: {e}"
        except Exception as e:
            file_result["status"] = f"ERROR: {e}"

        results[str(f)] = file_result

    # --- CROSS-REFERENCE LOGIC (Finding Disconnected Nerves) ---
    exposed_routes = []
    for ep in all_endpoints.values():
        m = re.search(r'[\'"](/[\w/-]+)[\'"]', ep['decorator'])
        if m:
            exposed_routes.append(m.group(1))

    global_api_calls = list(set(global_api_calls))
    
    # A route exists in FastAPI, but nothing in the Python code ever calls it
    uncalled_endpoints = [r for r in exposed_routes if r not in global_api_calls]
    
    # A Python script tries to call a route, but no FastAPI server is hosting it
    dead_calls = [c for c in global_api_calls if c not in exposed_routes and len(c) > 1]

    # --- Final Report Generation ---
    report = {
        "files_scanned": len(files),
        "deep_analysis": {
            "total_exposed_routes": list(set(exposed_routes)),
            "total_internal_api_calls": global_api_calls,
            "WARNING_uncalled_endpoints": uncalled_endpoints,
            "WARNING_dead_api_calls": dead_calls
        },
        "endpoints_found": all_endpoints,
        "models_found": all_models,
        "missing_dependencies": {},
        "per_file": results
    }

    # Clean up the missing dependencies list for the report
    for f, data in results.items():
        if data["imports_3rd_party_missing"]:
            report["missing_dependencies"][f] = list(set(data["imports_3rd_party_missing"]))

    Path("deep_audit_report.json").write_text(json.dumps(report, indent=2))
    
    print("\n" + "="*50)
    print(" 🌌 SWAYAMBHU DEEP ARCHITECTURE AUDIT COMPLETE")
    print("="*50)
    print(f" Files Scanned           : {len(files)}")
    print(f" Endpoints Exposed       : {len(exposed_routes)}")
    print(f" Pydantic Models Found   : {len(all_models)}")
    print(f" ⚠️ Disconnected Routes  : {len(dead_calls)} 'Dead Calls' found.")
    print(f" ⚠️ Missing Dependencies : {len(report['missing_dependencies'])} files affected.")
    print("="*50)
    print(" Detailed cross-reference saved to: deep_audit_report.json\n")

if __name__ == '__main__':
    run_deep_audit()