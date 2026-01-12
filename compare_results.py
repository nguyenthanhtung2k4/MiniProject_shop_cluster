import json
import pandas as pd
import sys
import os
from io import StringIO

def extract_notebook_output(notebook_path):
    if not os.path.exists(notebook_path):
        print(f"File not found: {notebook_path}")
        return None

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    results = {}
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Tìm bảng Silhouette
            if "clusterer.choose_k_by_silhouette" in source and "sil_df" in source and "\nsil_df" in source:
                # Tìm output của cell này
                for output in cell['outputs']:
                    if 'text/plain' in output.get('data', {}):
                        text_content = "".join(output['data']['text/plain'])
                        results['silhouette_table'] = text_content
            
            # 2. Tìm K đã chọn
            if "print('Chosen k =', k)" in source:
                 for output in cell['outputs']:
                    if 'name' in output and output['name'] == 'stdout':
                        text_content = "".join(output['text']).strip()
                        results['chosen_k'] = text_content

            # 3. Tìm bảng Summary (Profiling)
            if "summary = meta_out.groupby" in source and "\nsummary" in source[-10:]:
                 for output in cell['outputs']:
                    if 'text/plain' in output.get('data', {}):
                        text_content = "".join(output['data']['text/plain'])
                        results['summary_table'] = text_content

    return results

def main():
    notebooks = [
        ("Baseline (Rules Only)", "notebooks/runs/clustering_from_rules_baseline_run.ipynb"),
        ("Advanced (Rules + RFM, KMeans)", "notebooks/runs/clustering_from_rules_advanced_run.ipynb"),
        ("Advanced (Rules + RFM, Hierarchical)", "notebooks/runs/clustering_from_rules_hierarchical_run.ipynb")
    ]

    print("=== SO SÁNH KẾT QUẢ PHÂN CỤM ===\n")

    for label, path in notebooks:
        print(f"--- {label} ---")
        res = extract_notebook_output(path)
        if res:
            print(f"1. Số cụm K được chọn: {res.get('chosen_k', 'N/A')}")
            print("\n2. Bảng Silhouette Score (Top đầu):")
            # Chỉ in vài dòng đầu của bảng silhouette để gọn
            sil_lines = res.get('silhouette_table', 'N/A').split('\n')
            for line in sil_lines[:6]:
                print(line)
            
            print("\n3. Profiling Summary (Sơ bộ):")
            sum_lines = res.get('summary_table', 'N/A').split('\n')
            for line in sum_lines[:10]: # In top 10 dòng
                print(line)
        else:
            print("Chưa có kết quả hoặc file chưa chạy xong.")
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
