#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_results.py

Lee `temas_mapping.json` y `results_all.csv` (o los archivos que se especifiquen)
y genera `results_transformed_to_A.csv` (o el nombre de salida que se indique)
donde todas las respuestas están en el orden y codificación de Tema A.
"""

import json
import argparse
from pathlib import Path
import pandas as pd

def load_mapping(mapping_path: Path):
    with mapping_path.open(encoding='utf-8') as f:
        return json.load(f)

def build_inverted_options(mapping):
    inv_opts = {}
    for item in mapping:
        b = item['temaB']
        inv_opts[b] = {
            letra_b: letra_a
            for letra_a, letra_b in item['opciones'].items()
        }
    return inv_opts

def transform_row(row, mapping, inverted_opts, n_questions=42):
    new = {
        'file': row['file'],
        'tema': 'A'
    }
    if row['tema'] == 'A':
        # Copiamos Q1..Q{n_questions} tal cual
        for i in range(1, n_questions+1):
            col = f'Q{i}'
            new[col] = row.get(col, '')
    elif row['tema'] == 'B':
        # Para cada ítem en mapping, traducimos de B a A
        for item in mapping:
            a = item['temaA']
            b = item['temaB']
            raw = row.get(f'Q{b}', '')
            if raw == '' or raw == '-':
                new[f'Q{a}'] = raw
            else:
                new[f'Q{a}'] = inverted_opts[b].get(raw, raw)
    else:
        raise ValueError(f"Fila con tema desconocido: {row['tema']}")
    return new

def main():
    parser = argparse.ArgumentParser(
        description="Transforma un CSV de resultados de Tema A/B a la codificación de Tema A."
    )
    parser.add_argument(
        '--mapping', '-m',
        type=Path,
        default=Path('temas_mapping.json'),
        help="Path al JSON de mapping (default: temas_mapping.json)"
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('results_all.csv'),
        help="CSV de entrada con columna 'tema' (default: results_all.csv)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results_transformed_to_A.csv'),
        help="CSV de salida transformado (default: results_transformed_to_A.csv)"
    )
    parser.add_argument(
        '--questions', '-q',
        type=int,
        default=42,
        help="Número de preguntas (default: 42)"
    )
    args = parser.parse_args()

    # Cargar mapping y preparar opciones invertidas
    mapping = load_mapping(args.mapping)
    inverted_opts = build_inverted_options(mapping)

    # Leer CSV de resultados
    df = pd.read_csv(args.input, dtype=str).fillna('')

    # Transformar fila por fila
    output_rows = [
        transform_row(row, mapping, inverted_opts, args.questions)
        for _, row in df.iterrows()
    ]

    # Columnas en el orden deseado: file, Q1..Q{n}, tema
    cols = ['file'] + [f'Q{i}' for i in range(1, args.questions+1)] + ['tema']
    df_out = pd.DataFrame(output_rows, columns=cols)

    # Guardar resultado
    df_out.to_csv(args.output, index=False, encoding='utf-8')
    print(f"✔ Generado {args.output}")

if __name__ == '__main__':
    main()

