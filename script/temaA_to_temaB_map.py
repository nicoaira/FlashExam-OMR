#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_answersB.py

Lee `answersA.json` y `temas_mapping.json` (o los archivos que se especifiquen)
y genera `answersB.json` (o el nombre de salida indicado) con las respuestas
correctas recodificadas para Tema B.
"""

import json
import argparse
from pathlib import Path

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_answers_b(answers_a: dict, mapping: list):
    answers_b = {}
    for entry in mapping:
        pregunta_a = str(entry["temaA"])
        pregunta_b = str(entry["temaB"])
        opciones_map = entry["opciones"]

        if pregunta_a not in answers_a:
            continue  # no hay respuesta para esta pregunta en A

        opcion_correcta_a = answers_a[pregunta_a]
        # traducir a la opción correspondiente en B
        opcion_correcta_b = opciones_map.get(opcion_correcta_a, opcion_correcta_a)
        answers_b[pregunta_b] = opcion_correcta_b

    return answers_b

def main():
    parser = argparse.ArgumentParser(
        description="Genera answersB.json a partir de answersA.json y el mapping de temas."
    )
    parser.add_argument(
        "-a", "--answers-a",
        type=Path,
        default=Path("answersA.json"),
        help="JSON de respuestas de Tema A (default: answersA.json)"
    )
    parser.add_argument(
        "-m", "--mapping",
        type=Path,
        default=Path("temas_mapping.json"),
        help="JSON de mapping de temas (default: temas_mapping.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("answersB.json"),
        help="JSON de salida para respuestas de Tema B (default: answersB.json)"
    )
    args = parser.parse_args()

    # Cargar datos
    answers_a = load_json(args.answers_a)
    mapping   = load_json(args.mapping)

    # Generar respuestas B
    answers_b = build_answers_b(answers_a, mapping)

    # Guardar resultado
    save_json(answers_b, args.output)
    print(f"✔ Generado {args.output}")

if __name__ == "__main__":
    main()

