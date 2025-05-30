#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_exam_report.py

Lee `results_transformed_to_A.csv`, `grades_all.csv` y `answersA.json` (o los archivos que se indiquen)
del directorio de trabajo y genera un PDF con métricas generales y por pregunta,
incluyendo distribución de respuestas por pregunta.
"""

import json
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pointbiserialr

def load_data(results_path: Path, grades_path: Path, answers_path: Path):
    # Respuestas ya mapeadas a Tema A
    df = pd.read_csv(results_path, dtype=str).fillna('-')
    # Calificaciones reales
    grades = pd.read_csv(grades_path, usecols=['file', 'grade'])
    grades['grade'] = pd.to_numeric(grades['grade'], errors='coerce').fillna(0.0)
    # Merge para añadir columna 'grade'
    df = df.merge(grades, on='file', how='left')
    # Respuestas correctas
    with answers_path.open(encoding='utf-8') as f:
        answers = json.load(f)
    return df, answers

def compute_item_stats(df, answers):
    """Para cada pregunta, calcula dificultad y discriminación."""
    n_questions = len(answers)
    diffs, discs = {}, {}
    grades = df['grade'].values

    for q in range(1, n_questions + 1):
        col = f'Q{q}'
        correct_set = [opt.strip() for opt in answers[str(q)].split(',') if opt.strip()]
        correct_flags = df[col].apply(lambda r: 1 if r in correct_set else 0).astype(int)

        diffs[q] = correct_flags.mean()
        if correct_flags.std() == 0 or grades.std() == 0:
            discs[q] = np.nan
        else:
            r_pb, _ = pointbiserialr(correct_flags, grades)
            discs[q] = r_pb

    return diffs, discs

def plot_overall(df, pdf):
    fig = plt.figure(figsize=(8, 10))
    fig.clf()
    stats = df['grade'].agg(['count', 'mean', 'median', 'std'])
    text = (
        f"Estudiantes: {int(stats['count'])}\n"
        f"Media: {stats['mean']:.2f}\n"
        f"Mediana: {stats['median']:.2f}\n"
        f"Desv. típica: {stats['std']:.2f}"
    )
    ax1 = fig.add_subplot(211); ax1.axis('off')
    ax1.text(0.1, 0.5, text, fontsize=12, va='center')
    ax2 = fig.add_subplot(212)
    ax2.hist(df['grade'], bins=8, edgecolor='black')
    ax2.set_xlabel('Calificación')
    ax2.set_ylabel('Número de estudiantes')
    ax2.set_title('Distribución de calificaciones')
    fig.tight_layout()
    pdf.savefig(fig)

def plot_item_difficulty(diffs, pdf):
    items = list(diffs.keys())
    vals  = [diffs[q] for q in items]
    order = np.argsort(vals)
    sorted_q = [items[i] for i in order]
    sorted_v = [vals[i]    for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([f"Q{q}" for q in sorted_q], sorted_v)
    ax.set_ylabel('Dificultad (% aciertos)')
    ax.set_title('Item dificultad (ordenadas)')
    ax.set_xticklabels([f"Q{q}" for q in sorted_q], rotation=90, fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    pdf.savefig(fig)

def plot_item_discrimination(discs, pdf):
    items = list(discs.keys())
    vals  = [discs[q] for q in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([f"Q{q}" for q in items], vals)
    ax.set_ylabel('Índice de discriminación\n(punto-biserial)')
    ax.set_title('Item discriminación')
    ax.set_xticklabels([f"Q{q}" for q in items], rotation=90, fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)

def plot_question_panels(df, answers, pdf):
    n_questions = len(answers)
    options = ['A','B','C','D','NR']
    ncols = 6
    nrows = math.ceil(n_questions / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*3, nrows*3), squeeze=False)
    axes = axes.flatten()

    for q in range(1, n_questions+1):
        ax = axes[q-1]
        vc = df[f'Q{q}'].value_counts()
        counts = [
            vc.get(opt, 0) if opt!='NR' else vc.get('-', 0)
            for opt in options
        ]
        correct = [o.strip() for o in answers[str(q)].split(',') if o.strip()]
        colors = ['green' if opt in correct else 'lightblue' for opt in options]
        ax.bar(options, counts, color=colors)
        ax.set_title(f'Pregunta {q}', fontsize=9)
        ax.set_ylim(0, len(df))
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    # Quitar ejes sobrantes
    for idx in range(n_questions, len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    pdf.savefig(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Genera un informe PDF con estadísticas de examen."
    )
    parser.add_argument(
        "-r", "--results",
        type=Path,
        default=Path("results_transformed_to_A.csv"),
        help="CSV de respuestas transformadas a Tema A (default: results_transformed_to_A.csv)"
    )
    parser.add_argument(
        "-g", "--grades",
        type=Path,
        default=Path("grades_all.csv"),
        help="CSV de calificaciones (default: grades_all.csv)"
    )
    parser.add_argument(
        "-a", "--answers",
        type=Path,
        default=Path("answersA.json"),
        help="JSON de respuestas correctas Tema A (default: answersA.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("exam_report.pdf"),
        help="Nombre del PDF de salida (default: exam_report.pdf)"
    )
    args = parser.parse_args()

    df, answers = load_data(args.results, args.grades, args.answers)
    diffs, discs = compute_item_stats(df, answers)

    with PdfPages(args.output) as pdf:
        plot_overall(df, pdf)
        plot_item_difficulty(diffs, pdf)
        plot_item_discrimination(discs, pdf)
        plot_question_panels(df, answers, pdf)

    print(f"✔ Reporte generado: {args.output}")

if __name__ == '__main__':
    main()

