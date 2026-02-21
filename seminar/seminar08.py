"""Модуль для вычисления метрик бинарной классификации из файлов меток."""
import csv
import numpy as np


def load_labels(filename):
    """Загружает метки из текстового файла."""
    with open(filename, 'r', encoding='utf-8') as file:
        return np.array([int(line.strip()) for line in file])


def compute_metrics(y_true, y_pred):
    """Вычисляет метрики классификации на основе истинных и предсказанных меток."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    acc = (tp + tn) / len(y_true)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return acc, tpr, fpr, tnr, fnr, prec, rec, f1


def save_metrics(metrics, filename):
    """Сохраняет вычисленные метрики в CSV файл."""
    headers = ['acc', 'tpr', 'fpr', 'tnr', 'fnr', 'prec', 'rec', 'f1']
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(metrics)


def main():
    """Основная функция для загрузки данных, вычисления метрик и их сохранения."""
    y_true = load_labels('y_true.txt')
    y_pred = load_labels('y_pred.txt')

    # Вычисление метрик
    metrics = compute_metrics(y_true, y_pred)

    # Сохранение метрик в CSV файл
    save_metrics(metrics, 'seminar08_metrics.csv')


if __name__ == '__main__':
    main()
