"""FNN-GSO entegrasyonu: Energy Efficiency (regresyon) ve Seeds (sınıflandırma)."""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from course_core import (
    Config, examples, prepare_data, init_fnn,
    train_meta_fnn, predict, classification_metrics, rmse
)
from optimizers import optimizer_gso, optimizer_ga, optimizer_pso


def regression_metrics_full(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse_val = float(rmse(y_pred, y_true))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return dict(mae=mae, rmse=rmse_val, r2=r2)


def classification_metrics_full(y_true, pred_scores, classes):
    # Depodaki hazır metrikleri al (accuracy, macro_f1, balanced_accuracy, confusion_matrix)
    base = classification_metrics(y_true, pred_scores, classes)

    # Karışıklık matrisinden macro Precision ve macro Recall hesapla
    y_true_arr = np.asarray(y_true).ravel()
    classes_arr = np.asarray(classes)
    pred_idx = np.argmax(pred_scores, axis=1)
    y_pred_arr = classes_arr[pred_idx]

    k = len(classes_arr)
    cm = np.zeros((k, k), dtype=int)
    for idx_t, ct in enumerate(classes_arr):
        for idx_p, cp in enumerate(classes_arr):
            cm[idx_t, idx_p] = int(np.sum((y_true_arr == ct) & (y_pred_arr == cp)))

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_c = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

    base['macro_precision'] = float(np.mean(precision_c))
    base['macro_recall'] = float(np.mean(recall_c))
    base['confusion_matrix_verified'] = cm
    return base


def plot_convergence(histories, dataset_name, out_dir):
    plt.figure(figsize=(7, 4.5))
    for label, hist in histories.items():
        evals = np.arange(1, len(hist) + 1)
        plt.plot(evals, hist, linewidth=2, label=label)
    plt.xlabel('Amaç Fonksiyonu Değerlendirme Sayısı (FEs)')
    plt.ylabel('En İyi Doğrulama RMSE')
    plt.title(f'Yakınsama Grafiği — {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'convergence_{dataset_name}.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'  -> Grafik kaydedildi: {out_path}')


def main():
    out_dir = 'results_gso'
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config()
    target_names = {'energy_heating', 'seeds'}
    selected_datasets = [ex for ex in examples() if ex['name'] in target_names]

    optimizers = {
        'FNN-GSO': optimizer_gso,
        'FNN-GA': optimizer_ga,
        'FNN-PSO': optimizer_pso,
    }

    # İlk çalışan entegrasyon için cfg.seeds içindeki ilk tohumu (veya tümünü) kullan
    seed = cfg.seeds[0]

    for ex in selected_datasets:
        print(f"\n=== Veri Seti: {ex['name']} ({ex['task']}) | Seed: {seed} ===")
        data = prepare_data(**{k: v for k, v in ex.items() if k != 'name'})
        histories = {}

        for model_name, opt_fn in optimizers.items():
            initial = init_fnn(data['Xtrain'], cfg.rules, 1, seed)

            t0 = time.perf_counter()
            model, search = train_meta_fnn(
                initial,
                data['Xtrain'], data['ytrain'],
                data['Xval'], data['yval'],
                cfg, seed, opt_fn
            )
            elapsed = time.perf_counter() - t0

            pred = predict(model, data['Xtest']) * data['ystd'] + data['ymean']
            histories[model_name] = search['history']

            if ex['task'] == 'classification':
                m = classification_metrics_full(data['ytest'], pred, data['classes'])
                print(
                    f"  {model_name:8s} | Acc={m.get('accuracy', np.nan):.4f} "
                    f"| MacroP={m['macro_precision']:.4f} "
                    f"| MacroR={m['macro_recall']:.4f} "
                    f"| MacroF1={m.get('macro_f1', np.nan):.4f} "
                    f"| Evals={search['evals']} | Süre={elapsed:.2f}s"
                )
                print(f"    Karışıklık Matrisi:\n{m['confusion_matrix_verified']}")
            else:
                m = regression_metrics_full(data['ytest'], pred)
                print(
                    f"  {model_name:8s} | MAE={m['mae']:.4f} "
                    f"| RMSE={m['rmse']:.4f} "
                    f"| R2={m['r2']:.4f} "
                    f"| Evals={search['evals']} | Süre={elapsed:.2f}s"
                )

        plot_convergence(histories, ex['name'], out_dir)


if __name__ == '__main__':
    main()
