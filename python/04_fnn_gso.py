"""
FNN-GSO Tam Karşılaştırma Betiği
================================
Bu betik, Hibrid Zeki Sistemler dersi kapsamında Glider Snake Optimizer (GSO)
algoritmasının Bulanık-Sinir Ağı (TSK-1 FNN) yapısına entegrasyonunu test eder
ve ders yönergesinde istenen temel modellerle karşılaştırır.

Kullanılan Veri Setleri (İlk Çalışan Entegrasyon):
- Regresyon     : Energy Efficiency (energy_heating) -> MAE, RMSE, R^2
- Sınıflandırma : Seeds (seeds) -> Accuracy, macro Precision, macro Recall,
                  macro F1 ve Karışıklık Matrisi

Karşılaştırılan Modeller:
1. Temel Modeller          : Ridge, kNN
2. Tek Başına Sinir Ağları : MLP, ELM, RVFL
3. Gradyan Tabanlı FNN     : ANFIS (sezgisel olmadan eğitilen aynı mimari)
4. Sezgisel FNN Modelleri  : FNN-GSO (önerilen), FNN-GA, FNN-PSO
"""

import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Ders deposundaki çekirdek fonksiyonlar ve optimizasyon algoritmaları
from course_core import (
    Config, examples, prepare_data, init_fnn,
    train_mlp, train_random_nn, train_fnn, train_meta_fnn,
    ridge_solve, predict, classification_metrics, rmse
)
from optimizers import optimizer_gso, optimizer_gsodbo, optimizer_ga, optimizer_pso


def _extract_model(res):
    """
    Eğitim fonksiyonları bazen (model, geçmiş) şeklinde demet (tuple),
    bazen de doğrudan model sözlüğü döndürebilir. Bu yardımcı fonksiyon
    her iki durumda da yalnızca eğitilmiş modeli güvenle ayıklar.
    """
    return res[0] if isinstance(res, tuple) else res


def regression_metrics_full(y_true, y_pred):
    """
    Regresyon problemleri için ders yönergesinde istenen tüm ölçütleri
    orijinal hedef ölçeğinde hesaplar: MAE, RMSE ve R^2.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred

    mae = float(np.mean(np.abs(err)))                  # Ortalama Mutlak Hata
    rmse_val = float(rmse(y_pred, y_true))             # Kök Ortalama Kare Hata
    ss_res = float(np.sum(err ** 2))                   # Artık kareler toplamı
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))  # Toplam varyans
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")  # Belirleme katsayısı

    return dict(mae=mae, rmse=rmse_val, r2=r2)


def classification_metrics_full(y_true, pred_scores, classes):
    """
    Sınıflandırma problemleri için depodaki hazır ölçütlere (Accuracy, macro F1)
    ek olarak yönergede zorunlu tutulan macro Precision, macro Recall ve
    doğrulanmış karışıklık matrisini (Confusion Matrix) hesaplar.
    """
    # 1) Depodaki mevcut sınıflandırma metriklerini al
    base = classification_metrics(y_true, pred_scores, classes)

    # 2) En yüksek skorlu sınıfı (argmax) tahmin edilen etiket olarak seç
    y_true_arr = np.asarray(y_true).ravel()
    classes_arr = np.asarray(classes)
    pred_idx = np.argmax(pred_scores, axis=1)
    y_pred_arr = classes_arr[pred_idx]

    # 3) Karışıklık matrisini oluştur (satır: gerçek sınıf, sütun: tahmin edilen sınıf)
    k = len(classes_arr)
    cm = np.zeros((k, k), dtype=int)
    for idx_t, ct in enumerate(classes_arr):
        for idx_p, cp in enumerate(classes_arr):
            cm[idx_t, idx_p] = int(np.sum((y_true_arr == ct) & (y_pred_arr == cp)))

    # 4) Her sınıf için TP, FP, FN değerlerinden Precision ve Recall hesapla
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    # Sıfıra bölme hatasını önleyerek sınıf bazlı kesinlik ve duyarlılık bul
    precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_c = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

    # 5) Sınıf ortalamalarını (macro) sözlüğe ekle
    base['macro_precision'] = float(np.mean(precision_c))
    base['macro_recall'] = float(np.mean(recall_c))
    base['confusion_matrix_verified'] = cm
    return base


def fit_predict_ridge(Xtr, ytr, Xval, yval, Xte, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """
    Temel model 1: Ridge (L2 düzenlileştirmeli doğrusal model).
    - Sabit terim (bias) sütunu eklenir.
    - En iyi düzenlileştirme katsayısı (lambda) yalnızca doğrulama verisiyle seçilir.
    - Test verisi yalnızca son tahmin aşamasında kullanılır.
    """
    Atr = np.c_[np.ones((len(Xtr), 1)), Xtr]
    Aval = np.c_[np.ones((len(Xval), 1)), Xval]
    Ate = np.c_[np.ones((len(Xte), 1)), Xte]

    best_err, best_w = float('inf'), None
    for lam in lams:
        w = ridge_solve(Atr, ytr, lam)
        err = rmse(Aval @ w, yval)
        if err < best_err:
            best_err, best_w = err, w
    return Ate @ best_w


def fit_predict_knn(Xtr, ytr, Xval, yval, Xte, ks=(3, 5, 7, 9)):
    """
    Temel model 2: k-En Yakın Komşu (kNN).
    - Öklid uzaklığı ile en yakın k eğitim örneğinin ortalamasını alır.
    - En uygun k komşu sayısı yalnızca doğrulama verisi (Xval, yval) ile seçilir.
    """
    def _pred(X_eval, k):
        dists = np.sqrt(np.maximum(0.0, np.sum((X_eval[:, None, :] - Xtr[None, :, :]) ** 2, axis=2)))
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        return np.mean(ytr[idx], axis=1)

    best_k, best_err = ks[0], float('inf')
    for k in ks:
        if k >= len(Xtr):
            continue
        err = rmse(_pred(Xval, k), yval)
        if err < best_err:
            best_err, best_k = err, k
    return _pred(Xte, best_k)


def plot_convergence(histories, dataset_name, out_dir):
    """
    Sezgisel algoritmaların (FNN-GSO, FNN-GA, FNN-PSO) yakınsama eğrilerini çizer.
    X ekseni: Başlangıç popülasyonu dahil amaç fonksiyonu değerlendirme sayısı (FEs).
    Y ekseni: O ana kadar bulunan en iyi doğrulama RMSE değeri.
    """
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
    # Sonuç tabloları ve yakınsama grafikleri için çıktı klasörü oluştur
    out_dir = 'results_gso'
    os.makedirs(out_dir, exist_ok=True)

    # Varsayılan ders yapılandırmasını yükle (kural sayısı, bütçe=320 FEs, tohumlar)
    cfg = Config()

    # İlk çalışan entegrasyon için seçilen 1 regresyon + 1 sınıflandırma veri seti
    target_names = {'energy_heating', 'seeds'}
    selected_datasets = [ex for ex in examples() if ex['name'] in target_names]

    # Tekrarlanabilirlik için sabit başlangıç tohumu (seed)
    seed = cfg.seeds[0]
    summary_rows = []

    for ex in selected_datasets:
        print(f"\n================ Veri Seti: {ex['name']} ({ex['task']}) | Seed: {seed} ================")

        # Sabit veri bölünmesini yükle; ölçekleme yalnızca eğitim verisinden öğrenilir
        data = prepare_data(**{k: v for k, v in ex.items() if k != 'name'})
        Xtr, ytr = data['Xtrain'], data['ytrain']
        Xval, yval = data['Xval'], data['yval']
        Xte, yte = data['Xtest'], data['ytest']
        histories = {}

        def _record(model_name, raw_pred, elapsed, evals=''):
            """Model tahminlerini orijinal ölçeğe çevirir, metrikleri yazar ve CSV listesine ekler."""
            pred = raw_pred * data['ystd'] + data['ymean']
            row = dict(
                dataset=ex['name'], task=ex['task'], seed=seed, model=model_name,
                mae='', rmse='', r2='', accuracy='', macro_precision='',
                macro_recall='', macro_f1='', evals=evals, time_sec=round(elapsed, 4)
            )
            if ex['task'] == 'classification':
                m = classification_metrics_full(yte, pred, data['classes'])
                row.update(
                    accuracy=round(m.get('accuracy', np.nan), 4),
                    macro_precision=round(m['macro_precision'], 4),
                    macro_recall=round(m['macro_recall'], 4),
                    macro_f1=round(m.get('macro_f1', np.nan), 4),
                )
                print(
                    f"  {model_name:9s} | Acc={row['accuracy']:.4f} "
                    f"| MacroP={row['macro_precision']:.4f} "
                    f"| MacroR={row['macro_recall']:.4f} "
                    f"| MacroF1={row['macro_f1']:.4f} "
                    f"| Süre={elapsed:.2f}s"
                )
                if model_name in ('FNN-GSO', 'FNN-GSODBO'):
                    print(f"    [FNN-GSO Karışıklık Matrisi]:\n{m['confusion_matrix_verified']}")
            else:
                m = regression_metrics_full(yte, pred)
                row.update(
                    mae=round(m['mae'], 4),
                    rmse=round(m['rmse'], 4),
                    r2=round(m['r2'], 4),
                )
                print(
                    f"  {model_name:9s} | MAE={row['mae']:.4f} "
                    f"| RMSE={row['rmse']:.4f} "
                    f"| R2={row['r2']:.4f} "
                    f"| Süre={elapsed:.2f}s"
                )
            summary_rows.append(row)

        # --- 1) Temel Modeller (Ridge ve kNN) ---
        t0 = time.perf_counter()
        _record('Ridge', fit_predict_ridge(Xtr, ytr, Xval, yval, Xte), time.perf_counter() - t0)

        t0 = time.perf_counter()
        _record('kNN', fit_predict_knn(Xtr, ytr, Xval, yval, Xte), time.perf_counter() - t0)

        # --- 2) Tek Başına Sinir Ağları (MLP, ELM, RVFL) ---
        t0 = time.perf_counter()
        m_mlp = _extract_model(train_mlp(Xtr, ytr, Xval, yval, cfg, seed))
        _record('MLP', predict(m_mlp, Xte), time.perf_counter() - t0)

        t0 = time.perf_counter()
        m_elm = _extract_model(train_random_nn(Xtr, ytr, cfg, seed, direct=False))
        _record('ELM', predict(m_elm, Xte), time.perf_counter() - t0)

        t0 = time.perf_counter()
        m_rvfl = _extract_model(train_random_nn(Xtr, ytr, cfg, seed, direct=True))
        _record('RVFL', predict(m_rvfl, Xte), time.perf_counter() - t0)

        # --- 3) Sezgisel Olmadan Eğitilen Aynı FNN Mimarisi (ANFIS - Gradyan Tabanlı) ---
        init_base = init_fnn(Xtr, cfg.rules, 1, seed)
        t0 = time.perf_counter()
        m_anfis = _extract_model(train_fnn(init_base, Xtr, ytr, Xval, yval, cfg))
        _record('ANFIS', predict(m_anfis, Xte), time.perf_counter() - t0)

        # --- 4) Sezgisel Optimizasyonlu FNN Modelleri (FNN-GSO, FNN-GA, FNN-PSO) ---
        # Tüm sezgisel yöntemler aynı başlangıç FNN yapısından ve aynı 320 FEs bütçesinden başlar
        for name, opt_fn in [
    ('FNN-GSO', optimizer_gso),
    ('FNN-GSODBO', optimizer_gsodbo),
    ('FNN-GA', optimizer_ga),
    ('FNN-PSO', optimizer_pso),
]:
            initial = init_fnn(Xtr, cfg.rules, 1, seed)
            t0 = time.perf_counter()
            model, search = train_meta_fnn(initial, Xtr, ytr, Xval, yval, cfg, seed, opt_fn)
            elapsed = time.perf_counter() - t0
            histories[name] = search['history']
            _record(name, predict(model, Xte), elapsed, evals=search['evals'])

        # Üç sezgisel yöntemin yakınsama grafiğini kaydet
        plot_convergence(histories, ex['name'], out_dir)

    # Tüm modellerin sayısal sonuçlarını raporlamada kullanmak üzere CSV'ye yaz
    csv_path = os.path.join(out_dir, 'summary_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\nTüm özet sonuçlar kaydedildi: {csv_path}')


if __name__ == '__main__':
    main()
