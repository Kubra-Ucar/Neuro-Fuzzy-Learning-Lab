# FNN-GSO — Glider Snake Optimizer ile Bulanık-Sinir Ağı Entegrasyonu

Bu depo, **Hibrid Zeki Sistemler** yüksek lisans dersi (2026–2027 Güz) kapsamında **Glider Snake Optimizer (GSO)** metasezgisel algoritmasının TSK-1 Bulanık-Sinir Ağı (FNN) yapısına entegrasyonunu (`FNN-GSO`) ve temel modellerle karşılaştırmasını içerir.

- **Başlangıç Deposu Atfı:** Bu çalışma, Prof. Dr. Uğur Yüzgeç tarafından ders kapsamında paylaşılan **Neuro-Fuzzy-Learning-Lab** başlangıç deposu temel alınarak geliştirilmiştir.
- **Öğrenci:** Kübra Uçar

## Eklenen ve Değiştirilen Bölümler

1. **`python/optimizers.py` (Değiştirildi):**
   - Deponun `(objective, lower, upper, initial, opts)` arayüzüne tam uyumlu **`optimizer_gso`** fonksiyonu eklendi.
   - Başlangıç popülasyonunda ilk birey olarak başlangıç FNN parametre vektörü (`initial`) korunmakta, tüm aday değerlendirmeleri ortak `max_evals = 320` bütçesinden sayılmaktadır.
   - GSO'nun süzülme/keşif (sinüzoidal dalgalanma + çift kılavuz) ve iniş/sömürü fazları ile sınır kontrolü (`np.clip`) uygulandı.
2. **`python/run_fnn_gso.py` (Eklendi):**
   - **Energy Efficiency (`energy_heating`)** ve **Seeds (`seeds`)** veri setleri üzerinde `FNN-GSO` modelini `Ridge`, `kNN`, `MLP`, `ELM`, `RVFL`, `ANFIS`, `FNN-GA` ve `FNN-PSO` ile aynı sabit veri bölünmesi ve değerlendirme protokolünde karşılaştırır.
   - Regresyon için `MAE`, `RMSE`, `R²`; sınıflandırma için `Accuracy`, `macro Precision`, `macro Recall`, `macro F1`, karışıklık matrisi ve eğitim sürelerini hesaplar.
   - `results_gso/` klasörüne amaç fonksiyonu değerlendirme sayısına (`FEs`) bağlı yakınsama grafiklerini (`.png`) ve özet sonuç tablosunu (`summary_results.csv`) kaydeder.

## Kurulum ve Çalıştırma (Python)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r python/requirements.txt
.venv\Scripts\python python/run_fnn_gso.py
