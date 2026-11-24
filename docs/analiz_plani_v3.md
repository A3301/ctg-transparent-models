# Analiz planı v3: B ve C planları (Çalışma 2 devamı)

Tarih: 2026-11-23. Bu plan, hiçbir v3 modeli eğitilmeden yazıldı. Dondurma kanıtı, bu dosyanın SHA-256 özetinin `experiments/log.csv` dosyasına yazılmasıdır (`phase = S3-plan`). Bundan sonraki her değişiklik "plan dışı" diye işaretlenir.

## 1. Gerekçe

Çalışma 2'nin (v2) önceden kayıtlı H1 ve H2 hipotezleri desteklenmedi:

- Uzman denetimli uçtan uca NeSy-v2, olayları kural tabanlı çıkarıcıdan kötü buldu.
- NeSy-v2'nin kavramları uzmana göre neredeyse şans düzeyindeydi.
- Risk tahmini bu kavramlara duyarsızdı.

Çalışma 1'den bilinenler:

| Model | AUROC (tekrar ortalamalı) | Not |
|---|---|---|
| Kural kavramları → mantık katmanı (`nesy_rulesonly_W60`) | 0,584 | Sadık ama bilgi kaybı büyük |
| FIGO sayısal öznitelikleri + LR (`figo_lr_W60`) | 0,729 | |
| 3 HRV özniteliği + SVM (`georgoulas_W60`) | 0,781 | |
| CNN-Bayes (`cnn_ens_W60`) | 0,794 | |

Literatürden: FHR sinyal kaybı asfiksiyle ilişkili (npj Digit Med 2025, 10.1038/s41746-025-01665-4). Çalışma 2'nin K3 denetimi de bunu CTU-UHB'de destekledi.

İki yeni yol, iki ayrı darboğazı hedefler:

- **Plan C, "doğruluğu yapı gereği garanti" model.** Kavramlar ve ölçümler, doğruluğu uzmana karşı ölçülmüş deterministik çıkarıcıdan gelir:
  - FIGO öznitelikleri;
  - HRV;
  - bilgi taşıyan sinyal kalitesi ölçümleri.

  Model toplamsal ve şeffaftır (GAM): her girdinin risk katkısı tek bir şekil fonksiyonuyla görülebilir. Soru: Sadık ve şeffaf bir model, derin modelin ayırt ediciliğine ulaşır mı?
- **Plan B, kavramları iyileştirmek.** Uçtan uca eğitimi bırakıp **ardışık** (sequential) kavram modeline geçilir. Bu, Koh 2020'deki sıralı CBM'nin karşılığıdır ve sızıntıyı yapı gereği sınırlar:
  1. Uzman etiketleriyle **yalnız olay tespiti** için eğitilen bir dedektör. Model seçimi olay F1'iyle yapılır, sonuçla değil. Girdiye taban çizgisine göre sapma kanalı eklenir: FIGO tanımları taban çizgisine görelidir.
  2. Dedektörün olaylarından aynı deterministik FIGO kodu ile öznitelikler hesaplanır.
  3. Plan C ile aynı şeffaf model.

  Soru: Uzman denetimli dedektör kural tabanlı çıkarıcıdan daha sadık mı? Bu, sonucu iyileştirir mi?

## 2. Değişmeyenler

- Veri: 536 kayıt, W60. Etiket pH < 7,05 (38 pozitif).
- Bölmeler: `results/splits/outer_splits.json` (5 × 10).
- Ayar:
  - iç 3 kat; 6 rastgele yapılandırma;
  - %15 doğrulama ayrımı;
  - önsel düzeltme + sıcaklık ölçekleme;
  - tekrar ortalamalı birincil analiz.
- Sonuç değişkenleri asla girdi olamaz (`check_no_forbidden`).
- Dış test kayıtlarının uzman etiketleri ayar ve eğitim sırasında yok (Çalışma 2'deki `blank_test_targets` kuralı).

## 3. Plan C: şeffaf toplamsal model

### 3.1 Girdiler (kayıt başına, W60)

| Grup | Değişkenler | Kaynak |
|---|---|---|
| FIGO (23) | `FIGO_FEATURES` (Çalışma 1'deki `figo_lr` ile aynı) | `results/features/figo_W60.csv` |
| HRV (3) | `vlf`, `lf`, `sd2` | `results/features/hrv_W60.csv` |
| Sinyal kalitesi (4) | `fhr_missing`, `uc_missing`, `fhr_longest_gap_s`, `uc_longest_gap_s` | `results/preprocess/window_quality.csv` |

### 3.2 Modeller

- **`transp_gam` (birincil):**
  1. Medyan ile doldurma ve standartlaştırma.
  2. Her değişkene ayrı kübik B-spline tabanı: `SplineTransformer`, düğümler nicel noktalarda, `extrapolation = "linear"`.
  3. L2 lojistik regresyon, `class_weight = "balanced"`.

  Model toplamsaldır: değişkenler arası etkileşim yoktur. Arama uzayı:
  - C ~ log-uniform [10⁻³, 10¹];
  - düğüm sayısı ∈ {3, 4, 5}.
- **`transp_lr` (ikincil):** aynı girdiler, doğrusal L2 LR. C ~ log-uniform [10⁻³, 10²].
- **Ablasyonlar (betimsel, Holm dışı), `transp_gam` ile aynı:**
  - `transp_gam_noq`: kalite değişkenleri olmadan;
  - `transp_gam_nohrv`: HRV olmadan.

## 4. Plan B: ardışık kavram modeli

### 4.1 Aşama 1: uzman denetimli olay dedektörü (`det`)

**Girdi:** 1 Hz'de 5 kanal:

| Kanal | İçerik |
|---|---|
| 1 | FHR |
| 2 | FHR maskesi |
| 3 | UC |
| 4 | UC maskesi |
| 5 | **FHR − taban çizgisi** |

- Taban çizgisi, kural çıkarıcının `baseline_curve` fonksiyonuyla 4 Hz'de hesaplanır ve 1 Hz'e indirgenir.
- Sapma kanalı 25 bpm'ye bölünür. Geçersiz yerlerde 0.
- Taban çizgisi yalnızca sinyalden hesaplanır; sonuç ya da uzman etiketi içermez.

**Mimari:** NeSy-v2'nin kodlayıcısı ve çerçeve başlığı. Mantık katmanı ve sonuç başlığı yoktur.

**Kayıp:** Yalnız çerçeve kaybı. Bileşenleri:
- ACC, DEC ve UC için maskeli BCE;
- DEC tipi için CE.

Maske v2 §3 ile aynı: pozitifler her zaman denetlenir, negatifler yalnız q = 1 olan bölmelerde. Sonuç kaybı ve pencere kavram kaybı yoktur.

**Ayar:** 6 rastgele yapılandırma, NeSy-v2 ile aynı mimari arama uzayı. Seçim ölçütü iç doğrulamada **olay F1 ortalaması**: (ACC F1 + DEC F1) / 2, q = 1 bölgesinde. Her iç katta eşik, ızgaradaki en iyi değerdir; bu iyimser bir seçimdir ama bütün yapılandırmalar için aynıdır.

**Son model:**
- Dış eğitim kümesinin %85'iyle eğitilen 5 üyelik topluluk.
- Olay eşikleri %15 doğrulamada olay F1'iyle seçilir (v2 ile aynı ızgara ve son işleme).
- Dış test için katların eşiklerinin medyanı, tekrar ortalamalı olasılıklara uygulanır (v2 ile aynı).

**Çapraz uydurma (aşama 2 için):** Dış eğitim kayıtlarının dedektör çıktıları örnek içi olursa yanlı olur. Bu yüzden dış eğitim kümesi 3 kata bölünür. Her katın olayları, öbür iki katla eğitilmiş tek üyeli dedektörden alınır:
- yapılandırma: seçilen yapılandırma;
- eşikler: o dedektörün iç doğrulama eşiği.

Dış test kayıtlarının olayları 5 üyeli topluluktan alınır.

### 4.2 Aşama 2: dedektör olaylarından FIGO öznitelikleri ve şeffaf model

- Kural çıkarıcının `extract()` kodu aynen kullanılır; yalnız ACC, DEC ve UC olay listeleri dedektörden gelir.
  - Olay sınırları 4 s bölmelerden 4 Hz örneklere çevrilir.
  - Derinlik, alan ve tepe noktası, kural taban çizgisine göre hesaplanır.
  - DEC tipi, FIGO zamanlama kurallarıyla (`classify_decels`) atanır. Uzman tip başlığı kullanılmaz: v2'de L ve P tiplerini hiç tahmin etmedi.
- Taban çizgisi, değişkenlik, STV/LTV ve sinüzoidal örüntü kural çıkarıcıdan kalır; bunlar olay listesine bağlı değil.
- Kalite ve HRV değişkenleri Plan C ile aynı.
- Model: **`transpdet_gam`**. Arama uzayı ve iç ayar `transp_gam` ile aynı.

## 5. Hipotezler

| Kod | Hipotez | Ölçüt | Aile |
|---|---|---|---|
| **C1** (birincil) | `transp_gam` AUROC'u CNN-Bayes'ten aşağı değil | ΔAUROC'un iki taraflı %97,5 bootstrap GA alt sınırı > −0,05 | Aşağı olmama ailesi (C1, B2; Bonferroni → %97,5 GA) |
| **C2** (birincil) | `transp_gam`, `figo_lr`'den daha ayırt edici | ΔAUROC > 0; eşleştirilmiş bootstrap p (2000) | Üstünlük ailesi (Holm: C2, B1a, B1b) |
| **B1a** (birincil) | Dedektörün DEC olay F1'i kural çıkarıcıdan yüksek | v2 H1a ile aynı (q = 1 bölgesi, herhangi bir örtüşme, kayıt düzeyi bootstrap) | Üstünlük ailesi |
| **B1b** (birincil) | Dedektörün ACC olay F1'i kural çıkarıcıdan yüksek | Aynı | Üstünlük ailesi |
| **B2** | `transpdet_gam` AUROC'u CNN-Bayes'ten aşağı değil | C1 ile aynı | Aşağı olmama ailesi |
| B3 (betimsel) | `transpdet_gam` ile `transp_gam` arasındaki fark | ΔAUROC, %95 GA | — |

"Desteklendi" koşulları:
- Üstünlük hipotezleri için fark > 0 ve Holm düzeltmeli p < 0,05.
- Aşağı olmama hipotezleri için GA alt sınırı > −0,05.

## 6. İkincil analizler (önceden tanımlı, betimsel)

1. **Ayırt edicilik ve kalibrasyon:** AUROC, AUPRC, Brier, ECE ve Harrell C (sürekli pH). Bütün v3 modelleri için; karşılaştırma CNN-Bayes, `figo_lr` ve `georgoulas` ile.
2. **Şeffaflık:** `transp_gam` için her değişkenin şekil fonksiyonu.
   - Hesap: tekrar 0'ın 10 dış katında, değişkenin 5.–95. yüzdelik aralığında, öbür değişkenler medyandayken log-odds katkısı.
   - Raporlama: katlar arası yayılımla birlikte.
   - En büyük katkı aralığına göre ilk 8 değişken gösterilir.
3. **Kavram sadakati:**
   - Dedektörün UC F1'i;
   - IoU ≥ 0,5 ölçütüyle F1;
   - FHRMA dış doğrulaması: tekrar 0 topluluğu, v2 §6.5 ile aynı yöntem.
4. **Tamamlayıcılık:** `transp_gam` ve CNN-Bayes olasılıklarının basit ortalamasının AUROC'u. Ayar yoktur.
5. **Bilgi taşıyan eksiklik:** `transp_gam` ile `transp_gam_noq` arasındaki ΔAUROC.

## 7. Bu plan yazılırken bilinenler (ayar geçmişi)

- Çalışma 1'in bütün sonuçları (§1'deki tablo dahil).
- Çalışma 2 (v2) sonuçlarının tamamı (`docs/calisma2_sonuc_rapor.md`).
- K3: asidemik kayıtlar kayıt düzeyi kalite ölçütünü daha sık sağlamıyor.
- v3 modellerinin hiçbiri çalıştırılmadı. HRV ile kalite değişkenlerinin birlikte modellenmesi de denenmedi.

## 8. Yapılmayacaklar

- Başka model ailesi ve etkileşimli model (ör. XGBoost) birincil analize eklenmez.
- Dedektör için ek mimari arayışı yapılmaz: yalnız §4.1'deki 6 yapılandırma.
- Plan sonrası her ek analiz "plan dışı" diye işaretlenir.
