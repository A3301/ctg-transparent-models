# Analiz planı (DONDURULDU)

- **Sürüm:** 1.0
- **Dondurulma tarihi:** 2026-11-22
- **Durum:** Bu dosya git'e işlendiği andan itibaren dondurulmuştur. Sonraki her değişiklik ilgili yerde **[PLAN DIŞI, tarih, gerekçe]** etiketiyle işaretlenecek ve `docs/plan_disi_degisiklikler.md` dosyasına da yazılacaktır.
- **Dayanaklar:** `PROMPT.md`, `docs/faz_0_rapor.md`, `docs/faz_1_rapor.md`, `docs/yenilik_teyidi.md`, `docs/sota_tablosu.md`.
- **Ayar geçmişi:** Plan yazılırken hiçbir modelin sonucu görülmedi. Hiçbir model henüz eğitilmedi.

---

## 1. Amaç ve hipotezler

**Soru:** FIGO 2015 yorum kurallarını olasılıksal mantık katmanı olarak içeren ve Bayesçi belirsizlik veren bir model, CTU-UHB üzerinde fetal asidemiyi (pH < 7,05) aynı omurgalı saf derin modellerden ve literatür yöntemlerinden daha doğru, daha iyi kalibre ve daha yorumlanabilir biçimde tahmin edebilir mi?

| Kod | Hipotez | Test |
|---|---|---|
| **H1a** (ayırt edicilik, eşdeğerlik ya da üstünlük) | Önerilen model (NeSy-Bayes), aynı omurga ve aynı Bayesçi katmana sahip saf derin modele (CNN-Bayes) göre AUROC'ta aşağı değildir. | Eşleştirilmiş ΔAUROC = NeSy-Bayes − CNN-Bayes. **Aşağı olmama marjı −0,05.** 95% bootstrap güven aralığının alt sınırı > −0,05 ise H1a desteklenir. Alt sınır > 0 ise ve DeLong testi p < 0,05 ise üstünlük de iddia edilir. AUPRC farkı ikincil olarak raporlanır. |
| **H1b** (kalibrasyon) | NeSy-Bayes'in ECE'si ve Brier skoru CNN-Bayes'ten düşüktür. | Eşleştirilmiş bootstrap; farkın 95% güven aralığı tamamen < 0 ise desteklenir. Brier birincil, ECE ikincil ölçüttür (40 pozitifle ECE gürültülüdür). |
| **H2** (seçici tahmin) | Epistemik belirsizliğe göre en belirsiz %20 vaka klinisyene yönlendirildiğinde (kapsama %80), kalan vakalarda duyarlılık ve özgüllük, tüm vakalardaki değerlere göre artar ve rastgele yönlendirmeden daha iyidir. | Sabit eşikte (§5.3) Se_kalan − Se_tüm ve Sp_kalan − Sp_tüm farkları; eşleştirilmiş bootstrap güven aralıkları. Rastgele yönlendirme referansı 1000 tekrarla hesaplanır. Karşılaştırma olarak en büyük olasılığa göre (güvene göre) yönlendirme de raporlanır. |
| **H3** (keşifsel, isteğe bağlı) | Uzun bağlamlı SSM omurgası 90 dk'yı bölmeden işleyerek 30/60 dk CNN'den daha iyi AUROC verir. | Yalnızca betimsel; çoklu karşılaştırma ailesine girmez. Hesap bütçesi yeterse yapılır; yapılmazsa bu açıkça raporlanır. |

H1a'daki marj (−0,05) klinik olarak "aynı sınıfta" kabul edilebilir bir kayıptır. 38 pozitifle gücün düşük olduğu bilinmektedir. Sonuç ne çıkarsa çıksın, güven aralıklarıyla birlikte raporlanacaktır.

## 2. Veri, etiketler, popülasyon

- **Veri:** CTU-UHB v1.0.0, 552 kayıt. SHA-256 doğrulandı (Faz 0). Her kayıt ayrı bir hastadır.
- **Birincil etiket:** göbek arteri pH < 7,05. Tüm kayıtlarda tanımlı.
- **İkincil etiketler (duyarlılık):**
  - pH ≤ 7,15
  - BDecf ≥ 12 (11 eksik kayıt bu analizde dışarıda bırakılır)
  - Apgar5 < 7
- **Dışlama ölçütü:** Temizlik sonrası W60 penceresinde FHR eksik oranı > %50 olan kayıtlar dışlanır. Bu ölçüt etiketlere bakılmadan seçildi ve veri tabanının kendi dahil etme kuralına dayanır (her 30 dk'lık pencerede FHR kalitesi > %50). Sonuç:
  - 16 kayıt dışlandı (birincil etikete göre 2'si pozitif).
  - **Analiz popülasyonu 536 kayıt; birincil etikette 38 pozitif.**
  - Tüm pencerelerde aynı popülasyon kullanılır.
  - Duyarlılık analizinde dışlama yapılmaz (552 kayıt).
- **Girdiler:** yalnızca FHR ve UC sinyalleri ile bunların eksiklik maskeleri.
- **Asla girdi olamayanlar:**
  - pH, BDecf, pCO2, BE, Apgar, yenidoğan sonuçları
  - doğum şekli, I. ve II. evre süreleri
  - `Pos. II.st.` ve ondan türeyen II. evre göstergesi, kayıt süresi
  - doğum ağırlığı, cinsiyet, mekonyum

  Gerekçe: Faz 0'da kayıt süresi ile II. evre süresi arasında r = 0,98 bulundu. II. evre yokluğu 46 kaydın 42'sinde sezaryen anlamına geliyor.
- **II. evre göstergesinin tek kullanım alanı:** CV tabakalaması ve alt grup analizi.
- **Klinik ön bilgiler** (gebelik haftası, anne yaşı, gravida, parite, diyabet, hipertansiyon, preeklampsi, erken su gelmesi, ateş, prezentasyon, indüksiyon) yalnızca ayrı ve **"klinik-ek" diye işaretli keşifsel bir deneyde** kullanılabilir.

## 3. Ön işleme (`src/preprocess/clean.py`, `build_dataset.py`)

### FHR
1. 0 değerleri ve 50–210 bpm dışındaki değerler eksik sayılır.
2. Ardışık örnekler arasındaki > 25 bpm sıçramalarda sinyal parçalara bölünür. 5 sn'den kısa parçalar sıçrama (artefakt) sayılır. Bu eşikle tüm ardışık farkların yalnızca %0,03'ü bölünüyor.
3. ≤ 15 sn boşluklar doğrusal olarak doldurulur. Daha uzun boşluklar eksik kalır ve maskeyle işaretlenir.
4. Denenip reddedilen yöntem: "seviye kuralı" (±60 sn çevreden > 25 bpm sapan parçaları atmak) gerçek derin deselerasyonları sildi (`figures/preprocess/check_level_rule.png`). Kapalı tutulur.

### UC
1. NaN değerler, negatif değerler ve ≥ 10 sn süren tam sıfır dizileri eksik sayılır.
2. 60 sn'lik pencerede değişim aralığı ≤ 1 birim olan ve ≥ 60 sn süren düz çizgi bölümleri eksik sayılır (çalışmayan sensör; `figures/preprocess/check_uc_flat.png`).
3. ≤ 15 sn boşluklar doldurulur.

### Model girdisi
- **Örnekleme:** 1 Hz. Her saniye, geçerli 4 Hz örneklerin ortalamasıdır. Saniyenin en az yarısı geçerliyse maske 1, değilse 0'dır.
- **Ölçekleme:** veriden türetilmeyen sabit sabitler kullanılır, böylece katlar arası sızıntı olmaz:
  - FHR: (FHR − 140) / 30
  - UC: (UC − 20) / 30
- **Eksik değerler:** 0'a eşitlenir, ayrıca bir maske kanalı verilir.
- **Kanallar:** [FHR, FHR maskesi, UC, UC maskesi].
- **Yeniden uygulanan yöntemler** makaledeki örnekleme hızını ve sinyal kümesini kullanır (§6.2).

### Pencereler
Tüm kayıtlarda `Sig2Birth = 0`, yani sinyalin sonu doğum anıdır.

| Pencere | Tanım | Rolü |
|---|---|---|
| **W60** | Doğumdan önceki son 60 dk | Birincil |
| W30 | Son 30 dk | Duyarlılık |
| W90 | Son 90 dk. Daha kısa kayıtlarda baştan maskeli dolgu yapılır; yalnızca 39 kayıt tam 90 dk'dır | Duyarlılık ve H3 |
| **S1_60** | I. evrenin son 60 dk'sı (II. evre başlangıcında biter; II. evre yoksa sinyalin sonunda) | **Önceden belirlenmiş ana ikincil analiz** |

S1_60 ana metinde W60 ile yan yana raporlanır. Gerekçe: W60'ta II. evre (ıkınma) örüntüsünün varlığı, doğum şeklini dolaylı yoldan taşıyabilir.

## 4. Veri bölme ve ayar protokolü (bütün modellerde aynı)

### Dış döngü
- 5 tekrar × tabakalı 10 kat, kayıt (hasta) düzeyinde.
- Tabakalama değişkeni: birincil etiket × II. evre varlığı.
- Tohumlar: `20261122 + tekrar_no`.
- Katlar `results/splits/outer_splits.json` dosyasına yazılır ve bütün modeller aynı katları kullanır.

### İç döngü (ayar)
- Dış-eğitim kümesi içinde tabakalı 3 kat.
- Rastgele arama bütçesi:
  - Derin modeller: 6 yapılandırma.
  - Klasik modeller: 30 yapılandırma.
- Seçim ölçütü: iç katların ortalama AUROC'u.
- Ablasyonlar, önerilen modelin ilgili dış katta seçilen yapılandırmasını kullanır. Yeniden ayar yapılmaz; bu, ablasyonu adil kılar.

### Son eğitim
- Seçilen yapılandırma dış-eğitim kümesinin tamamında eğitilir.
- Derin modellerde dış-eğitim kümesinin tabakalı %15'i iç doğrulama olarak ayrılır. Bu küme şunlar için kullanılır:
  - erken durdurma (en fazla 60 epok, sabır 10)
  - sıcaklık ölçekleme **[PLAN DIŞI #2: öncesinde önsel düzeltme]**
  - karar eşiklerinin seçimi

### Sınıf dengesizliği
- Bütün modellerde sınıf ağırlıklı kayıp kullanılır: pozitif ağırlık = n_negatif / n_pozitif (dış-eğitim kümesinden hesaplanır).
- Yeniden örnekleme, SMOTE ya da GAN kullanılmaz.

### Test katları
Ayar ve eşik seçiminde test katlarına hiçbir zaman bakılmaz.

### Tahmin birleştirme
- Her tekrarda 10 test katının tahminleri birleştirilir (havuzlanmış OOF). Böylece her kayıt için tekrar başına bir tahmin elde edilir.
- **Birincil analiz**, her kaydın 5 tekrardaki OOF olasılıklarının ortalamasıyla yapılır (536 tahmin).
- Tekrarlar arası değişkenlik ayrıca raporlanır: tekrar başına AUROC, ortalama ± SD.

### Sızıntı kontrol listesi (her deneyde koşulur ve loglanır)
- [ ] Dış-eğitim ve dış-test kayıt kimlikleri ayrık.
- [ ] Normalizasyon sabit sabitlerle yapılıyor ya da yalnızca dış-eğitim kümesinden hesaplanıyor.
- [ ] Öznitelik seçimi, ayar, eşik ve sıcaklık yalnızca dış-eğitim içinde.
- [ ] Girdi listesinde yasaklı değişken yok (otomatik `assert`).

## 5. Metrikler ve istatistik

### 5.1 Metrikler (birincil analizde 536 kaydın tekrar ortalamalı tahminleri üzerinden)
- **Ayırt edicilik:**
  - **AUROC (birincil uç nokta)**
  - AUPRC
  - %85 ve %90 özgüllükte duyarlılık
  - %5, %10, %15 ve %20 yanlış pozitif oranında TPR (Petrozziello ve Mendis ile karşılaştırma için)
- **Eşikli metrikler:**
  - Se, Sp, G-mean ve QI = √(Se·PPV) **[PLAN DIŞI, 2026-11-22: QI = √(Se·Sp) olarak düzeltildi; bkz. `docs/plan_disi_degisiklikler.md` #1]**
  - Eşik: iç doğrulamada G-mean'i en büyükleyen değer, dış kat başına
- **Kalibrasyon:**
  - Brier skoru
  - ECE (10 eşit frekanslı aralık)
  - kalibrasyon eğimi ve kesişimi
  - güvenilirlik diyagramı
- **Seçici tahmin:**
  - kapsama–risk eğrisi ve AURC
  - %80 ve %90 kapsamada Se, Sp ve AUROC
- **Klinik fayda:** 0,02–0,30 eşik aralığında karar eğrisi (net fayda).
- **Güven aralıkları:** Bütün metrikler kayıt düzeyinde 2000 bootstrap örneklemesiyle %95 yüzdelik güven aralığıyla raporlanır.

### 5.2 Testler
- **AUROC farkları:** eşleştirilmiş DeLong testi, tekrar ortalamalı tahminler üzerinde.
- **Çoklu karşılaştırma ailesi:** NeSy-Bayes'in şu modellerle karşılaştırması, toplam 9:
  - CNN-Bayes (H1)
  - FIGO-LR, FIGO-XGB, Georgoulas-SVM
  - CNN, GRU, Transformer
  - MCNN, FHR-LINet
- **Düzeltme:** Holm.
- **Diğer metrikler:** eşleştirilmiş bootstrap (2000), fark ve %95 güven aralığı.
- **Sınırlılık notu:** CV tahminleri tam bağımsız değildir. DeLong p değerleri bu nedenle yaklaşık kabul edilir; yorum güven aralıklarına dayanır.

### 5.3 Seçici tahmin (H2) ayrıntısı
- **Belirsizlik ölçüsü:** epistemik belirsizlik, yani topluluk üyeleri arasındaki karşılıklı bilgi (mutual information). İkincil ölçüler: toplam öngörücü entropi ve aleatorik bileşen.
- **Karar eşiği:** iç doğrulamada %90 özgüllük sağlayan eşik, dış kat başına önceden belirlenir.
- **Kapsama:** %80, belirsizlik sıralamasıyla. Kesim değeri iç doğrulama belirsizlik dağılımının %80 yüzdeliğidir. Böylece test kümesindeki kapsama yaklaşık %80 olur ve gerçekleşen kapsama raporlanır.

## 6. Modeller

### 6.1 Temel modeller (Faz 3)
| Model | Tanım |
|---|---|
| **FIGO-LR**, **FIGO-XGB** | Kural tabanlı öznitelik çıkarıcının ürettiği FIGO öznitelikleri: taban çizgisi, STV/LTV, akselerasyon sayısı, deselerasyon sayısı, türleri ve süreleri, sinüzoidal örüntü, kasılma sıklığı. Eşikler `src/features/figo.py` içinde belgelenir. |
| **CNN** | 1B-CNN/TCN, artık (residual) bloklar. Önerilen modelin omurgası. |
| **GRU** | Çift yönlü GRU. |
| **Transformer** | Yamalı (patch) Transformer. |
| **CNN-Bayes** | CNN omurgasının 5 üyeli derin topluluğu + sıcaklık ölçekleme. H1 karşılaştırma modeli. |

### 6.2 Yeniden uygulanan yöntemler (Faz 1 seçimi)
| Model | Tanım | Tutarlılık kontrolü |
|---|---|---|
| **MCNN** (Petrozziello 2019) | FHR + UC + kalite, son 60 dk, 0,25 Hz | — |
| **FHR-LINet** (Mendis 2024) | Yalnızca FHR, 0,25 Hz | Makaledeki koşullarda (5×5 CV, pH < 7,05, 552 kayıt) TPR@FPR değerleri karşılaştırılır |
| **Georgoulas-SVM** (2017) | HRV öznitelikleri (VLF, LF, SD2) + RBF çekirdekli SVM | pH ≤ 7,05, I. evrenin son 30 dk'sı, G-mean karşılaştırılır. Not: LS-SVM yerine scikit-learn SVC kullanılır; bu fark raporlanır. |

### 6.3 Önerilen model (Faz 4)
- **Omurga:** CNN (§6.1), önceden sabitlenir. PROMPT omurganın "Faz 3'te en iyi çıkan" model olmasını öneriyor. Ancak Faz 3 sonuçları dış test katlarından geldiği için onlara göre seçim yapmak test verisine bakarak model seçmek olur. **Bu, PROMPT'tan bilinçli bir sapmadır.** Faz 3'te başka bir kodlayıcı daha iyi çıkarsa bu durum raporlanır, ama omurga değiştirilmez.
- **Kavram başlıkları:**
  - taban çizgisi: bradikardi / normal / taşikardi **[PLAN DIŞI #3: 4 sınıf; #4: ek kavramlar]**
  - değişkenlik: azalmış / normal / artmış / sinüzoidal
  - akselerasyon: var / yok
  - tekrarlayan geç deselerasyon
  - tekrarlayan değişken deselerasyon
  - uzamış deselerasyon
  - erken deselerasyon
  - uzun süreli (≥ 30 dk) azalmış değişkenlik
- **Zayıf denetim:** kural tabanlı çıkarıcının ürettiği zayıf etiketlerle küçük ağırlıklı (λ ayarlanır) yardımcı kayıp.
- **Mantık katmanı:**
  - FIGO 2015 normal/şüpheli/patolojik sınıflaması ve bundan asidemiye geçiş, öğrenilebilir kural ağırlıklarıyla bir ProbLog programı olarak yazılır. **[PLAN DIŞI #5: ek kavram düzeyi kurallar]**
  - Çıkarım tam numaralandırmayla yapılır (ağırlıklı model sayımı).
  - Uygulama sırası PROMPT'taki gibidir: DeepProbLog → Scallop → kendi PyTorch katmanı. Hangisinin kullanıldığı ve nedeni raporlanır.
  - Kendi katmanı kullanılırsa, ProbLog ile aynı olasılıkları verdiği sayısal olarak doğrulanır.
- **Bayesçi katman:** 5 üyeli derin topluluk ve sıcaklık ölçekleme. Epistemik ve aleatorik belirsizlik ayrıştırılır.
- **Açıklama:** vaka başına kavram olasılıkları ve kural katkılarından üretilen tek paragraflık özet.

### 6.4 Ablasyonlar
1. Mantık katmanı yok: kavramlar → doğrusal sınıflandırıcı.
2. Kavram denetimi yok (λ = 0).
3. Bayesçi katman yok: tek model + sıcaklık ölçekleme.
4. Yalnızca kurallar: sinir ağı yok; zayıf etiket kavramları aynı mantık programına girer, yalnızca kural ağırlıkları öğrenilir.
5. Yalnızca FIGO öznitelikleri (= FIGO-LR ve FIGO-XGB).

### 6.5 Duyarlılık analizleri

Hesap bütçesi nedeniyle bu analizler yalnızca NeSy-Bayes, CNN-Bayes ve FIGO-LR için yapılır:
- Etiketler: pH ≤ 7,15, BDecf ≥ 12, Apgar5 < 7.
- Pencereler: W30, W90, S1_60.
- Dışlama olmadan: 552 kayıt.
- Sinyal kalitesi katmanları: W60 FHR eksikliği üçte birlik dilimlerine göre (yalnızca birincil modelin tahminleri üzerinde, alt grup metrikleri).
- II. evre var ya da yok alt grupları (betimsel).
- Klinik-ek deneyi: NeSy-Bayes + klinik ön bilgiler. Keşifsel ve ayrı işaretli.

## 7. Raporlama
- Bütün deneyler `experiments/log.csv` dosyasına yazılır: tarih, yapılandırma, tohum, metrikler.
- Olumsuz sonuçlar ve desteklenmeyen hipotezler açıkça yazılır.
- Literatürle karşılaştırma yalnızca protokolü karşılaştırılabilir olan çalışmalarla (A sınıfı) yapılır. Diğer çalışmalar için protokol farkı belirtilir.
- Raporlama TRIPOD+AI kontrol listesine göre yapılır.

## 8. Bilinen sınırlılıklar (önceden)
- 38 pozitif → geniş güven aralıkları ve düşük güç.
- Dış doğrulama yok.
- Kavramlar için uzman etiketi yok; kavram doğruluğu yalnızca kural tabanlı zayıf etiketlere göre ölçülebilir.
- W60 penceresi II. evre örüntüsü taşır; S1_60 analizi bu nedenle eklendi.
