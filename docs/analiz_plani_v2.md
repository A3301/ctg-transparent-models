# Analiz planı v2: Çalışma 2 (DONDURULDU)

- **Sürüm:** 2.0
- **Dondurulma tarihi:** 2026-11-23, git etiketi `analiz-plani-v2.0`.
- **Durum:** Git'e işlendiği andan itibaren dondurulmuştur. Sonraki her değişiklik ilgili yerde **[PLAN DIŞI v2, tarih, gerekçe]** etiketiyle işaretlenir ve `docs/plan_disi_degisiklikler.md` dosyasına "v2-#" numarasıyla yazılır.
- **v1 ile ilişki:** `docs/analiz_plani.md` (v1.0) değişmeden yürürlükte kalır. v2, v1'in veri, ön işleme, bölme ve ayar protokolünü (§2–§4) aynen devralır; burada yalnızca farklar ve yeni öğeler yazılıdır.
- **Dayanaklar:**
  - `docs/calisma2_ozgun_deger.md`: motivasyon, yenilik, katkılar
  - `docs/aciklama_verileri.md`: uzman etiketleri
  - `docs/kavram_dogrulugu.md`: kural tabanlı çıkarıcının ve Çalışma 1 kavram başlıklarının uzman etiketine göre doğruluğu
  - `results/annotations/power_pilot.json`: sürekli pH güç pilotu

### Ayar geçmişi (plan yazılırken bilinenler)

Dürüstlük için, plan yazılmadan önce görülen her şey burada listelenir:

1. **Çalışma 1'in bütün sonuçları** (ayırt edicilik, kalibrasyon, ablasyonlar, duyarlılık analizleri). Çalışma 1 modelleri burada yalnızca kontrol koşulu olarak, yeniden eğitilmeden kullanılır.
2. **Kural tabanlı çıkarıcının ve Çalışma 1 kavram başlıklarının uzman etiketine göre sadakati** (`docs/kavram_dogrulugu.md`). Sonuç değişkeni kullanılmadı.
3. **Romagnoli etiket yoğunluğunun FHR kaybına göre dağılımı** (keşifsel). Sonuç değişkeni kullanılmadı. §3.2'deki kalite maskesi eşikleri bu bilgiyle seçildi.
4. **Güç pilotu:** Çalışma 1'in kat dışı tahminleri ile sürekli pH arasındaki C-indeksi, yalnızca güven aralığı genişliklerini görmek için hesaplandı. Sonuç değişkeni kullanıldı, ama Çalışma 2'nin hiçbir modeli henüz yok.

**Çalışma 2'nin önerilen modeli (NeSy-v2) bu plan yazılırken ne kodlanmış ne de eğitilmiştir.**

---

## 1. Soru, uç noktalar ve hipotezler

**Soru:** Uzman olay etiketleriyle, kaliteye duyarlı biçimde denetlenen kavram katmanı ve öğrenilebilir olasılıksal FIGO mantığı, (i) kavramların uzman etiketine sadakatini zayıf denetime ve kural tabanlı çıkarıcıya göre artırır mı, (ii) bunu yaparken asidemi ayırt ediciliğinden ödün verir mi?

| Kod | Hipotez | Uç nokta ve test | Aile |
|---|---|---|---|
| **H1a** (birincil) | NeSy-v2'nin deselerasyon (DEC) olay tespiti, kural tabanlı çıkarıcıdan daha iyidir | Değerlendirme bölgesinde (§3.3) havuzlanmış olay düzeyi **F1 farkı** (NeSy-v2 − kural). Kayıt düzeyinde eşleştirilmiş bootstrap (2000) | Holm, H1 ailesi (3 test) |
| **H1b** (birincil) | Akselerasyon (ACC) olay tespitinde aynı | Aynı | Holm, H1 ailesi |
| **H1c** (birincil) | NeSy-v2'nin pencere düzeyi "akselerasyon var" kavramı, uzmandan türetilen etikete göre Çalışma 1'in zayıf denetimli başlığından (`nesy_ens_W60`) daha ayırt edicidir | Uzman etiketine karşı **ΔAUROC**; eşleştirilmiş DeLong. Yalnız açıklanmış ve kayıt düzeyi kalite ölçütünü (§3.2) sağlayan kayıtlarda; tekrar ortalamalı başlık olasılıklarıyla | Holm, H1 ailesi |
| **H2** | Ayrı optimizer grubuyla eğitilen kural ağırlıkları başlangıçtan ayrılır ve FIGO sıralamasıyla tutarlıdır | (a) "patolojik" kural ağırlığının, 50 dış kat ensemble ortalamaları üzerinden medyanı ile başlangıç değeri 0,10 arasındaki mutlak fark ≥ 0,05; (b) dış katların ≥ %90'ında w(patolojik) > w(şüpheli). İkisi birlikte sağlanırsa desteklenir | Betimsel |
| **H3** | NeSy-v2, pH < 7,05 için AUROC'ta CNN-Bayes'ten aşağı değildir | v1 H1a ile aynı: ΔAUROC'un %95 bootstrap GA alt sınırı > −0,05 | Tek test |
| **H4** (ikincil) | Sürekli pH için C-indeksi | Harrell C (yüksek risk ↔ düşük pH); NeSy-v2 ile CNN-Bayes farkı, eşleştirilmiş bootstrap | Betimsel |

- **Birincil sonuç bildirimi:** H1a–c, Holm düzeltmeli. Fark > 0 ve düzeltilmiş p < 0,05 ise "desteklendi".
- **Ayırt edicilikte üstünlük iddia edilmez.** Bu çalışmanın asıl iddiası kavram sadakatidir.

## 2. Veri ve popülasyon (v1 §2–§3'ten aynen)

- CTU-UHB, W60 penceresi, v1 ön işlemesi, 536 kayıt, 38 pozitif. Girdi kanalları [FHR, FHR maskesi, UC, UC maskesi] ve yasaklı değişken listesi aynen geçerlidir.
- **Yeni:** Uzman etiketleri, yalnızca **eğitim hedefi** ve **değerlendirme referansı** olarak kullanılır; asla model girdisi değildir.
  - Romagnoli 2020: olay düzeyi ACC, DEC ve tipi (E/L/V/P), UC, BC, TC.
  - FHRMA: 66 kayıt; yalnız dış kavram doğrulaması.
- **Bir dış katın test kayıtlarının uzman etiketleri, o katın eğitiminde, ayarında, eşik seçiminde ve kalibrasyonunda hiçbir biçimde kullanılmaz.**
- Pencereler: Yalnız W60. W30, W90 ve S1_60 duyarlılık analizleri **yapılmayacak** (hesap bütçesi; Çalışma 1'de yapıldı).

## 3. Uzman kavram hedefleri ve kalite maskesi

### 3.1 Zaman çözünürlüklü hedefler (çerçeve düzeyi)

- **Çözünürlük:** 4 sn'lik bölmeler (W60'ta 900 bölme).
- **Etiketler:** Bir bölme, Romagnoli olay aralığıyla en az %50 örtüşüyorsa o olay için 1 sayılır.
- **Başlıklar:**
  - ACC
  - DEC
  - DEC tipi (E/L/V/P, yalnız DEC=1 bölmelerinde; 4 sınıf)
  - UC
- **BC ve TC** çok seyrek (43 ve 54 olay). Çerçeve başlığına alınmaz; yalnız betimsel raporlanır.
- **Sansürlü olaylar** (kayıt sonuna kadar kapanmayan): bitişleri açıklama uzunluğu kabul edilir.
- **Kaynak hataları** (başlangıçsız bitiş, 21 adet): olay üretilmez.

### 3.2 Kalite maskesi ("etiket yok" ≠ "olay yok")

Gerekçe: Romagnoli'nin, kalitesi düşük kayıtlarda eksik etiketlediği gösterildi (`docs/kavram_dogrulugu.md` §3).

- **Bölme düzeyinde maske** q(t) = 1, ancak ve ancak şunların hepsi sağlanıyorsa:
  1. Kayıt açıklanmış (en az bir Romagnoli olayı var; 23 kayıt bu koşulu sağlamıyor).
  2. Bölme açıklama kapsamı içinde.
  3. Bölmenin merkezlendiği 60 sn'de, temizlenmiş 1 Hz FHR maskesinin geçerli oranı ≥ %90.
- **Kayıt düzeyi ölçüt:** W60 FHR eksikliği ≤ %20 (kayıtların yaklaşık üçte ikisi). Bu ölçütü sağlamayan kayıtlarda q(t) = 0 kabul edilir.
- **Asimetrik denetim:**
  - Uzman olayı içindeki bölmeler (pozitifler) her zaman denetlenir.
  - Negatif bölmeler yalnızca q(t) = 1 ise denetlenir.
  - Diğer bölmelerde çerçeve kaybı hesaplanmaz.

### 3.3 Değerlendirme bölgesi

- H1a ve H1b, yalnızca dış test kayıtlarında ve q(t) = 1 bölgesinde değerlendirilir.
- Bir olay (uzman ya da tahmin), süresinin en az %50'si bu bölgedeyse sayılır.
- **İkincil olarak**, maske uygulanmadan (yalnız açıklama kapsamı içinde) aynı metrikler de raporlanır.

### 3.4 Pencere düzeyi kavramlar

Uzmandan türetilen tanımlar `src/eval/concept_fidelity.py` ile aynıdır:
- Akselerasyon: herhangi bir ACC.
- Erken deselerasyon: herhangi bir E.
- Uzamış deselerasyon: herhangi bir P.
- > 5 dk deselerasyon: herhangi bir DEC > 300 sn.
- Tekrarlayan geç / değişken / herhangi: uzman kasılmalarının > %50'sinde, FHR dibi kasılma tepesine ±60 sn içinde olan, ilgili tipte bir DEC.

Kurallar:
- **Negatif etiketler** yalnızca kayıt düzeyi kalite ölçütünü sağlayan kayıtlarda kullanılır. Diğer kayıtlarda yalnız pozitif etiketler kullanılır; negatifler "bilinmiyor" sayılır.
- **Uzman referansı olmayan kavramlar** kural tabanlı zayıf etiketlerle denetlenmeye devam eder ve **zayıf** diye işaretlenir. Bunlar: taban çizgisi kategorisi, değişkenlik kategorisi, uzun süreli azalmış/artmış değişkenlik. Bu kavramlar için sadakat iddiası yapılmaz.
- **Uzun süreli tekrarlayan geç deselerasyon:** Uzman DEC'leriyle hesaplanır. Ancak tanımdaki "azalmış değişkenlik" koşulu zayıf etiketten gelir (karma etiket).

## 4. Modeller

### 4.1 NeSy-v2 (önerilen)

- **Omurga:** v1 ile aynı `CNNEncoder` ve aynı ayar uzayı (width, n_stages, kernel, lr, wd, drop).
- **Çerçeve başlığı:** İlk artık bloğun çıktısı (adım 4 sn) ile en derin özellik haritasının 4 sn'ye doğrusal olarak yukarı örneklenmiş hâli birleştirilir. 1×1 evrişimle ACC, DEC, UC (sigmoid) ve DEC tipi (softmax 4) üretilir.
- **Pencere kavram başlıkları:** Çalışma 1 ile aynı 18 boyut (taban 4, değişkenlik 4, 10 ikili). Mantık katmanına bunlar girer.
- **Mantık katmanı:** Çalışma 1'in FIGO programı ve tam WMC katmanı, değişmeden.
- **Kayıp:**

  L = BCE_sonuç (pozitif ağırlık, v1 ile aynı) + 1,0 · L_çerçeve + 1,0 · L_pencere-uzman + 0,1 · L_pencere-zayıf

  - L_çerçeve: maskeli; ACC, DEC, UC için BCE; tip için çapraz entropi (yalnız DEC bölmeleri). Bileşenlerin ortalaması alınır.
  - L_pencere-uzman: §3.4'teki uzman kavramları; "bilinmiyor" olanlar hariç tutulur.
  - L_pencere-zayıf: uzman referansı olmayan kavramlar.
  - **Kayıp ağırlıkları sabittir, ayarlanmaz.** Gerekçe: ayar, iç AUROC ile yapıldığı için kavram denetimi küçültülerek seçilebilirdi.
- **Kural parametreleri için ayrı optimizer grubu:** öğrenme oranı = 10 × omurga öğrenme oranı, ağırlık sönümü = 0. Çalışma 1'deki "kural ağırlıkları başlangıçta kaldı" sorununun doğrudan düzeltmesi.
- **Bayesçi katman ve kalibrasyon:** v1 ile aynı. 5 üyeli topluluk, önsel düzeltme + sıcaklık ölçekleme (iç doğrulama %15'i üzerinde), aynı erken durdurma.
- **Çerçeve olay eşiği:** Her başlık için, dış-eğitimin %15'lik iç doğrulama kümesinde olay düzeyi F1'i en büyükleyen olasılık eşiği seçilir. Aday eşikler 0,1–0,9, 0,05 adımla.

### 4.2 Olay sonrası işleme (bütün çerçeve çıktıları için sabit)

1. Topluluk ortalaması alınmış çerçeve olasılığı eşikten büyükse bölme pozitif sayılır.
2. ≤ 1 bölme (4 sn) aralıklı pozitif koşular birleştirilir.
3. En kısa süre: ACC ve DEC için 4 bölme (16 sn), UC için 8 bölme (32 sn). Daha kısa koşular atılır.
4. Tespit edilen DEC'in tipi, koşu boyunca ortalama tip olasılığının en büyüğüdür.

### 4.3 Kontrol koşulları (yeniden eğitilmez; Çalışma 1 çıktıları)

| Koşul | Kullanım |
|---|---|
| Kural tabanlı çıkarıcı (`src/features/figo.py`, W60) | H1a, H1b |
| `nesy_ens_W60`: zayıf denetimli NeSy | H1c; sızıntı karşılaştırması |
| `nesy_nocsup_W60`: kavram denetimsiz NeSy | Betimsel |
| `cnn_ens_W60`: CNN-Bayes | H3, H4 |

### 4.4 Ablasyon (önceden tanımlı, ikincil)

**NeSy-v2-maskesiz:** Kalite maskesi yok. Kapsam içindeki her etiketsiz bölme ve her pencere kavramı negatif kabul edilir. Diğer her şey NeSy-v2 ile aynıdır; ilgili dış katta NeSy-v2'nin seçilen yapılandırması kullanılır.

Amaç: K3'ün pratik önemini ölçmek. Çıktılar: H1 metrikleri (maskeli değerlendirme bölgesinde) ve ΔAUROC. Bu ablasyon Holm ailesine girmez.

**Başka model varyantı denenmez.** (38 pozitif; çoklu deneme riski.)

## 5. Bölme, ayar ve birleştirme

- **Bölme:** `results/splits/outer_splits.json`; v1 ile aynı 5 × 10 dış kat; iç 3 kat; 6 rastgele yapılandırma; seçim ölçütü iç AUROC. Çalışma 1'le birebir karşılaştırılabilirlik için hiçbiri değişmez.
- **Birleştirme:** Birincil analiz, 5 tekrarın test tahminlerinin ortalaması üzerinden yapılır:
  - Sonuç tahmini: kayıt başına olasılık ortalaması.
  - Çerçeve çıktıları: bölme başına olasılık ortalaması.
  - Kavram başlıkları: kayıt başına olasılık ortalaması.
- **Çerçeve eşiği:** Kat başına iç doğrulamada seçildiği için, tekrar ortalamalı çerçeve olasılıklarına ilgili katların eşiklerinin **medyanı** uygulanır.
- Tekrar başına değerler ayrıca raporlanır.

## 6. Metrikler ve istatistik

### 6.1 Kavram sadakati (H1 ve ikincil)

- **Olay düzeyi (ACC, DEC, UC):** Bire bir, örtüşmeye göre açgözlü eşleştirme. Birincil ölçüt: herhangi bir örtüşme; ikincil: IoU ≥ 0,5. Havuzlanmış Se, PPV, F1; kayıt düzeyinde 2000 bootstrap.
- **DEC tipi:** Eşleşen çiftlerde E/L/V/P makro-F1 ve karışıklık matrisi. NeSy-v2 ile kural tabanlı çıkarıcı karşılaştırılır. Keşifsel; L ve P örnekleri az.
- **Pencere kavramları:** Uzmandan türetilen etikete karşı AUROC ve κ (eşik 0,5). Karşılaştırılanlar: NeSy-v2, Çalışma 1 başlıkları, zayıf etiketler.
  - Pencere düzeyinde ≥ 20 pozitifi olmayan kavramlarda pencere sadakati **raporlanmaz**. Kural tabanlı çıkarıcının ön çalışmasına göre bunlar: > 5 dk, tekrarlayan geç, tekrarlayan değişken.
- **Taban çizgisi:** Çerçeve başlığında yok. Kavram düzeyinde sadakati ölçülmez.

### 6.2 Sızıntı (önceden tanımlı, ikincil)

- **Müdahale testi:** Test kayıtlarında, uzman referansı olan pencere kavramları (akselerasyon, erken, uzamış, > 5 dk, tekrarlayan geç / değişken / herhangi) mantık katmanına modelin tahmini yerine uzmandan türetilen değerlerle verilir. Bu yalnızca kayıt düzeyi kalite ölçütünü sağlayan kayıtlarda yapılır. Raporlananlar:
  - Sonuç log-kaybındaki değişim
  - AUROC değişimi
  - Modelin olasılıkları ile müdahaleli olasılıklar arasındaki ortalama mutlak fark
- Aynı test, ancak çerçeve başlıkları olmadan, `nesy_ens_W60` için de raporlanır. Çalışma 1'in kavram başlıkları kayıtlı (`x_c_bits`); mantık katmanı, kayıtlı kural ağırlıklarıyla yeniden hesaplanabilir.
- **Yorum:** Müdahaleyle değişim ne kadar küçükse, sonuç kavramlar üzerinden o kadar akıyor demektir. Bu ölçüt Havasi 2022 ve Parisini 2025'teki "müdahale" yaklaşımlarının basit bir uyarlamasıdır; sınırlılığı raporlanır.

### 6.3 Ayırt edicilik ve kalibrasyon (H3, H4, ikincil)

- v1 §5.1 metrikleri aynen: AUROC, AUPRC, Brier, ECE, kalibrasyon eğimi, karar eğrisi.
- **H3:** NeSy-v2 − CNN-Bayes; eşleştirilmiş bootstrap (2000) ve DeLong.
- **H4:** Harrell C; eşleştirilmiş bootstrap (2000).
- **Seçici tahmin:** v1 §5.3 ile aynı biçimde **yalnız betimsel** (Çalışma 1'de H2 desteklenmedi).

### 6.4 Referans standardı denetimi (K3, önceden tanımlı, betimsel)

- Kayıt düzeyinde Romagnoli etiket yoğunluğu (olay/saat, açıklama kapsamında) ile W60 FHR eksikliği arasındaki Spearman korelasyonu.
- FHR eksikliği üçte birlik dilimleri içinde etiket yoğunluğunun pH < 7,05 ve ≥ 7,05 grupları arasında karşılaştırılması (Mann–Whitney; betimsel).
- Maskelenen bölge oranının sonuç gruplarına göre dağılımı.

### 6.5 Dış kavram doğrulaması: FHRMA (ikincil)

- **Veri:** FHRMA'nın 66 eğitim kaydı.
- **Uygulama:** Kayıtlar art arda 60 dk'lık pencerelere bölünür; son pencere maskeli dolguyla tamamlanır. Ön işleme v1 ile aynıdır: TOCO, UC kanalı olarak aynı ölçeklemeyle kullanılır. TOCO tamamen sıfırsa UC maskesi 0 yapılır.
- **Model:** Tekrar 0'ın 10 dış kat topluluğunun ortalaması (50 üye).
- **Değerlendirme:** Uzmanın "analiz etme" ve "güvenilmez" işaretlediği bölümler dışlanır. ACC ve DEC olay F1'i; NeSy-v2 ile kural tabanlı çıkarıcı karşılaştırılır.
- Taban çizgisi değerlendirmesi yalnız kural tabanlı çıkarıcı için (`docs/kavram_dogrulugu.md`).

### 6.6 Bilinçli olarak yapılmayacaklar

| Konu | Gerekçe |
|---|---|
| 9 uzmanın FIGO sınıflarıyla karşılaştırma | Hruban/Spilka CSV'sindeki `eval_step1–4` sütunlarının anlamı doğrulanamadı. Doğrulanırsa plan dışı ek analiz olarak eklenebilir |
| Kontrollü maskelemeyle eksik veri dayanıklılığı | Çalışma 1 model ağırlıkları kaydedilmedi, karşılaştırma koşulu yok. Ayrıca sinyal kaybı sonuçla ilişkili (bilgi taşıyan eksiklik). Gelecek çalışma |
| Nöral ODE/LTC omurgası | Kullanıcı kararı (2026-11-23) |
| Sürekli pH regresyon başlığı | Model sayısını artırmamak için. H4, sınıflandırma risk skoru üzerinden |
| W30/W90/S1_60, ikincil etiketler | Hesap bütçesi |

## 7. Sızıntı kontrol listesi (v1 §4'e ek)

- [ ] Bir dış katın test kayıtlarının Romagnoli etiketleri, o katın eğitim, ayar, eşik ve kalibrasyon adımlarında yüklenmiyor (otomatik `assert`: hedef tablosu yalnız dış-eğitim kimlikleriyle süzülür).
- [ ] Kalite maskesi yalnız sinyal maskesinden ve açıklama kapsamından hesaplanıyor; sonuç değişkeni kullanılmıyor.
- [ ] Uzman etiketleri girdi kanallarında yok.

## 8. Güç gerekçesi

Kural tabanlı çıkarıcının uzman etiketine göre bootstrap güven aralıkları (`docs/kavram_dogrulugu.md`, maskesiz) kullanılarak:

| Uç nokta | Tahmini SE | %80 güç, iki yönlü α = 0,05 için en küçük saptanabilir fark |
|---|---|---|
| DEC F1 (1910 olay; GA genişliği 0,06) | ≈ 0,015 | ≈ 0,04–0,05 |
| ACC F1 (666 olay; GA genişliği 0,08) | ≈ 0,02 | ≈ 0,06 |
| Akselerasyon pencere AUROC (n ≈ 340 kalite ölçütlü kayıt, yaygınlık ≈ 0,45) | ≈ 0,03 | ≈ 0,08 |

- Maske ve kalite ölçütüyle olay sayısı azalacağından SE'ler biraz büyüyebilir.
- Eşleştirilmiş karşılaştırmada gerçek SE daha küçük olabilir.
- **H3:** Çalışma 1'de ΔAUROC'un SE'si ≈ 0,016. Gerçek fark 0 ise GA alt sınırı ≈ −0,03 olur ve −0,05 marjı karşılanabilir.
- **H4:** Pilotta model farklarının SE'si C-indeksinde ≈ 0,006–0,018.

## 9. Raporlama

- Bütün koşular `experiments/log.csv` dosyasına yazılır.
- Desteklenmeyen hipotezler açıkça yazılır.
- Çalışma 1'in olumsuz sonuçları kontrol koşulu ve motivasyon olarak ana metinde yer alır.
- Romagnoli dosyaları ve kayıt düzeyindeki türevleri yayımlanmaz (CC BY-NC-ND); yalnız toplu sonuçlar ve kod paylaşılır.
- TRIPOD+AI kontrol listesine göre raporlanır.

## 10. Önceden bilinen sınırlılıklar

- Romagnoli tek uzmanlı ve yarı otomatik; FHRMA farklı merkezden ve yalnız I. evreden.
- Taban çizgisi ve değişkenlik kavramlarında uzman referansı yok (zayıf denetim sürüyor).
- L ve P tipi deselerasyonlar az; tip sonuçları keşifsel.
- 38 pozitif: ayırt edicilikte düşük güç (H3 yalnız aşağı olmama).
- Kalite maskesi eşikleri, sonuç değişkeni kullanılmadan ama etiket yoğunluğu analizi görüldükten sonra seçildi (bkz. Ayar geçmişi).
- Dış pH doğrulaması yok.
