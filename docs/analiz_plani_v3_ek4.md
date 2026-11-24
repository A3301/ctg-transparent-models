# Analiz planı v3, Ek 4: taşınabilirlik yedek planları B ve C

Tarih: 2026-11-23.

**Zamanlama:** Ek 3 sonuçları görüldükten sonra, bu ekteki hiçbir analiz koşulmadan yazıldı. `docs/yontem_yedek_planlari.md` §1'deki önceden yazılmış B ve C planlarının uygulamasıdır. **Betimsel analiz.**

**Tetik (Ek 3 sonuçları):**
- C1 ayrımında şeffaf modellerin AUROC'u CNN'den düşük çıktı: doğrusal −0,062, GAM −0,135; ikisi de anlamlı değil.
- Bütün modellerde kalibrasyon kesişimi büyük: +1,5 ile +2,2.

## Plan B: hedef grupta yeniden kalibrasyon

- Ek 3'ün test tahminlerine logit ölçeğinde yeniden kalibrasyon uygulanır. İki seçenek:
  - yalnız kesişim güncellemesi;
  - kesişim ve eğim güncellemesi.
- Yöntem: hedef grup içinde birini-dışarıda-bırak çapraz doğrulama.
- Raporlanan: güncelleme öncesi ve sonrası Brier skoru, kalibrasyon eğimi ve kesişimi.
- AUROC değişmez (monoton dönüşüm); ayrı raporlanmaz.

## Plan C: UC'den bağımsız şeffaf modeller

- **Yeni modeller:** `transp_lr_nouc`, `transp_gam_nouc`.
- **Girdi:** `transp_*` ile aynı, ancak kontraksiyona bağlı 7 değişken çıkarılır:
  - `uc_per_10min`;
  - `late_frac_contr`, `variable_frac_contr`, `early_frac_contr`, `any_dec_frac_contr`;
  - `uc_missing`, `uc_longest_gap_s`.
- Kalan değişken sayısı 23.
- **Koşular:**
  - Ana protokolde 5×10 iç içe çapraz doğrulama (W60, pH < 7,05), CNN ve `transp_*` ile karşılaştırma için;
  - Ek 3'ün C1 ve C3 ayrımları.
- **Raporlanan:**
  - AUROC (%95 GA);
  - CNN'e göre ΔAUROC;
  - aynı ayrımda `transp_*` ile fark.

## Yorum kuralı (önceden)

Taşınabilirlik ayrımlarında yalnız 13 ve 29 test vakası var. Bu nedenle:
- tek ayrımda görülen farklar bulgu olarak değil, yön olarak yorumlanır;
- "CNN daha dayanıklı" ya da "şeffaf model daha dayanıklı" ifadesi, ancak iki ayrımda aynı yönde ve güven aralıkları sıfırı dışlıyorsa kullanılır.
