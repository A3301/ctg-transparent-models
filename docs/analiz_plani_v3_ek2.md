# Analiz planı v3, Ek 2: Plan E (kural rehberli dedektör)

Tarih: 2026-11-23. **Zamanlama:** Plan B, C ve D sonuçları görüldükten sonra, Plan E çalıştırılmadan yazıldı. Dondurma kanıtı: SHA-256 özeti `experiments/log.csv` dosyasında (`phase = S3-plan`).

## Neden

Plan B'nin ardışık dedektörü (`det`), uçtan uca NeSy-v2'ye göre kavram sadakatini belirgin biçimde artırdı. Kural tabanlı çıkarıcıyı ise ACC ve DEC'te geçemedi:

| Olay | `det` F1 | Kural tabanlı F1 |
|---|---|---|
| ACC | 0,449 | 0,474 |
| DEC | 0,627 | 0,649 |

İki yöntemin hata profili farklı. Dedektörün PPV'si yüksek, duyarlılığı düşük; kural çıkarıcıda tersi. Kural çıkarıcı FIGO tanımlarını açıkça kodluyor, dedektör ise uzman sınır kararlarını öğreniyor. Kural çıktısını dedektöre girdi olarak vermek, dedektörün yalnızca **kuralları düzeltmeyi** öğrenmesini sağlar. Böylece kuralın duyarlılığı ile dedektörün seçiciliği birleşebilir.

## Ne değişiyor (`detk`)

- **Girdi:** `det`'in 5 kanalına 3 kanal eklenir. Bunlar, kural çıkarıcının 4 Hz'de bulduğu ACC, DEC ve UC olaylarının 1 Hz göstergeleri (0/1). Toplam 8 kanal.
  - Kanallar yalnızca sinyalden hesaplanır: `detect_events`, `detect_contractions`.
  - Sonuç değişkeni ya da uzman etiketi kullanılmaz.
- **Değişmeyenler** (`det` ile aynı):
  - mimari ve kayıp;
  - 6 yapılandırmalı iç ayar ve olay F1'iyle seçim;
  - 5 üyeli topluluk, eşik seçimi, çapraz uydurma;
  - aşama 2 (`transpdetk_gam`, 30 yapılandırma).

## Hipotezler

| Kod | Hipotez | Ölçüt | Aile |
|---|---|---|---|
| **E1a** | `detk` DEC olay F1'i kural çıkarıcıdan yüksek | v3 B1a ile aynı | Holm (E1a, E1b) |
| **E1b** | `detk` ACC olay F1'i kural çıkarıcıdan yüksek | v3 B1b ile aynı | Holm (E1a, E1b) |
| E2 (ikincil) | FHRMA'da `detk` ile kural çıkarıcının ACC/DEC F1 farkı | v3 §6.3 ile aynı yöntem | Betimsel |
| E3 (betimsel) | `transpdetk_gam` AUROC'u ve CNN-Bayes'e göre ΔAUROC | %95 GA | — |
| E4 (betimsel) | `detk` ile `det` olay F1 farkı | %95 GA | — |

## Sınırlılık (önceden yazılı)

- Bu ek, Plan B, C ve D'nin sonuçları bilinerek eklendi.
- Plan E, Plan B'ye karşı tek bir tasarım değişikliğidir. Başka varyant denenmez.
