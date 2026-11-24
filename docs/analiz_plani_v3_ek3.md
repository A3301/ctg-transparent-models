# Analiz planı v3, Ek 3: Plan C, taşınabilirlik ("dışa benzer") doğrulaması

Tarih: 2026-11-23. Hiçbir C koşusu yapılmadan yazıldı. Dondurma kanıtı: SHA-256 özeti `experiments/log.csv` dosyasında (`phase = S3-plan`). **Betimsel analiz; hipotez testi yok.**

## Amaç

Dış veri yokken modellerin dağılım kaymasına dayanıklılığını açık veriyle sınamak.

## Ayrımlar (536 analiz kaydı, W60, pH < 7,05)

| Kod | Eğitim | Test | Gerekçe |
|---|---|---|---|
| **C1** | Vajinal doğum kayıtları (ID 1001–1506) | Sezaryen kayıtları (ID 2001–2046) | Farklı klinik yol; kayıt operatif doğumdan önce biter. Grup ataması PhysioNet'teki 46 sezaryen sayısıyla doğrulanıyor |
| **C3** | FHR eksikliği ≤ medyan (W60) | FHR eksikliği > medyan | Sinyal kalitesi kayması |

## Modeller

Protokol Çalışma 1 ve v3 ile aynı:
- iç 3 katlı ayar (sklearn için 30, CNN için 6 yapılandırma);
- %15 kalibrasyon ayrımı;
- önsel düzeltme ve sıcaklık ölçekleme.

| Model | Yapılandırma |
|---|---|
| `transp_lr`, `transp_gam` | Şeffaf |
| `figo_lr`, `georgoulas` | Karşılaştırma |
| `cnn` | 5 üyeli topluluk |

**Tekrar ve tohum:** Her ayrım için 5 tekrar, her tekrar farklı tohumla (iç ayar ve kalibrasyon ayrımı değişir). Test tahminleri tekrarlar üzerinden ortalanır.

## Raporlama

- Test grubunda AUROC (%95 bootstrap GA, 2000), Brier skoru, kalibrasyon eğimi ve kesişimi.
- Şeffaf modeller ile CNN arasında ΔAUROC (%95 GA).
- Test gruplarındaki asidemik vaka sayısı küçük (C1'de 13); sonuçlar betimseldir.
