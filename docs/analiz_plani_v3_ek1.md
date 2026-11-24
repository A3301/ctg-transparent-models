# Analiz planı v3, Ek 1: Plan D (şeffaf modellerin sağlamlığı)

Tarih: 2026-11-23. **Zamanlama:** Plan C sonuçları görüldükten sonra, Plan B sonuçları görülmeden yazıldı. Bu yüzden Plan D'nin bütün analizleri **betimseldir** ve hipotez testi yapılmaz. Dondurma kanıtı: SHA-256 özeti `experiments/log.csv` dosyasında (`phase = S3-plan`).

## Neden

Plan C'nin ikincil modeli `transp_lr`, birincil analizde CNN-Bayes'ten sayıca yüksek çıktı:

| Model | AUROC | Tek tekrar AUROC |
|---|---|---|
| `transp_lr` | 0,812 | 0,785 ± 0,013 |
| CNN-Bayes | 0,794 | 0,745 ± 0,048 |

Birincil hipotezler C1 ve C2 ise desteklenmedi: güven aralıkları geniş. Tek bir sonuç tanımı ve tek bir pencereyle elde edilen bu fark rastlantı olabilir. Plan D, Çalışma 1'in duyarlılık analizlerini şeffaf modellere aynen uygulayarak sonucun **farklı etiket ve pencerelerde tutarlı olup olmadığını** gösterir.

## Ne yapılacak

- **Modeller:**
  - `transp_lr`, `transp_gam`;
  - girdiler v3 §3.1'deki gibi, ilgili pencerenin öznitelik tablolarından.
- **Ayar:** Tam iç içe ayar (30 yapılandırma), Çalışma 1'deki `figo_lr` duyarlılık koşularıyla aynı. Bölme dosyaları, Çalışma 1'de her ayar için kullanılanlar.
- **Ayarlar (7):**

  | Grup | Ayar | Ne değişiyor |
  |---|---|---|
  | Sonuç tanımı | pH < 7,15 | Etiket |
  | | BDecf ≥ 12 | Etiket |
  | | Apgar5 < 7 | Etiket |
  | Kayıt kümesi | 552 kayıt (dışlama yok) | Kayıtlar |
  | Pencere | W30 | Son 30 dk |
  | | W90 | Son 90 dk |
  | | S1_60 | I. evrenin son 60 dk'sı |

- **Karşılaştırma:** Aynı ayardaki Çalışma 1 modelleri:
  - `cnn_ens_*` (sabit yapılandırmalı);
  - `figo_lr_*`.

  Her ayar için ΔAUROC (şeffaf − CNN) ve %95 eşleştirilmiş bootstrap GA (2000).
- **Raporlama:** Tekrar ortalamalı ve tek tekrar AUROC. Tutarlılık özeti: 7 ayarın kaçında ΔAUROC > 0. Test yok, Holm yok.

## Sınırlılık (önceden yazılı)

Bu ek, Plan C sonuçları bilinerek eklendi. Bulgular doğrulayıcı değil, betimsel kabul edilir. Doğrulama ancak dış veriyle yapılabilir.
