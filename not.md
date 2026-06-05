### 
- **Dosya:** `preprocessing.py` (Hepsi)
- **Güncelleme:** Her şey yolunda.


**Güncelleme:** `temizlenmis_yorumlar.csv` dosyasındaki verilere vektör embedding işlemi uygulayabilmek için colabde `Crimson_desert.ipynb` adlı yeni bir dosya oluşturduk. Buradan devam edeceğiz.

### Elimizdeki yorumları sayısal vektörlere, ihtiyacımız olan diğer verileri (oynama_suresi gibi) metadata formatına çevirdik. Bunları ChromaDB kullanarak vektör veritabanına kaydettik.
- **Dosya:** `Crimson_desert.ipynb` ([15])
- **Güncelleme:** Her şey yolunda.

### chroma_db vektör veritabanını app.py ile bağladık. `benzer_yorumlari_bul` fonksiyonu ile kullanıcı sorgusuna göre en alakalı 3 yorumu getirebiliyoruz.
- **Dosya:** `app.py` (Hepsi)
- **Sorun:** LLM'in örneğin oyunda en sık karşılaşılan hata nedir sorusuna on binlerce yorumu vektör veritabanı yardımı ile analiz edip çıkan sonucu doğal dil ile kullanıcıya vermesini nasıl sağlayabiliriz?
- **Çözüm:** Elimizdeki yorumları kümelere ayırıp her kümeden yorumu en iyi anlatan 1 temsilci yorum seçeceğiz. Temsilci yorumları ve her kümedeki toplam yorum sayısını LLM'e göndereceğiz. LLM bu bağlam bilgisine göre oyun hatalarını yorumlayacaktır.

### UMAP ile vektör veritabanındaki çok boyutlu vektörleri HDBSCAN algoritmasının kümeleme yapabilmesi için boyut indirgeme işlemi yapıyoruz. Ardından vektörlerin birbirlerine yakınlıklarına göre kümeleme işlemi yapıyoruz.
- **Dosya:** `ReviewClusterer.py` (Hepsi)
- **Güncelleme:** Her şey yolunda.

### Vektör veritabanından verileri çekip ReviewClusterer sınıfına gönderiyoruz. Buradan kümeleme işlemi sonucunda küme labelleri alıyoruz.
- **Dosya:** `vektor_veri_cekimi.py` (Hepsi)
- **Güncelleme:** Her şey yolunda.

### Hiçbir kümeye dahil olmayan -1 ile işaretlenmiş yorumları temizliyoruz.
- **Dosya:** `vektor_veri_temizligi.py` (Hepsi)
- **Güncelleme:** Her şey yolunda.

### 15.000 yorum verisini ingilizce ve sadece olumsuz yorumlar olacak şekilde yeniden çetik. Ardından `crimson_desert_ingilizce_olumsuz_yorumlari.csv` dosyasına kaydettik. Bunların 1355 tanesi kullanılabilir yorum oldu.
- **Dosya:** `1_veri_cekimi.py` (Hepsi)
- **Güncelleme:** Her şey yolunda.

###
- **Dosya:** `preprocessing.py` (Hepsi)
- **Güncelleme:** Yeni preprocessing işlemini colabde yaptık.

### Temizlenmiş yorumları 59 kümeye ayırıp küemeledik. Her yorumun yorum değeri skorunu oynama süresi ve faydalı bulunma sayısına göre belirledik. Bu yorumların küme içerisindeki yorum değeri ortalamalarına göre her bir küme için küme kritiklik skoru hesapladık. 
- **Dosya:** `google colab Crimson_Desert_Embedding` (Hepsi)
- **Güncelleme:** Kümeleri ve 1355 yorumu aynı vektör veritabanına kaydedeceğiz. Her kümeyi tek tek LLM'e gönderip küme ismi, küme özeti vb. gerekli bilgileri almalıyız. 