# ⚔️ Crimson Desert — Oyuncu Geri Bildirim Analiz Sistemi

Steam üzerindeki **Crimson Desert** oyuncu yorumlarını NLP teknikleriyle analiz
ederek, geliştirici ekibin hata ve şikâyetleri hızlıca tespit etmesini sağlayan
yapay zeka destekli bir QA (Quality Assurance) asistanıdır.

> **Amaç:** 15.000+ olumsuz oyuncu yorumunu otomatik olarak kümeleyip, doğal dil
> sorguları ile aranabilir hale getirmek — böylece geliştirme ekibinin
> "oyuncular en çok neden şikâyetçi?" gibi sorulara veri odaklı yanıtlar
> almasını sağlamak.

---

## 📸 Arayüz Önizlemesi

Uygulama, tarayıcı üzerinde çalışan bir sohbet arayüzü sunar. Geliştirici doğal
dilde soru sorar, sistem veritabanını analiz edip rapor üretir:

```
🧑‍💻 Sen: İade eden oyuncular en çok neden şikâyetçi?
🤖 Asistan: İade eden oyuncuların en sık dile getirdiği sorunlar şunlar:
   • Optimizasyon ve FPS düşüşleri
   • Savaş mekaniğinin tekrara düşmesi
   • Açık dünyanın boş ve ruhsuz hissettirmesi
```

---

## 🏗️ Sistem Mimarisi

Proje, uçtan uca bir NLP pipeline üzerine inşa edilmiştir:

```
Steam API → Veri Çekimi → Ön İşleme → Embedding → Kümeleme → Vektör DB → RAG Asistanı
```

### Detaylı Akış

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VERİ HAZIRLAMA                               │
│                                                                     │
│  1. Steam API'den 15.000+ olumsuz yorum çekildi                    │
│  2. Yorumlar temizlendi ve İngilizce olanlar filtrelendi            │
│  3. Sentence Transformers ile vektör embedding uygulandı            │
│  4. UMAP ile boyut indirgeme → HDBSCAN ile kümeleme yapıldı        │
│  5. 59 anlam kümesi oluşturuldu, her kümeye LLM ile isim verildi   │
│  6. Küme özetleri + tekil yorumlar ChromaDB'ye kaydedildi           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG SORGULAMA SİSTEMİ                           │
│                                                                     │
│  Kullanıcı Sorusu                                                   │
│       │                                                             │
│       ▼                                                             │
│  [Trafik Polisi LLM] → Sorguyu analiz et, filtre oluştur           │
│       │                   (oynama_suresi, iade_edildi vb.)           │
│       ▼                                                             │
│  [ChromaDB Vektör Araması] → En alakalı yorumları bul               │
│       │                                                             │
│       ▼                                                             │
│  [Parent-Child Mimarisi] → Bulunan yorumların küme özetlerini çek   │
│       │                                                             │
│       ▼                                                             │
│  [Cevaplayıcı LLM] → Tüm bağlamı okuyup rapor üret                │
│       │                                                             │
│       ▼                                                             │
│  Geliştirici Ekibe Yanıt                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Kullanılan Teknikler ve Teknolojiler

| Kategori              | Teknoloji                                                       | Açıklama                                                          |
| --------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Veri Çekimi**       | Steam Web API, Requests                                         | 15.000+ olumsuz yorum otomatik olarak çekildi                     |
| **NLP Embedding**     | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) | Çok dilli yorumlar 384 boyutlu vektörlere dönüştürüldü            |
| **Boyut İndirgeme**   | UMAP                                                            | Yüksek boyutlu vektörler kümeleme için 5 boyuta indirgendi        |
| **Kümeleme**          | HDBSCAN                                                         | Yorumlar 59 anlam kümesine otomatik olarak gruplandı              |
| **Vektör Veritabanı** | ChromaDB                                                        | Küme özetleri ve tekil yorumlar Parent-Child mimarisinde saklandı |
| **LLM**               | Google Gemini 2.5 Flash                                         | Sorgu analizi (Trafik Polisi) ve rapor üretimi (Cevaplayıcı)      |
| **Web Arayüzü**       | Flask + HTML/CSS/JS                                             | Gerçek zamanlı sohbet tabanlı QA arayüzü                          |
| **API Yönetimi**      | Çoklu API Anahtarı Rotasyonu                                    | Kota aşımında otomatik anahtar değiştirme mekanizması             |

---

## 📂 Proje Yapısı

```
Crimson_Desert_NLP_Projesi/
│
├── 1_veri_cekimi.py          # Steam API'den yorum çeken scraper sınıfı
├── ReviewClusterer.py        # UMAP + HDBSCAN kümeleme modülü
├── app_core.py               # Temel vektör arama fonksiyonu (modüler çekirdek)
├── app.py                    # Konsol tabanlı RAG chatbot (terminal test aracı)
├── flask_app.py              # Flask web sunucusu (API + arayüz)
├── web_app.py                # Streamlit alternatif arayüz
│
├── templates/
│   └── index.html            # Flask arayüzünün HTML/CSS/JS şablonu
│
├── oyun_rag_veritabani/      # ChromaDB vektör veritabanı (kalıcı)
│
├── .env                      # API anahtarları (Git'e dahil değil)
├── .gitignore                # Hassas ve büyük dosyaları Git'ten hariç tutar
├── not.md                    # Geliştirme sürecindeki notlar
└── README.md                 # Bu dosya
```

---

## 🔑 Temel Özellikler

### 🚦 Trafik Polisi Mimarisi (Akıllı Sorgu Yönlendirme)

Kullanıcının doğal dildeki sorusu önce bir **yönlendirici LLM** tarafından
analiz edilir. Bu LLM, sorudan metadata filtrelerini çıkarır:

- `"10 saatten fazla oynayanlar"` → `{"oynama_suresi": {"$gt": 600}}`
- `"iade edenler"` → `{"iade_edildi": true}`

### 🧬 Parent-Child (Ebeveyn-Çocuk) Belge Mimarisi

Veritabanında iki tür belge vardır:

- **Çocuk (tekil_yorum):** Her bir oyuncu yorumu
- **Ebeveyn (kume_ozeti):** Her kümenin genel özeti

Arama sonuçlarında bulunan tekil yorumların ait olduğu kümenin özeti de otomatik
olarak bağlama eklenir. Bu sayede LLM hem detayı hem büyük resmi görür.

### 🔄 Otomatik API Anahtar Rotasyonu

Ücretsiz Gemini API kotası dolduğunda, sistem otomatik olarak bir sonraki
anahtara geçer. Bu sayede kesintisiz analiz mümkün olur.

### 📊 Küme Kritiklik Skorlaması

Her yorum için **yorum değeri skoru** (oynama süresi × faydalı bulunma), her
küme için **küme kritiklik skoru** hesaplanır. Bu skorlar, en acil sorunların
önceliklendirilmesini sağlar.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```
Python 3.10+
```

### 1. Repoyu Klonla

```bash
git clone https://github.com/tunakmynk/game-bug-tespiti-LLM.git
cd game-bug-tespiti-LLM
```

### 2. Bağımlılıkları Yükle

```bash
pip install flask chromadb google-generativeai sentence-transformers python-dotenv
```

### 3. API Anahtarlarını Ayarla

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
GEMINI_API_KEY_1=your_api_key_here
GEMINI_API_KEY_2=your_api_key_here
GEMINI_API_KEY_3=your_api_key_here
GEMINI_API_KEY_4=your_api_key_here
```

> API anahtarlarını [Google AI Studio](https://aistudio.google.com/apikey)
> üzerinden ücretsiz alabilirsiniz.

### 4. Uygulamayı Başlat

```bash
python flask_app.py
```

Tarayıcıda `http://localhost:5000` adresini açın.

---

## 🎮 Örnek Sorgular

| Sorgu                                             | Açıklama                                            |
| ------------------------------------------------- | --------------------------------------------------- |
| `Oyundaki genel optimizasyon sorunları nelerdir?` | Performans ile ilgili kümeleri ve yorumları getirir |
| `İade eden oyuncular en çok neyden şikâyetçi?`    | `iade_edildi: true` filtresi ile arama yapar        |
| `10 saatten fazla oynayıp iade edenler ne diyor?` | Birden fazla metadata filtresi (`$and`) kullanır    |
| `Savaş sistemi hakkında ne düşünüyorlar?`         | Savaş mekanikleri kümesini bulup özetler            |
| `En kritik hatalar hangileri?`                    | Küme kritiklik skoruna göre sıralar                 |

---

## 📈 Veri Pipeline Özeti

```
15.000+ ham yorum
    ↓ (Dil filtreleme + olumsuz yorum seçimi)
1.355 kullanılabilir İngilizce olumsuz yorum
    ↓ (Sentence Transformers embedding)
1.355 × 384 boyutlu vektör
    ↓ (UMAP boyut indirgeme: 384 → 5)
1.355 × 5 boyutlu vektör
    ↓ (HDBSCAN kümeleme)
59 anlam kümesi + gürültü yorumların temizlenmesi
    ↓ (LLM ile küme isimlendirme ve özetleme)
59 küme özeti + tekil yorumlar → ChromaDB
```

---

## 🛠️ Geliştirme Süreci

Bu proje aşamalı olarak geliştirilmiştir:

1. **Veri Toplama** — Steam Web API ile 15.000+ yorum çekildi
2. **Ön İşleme** — Dil filtreleme, temizleme ve olumsuz yorum seçimi
3. **Vektörleştirme** — Çok dilli Sentence Transformer modeli ile embedding
4. **Kümeleme** — UMAP + HDBSCAN ile unsupervised kümeleme (59 küme)
5. **Skorlama** — Yorum değeri ve küme kritiklik skorları hesaplandı
6. **Veritabanı** — ChromaDB'ye Parent-Child mimarisinde kayıt
7. **RAG Sistemi** — Trafik Polisi + Cevaplayıcı çift LLM mimarisi
8. **Web Arayüzü** — Flask tabanlı gerçek zamanlı sohbet arayüzü


## Görsel Arayüz

<img width="1902" height="912" alt="Crimson Desert - Geliştirici QA Asistanı Arayüz Görünümü ve Gösterim" src="https://github.com/user-attachments/assets/0e7ab7c1-267b-4887-b80d-29d98766fcf8" />
<img width="1896" height="907" alt="Crimson Desert - Geliştirici QA Asistanı Arayüz Görünümü ve Gösterim 2" src="https://github.com/user-attachments/assets/6744c166-aec8-4e96-9d35-1ea38ce00404" />


---

## 📄 Lisans

Bu proje eğitim ve portföy amaçlı geliştirilmiştir.
