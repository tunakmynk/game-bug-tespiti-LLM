import requests
import time
import pandas as pd

class SteamScraper:
    def __init__(self, app_id="3321460", target_reviews=15000):
        self.app_id = app_id
        self.target_reviews = target_reviews
        self.temel_url = f"https://store.steampowered.com/appreviews/{self.app_id}"
        self.kimlik_karti = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        }

    def scrape_reviews(self):
        cursor = "*"
        tum_yorumlar = []
        sayfa_sayaci = 0
        hedef_sayfa = self.target_reviews // 100

        print(f"Toplam {hedef_sayfa} yepyeni sayfa çekilecek...")

        while sayfa_sayaci < hedef_sayfa:
            parametreler = {
                "json": 1,
                "cursor": cursor,
                "language": "english", # İngilizce veri çekimi
                "num_per_page": 100 ,   # Tek seferde maksimum yorum
                "review_type": "negative" # Yeni filtremiz: Sadece olumsuz yorumlar
            }
            
            try:
                response = requests.get(self.temel_url, headers=self.kimlik_karti, params=parametreler, timeout=15)
                veri = response.json()
                
                # --- 1. GÜVENLİK KİLİDİ: Sayfada yorum kalmadıysa ---
                if 'reviews' not in veri or veri['reviews'] == []:
                    print("Sayfada başka yorum kalmadı. Döngü tamamlanıyor...")
                    break
                    
                # --- 2. GÜVENLİK KİLİDİ: İmleç takıldıysa (Sonsuz döngü engeli) ---
                if veri['cursor'] == cursor:
                    print("Steam API imleci takıldı (aynı sayfayı döndürüyor). Döngü kırılıyor...")
                    break
                
                # Güvenlikten başarıyla geçen yorumları ana listeye ekle
                tum_yorumlar.extend(veri['reviews'])
                print(f"{sayfa_sayaci+1}. sayfa çekildi!")
                
                # Sonraki sayfa için imleci güncelle
                cursor = veri['cursor']
                sayfa_sayaci += 1
                time.sleep(2) # İstekler arası bekleme süresi

            except Exception as e:
                print(f"Bir hata oluştu {e}. 5 saniye bekleyip tekrar deniyoruz...")
                time.sleep(5)
        
        df = pd.DataFrame(tum_yorumlar)
        return df

# Kullanım Örneği:
# scraper = SteamScraper(target_reviews=15000)
# df_reviews = scraper.scrape_reviews()

# 1. Sınıfı başlat (15.000 yorum hedefiyle)
scraper = SteamScraper(target_reviews=15000)

# 2. Sınıfın içindeki metodu çağırarak veriyi çek
df_reviews = scraper.scrape_reviews()

# 3. Çekilen veriyi bilgisayarına CSV dosyası olarak kaydet
df_reviews.to_csv("crimson_desert_ingilizce_olumsuz_yorumlari.csv", index=False)
print("İşlem tamamlandı, crimson_desert_ingilizce_olumsuz_yorumlari_deneme.csv dosyası kaydedildi!")