import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import json
import time
import os
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri ortam değişkenlerine yükler
load_dotenv()

# --- 1. AYARLAR VE ÇOKLU API ANAHTARI YÖNETİMİ ---

# API anahtarları artık .env dosyasından okunuyor (güvenlik için)
API_ANAHTARLARI = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]
aktif_anahtar_indeksi = 0

def modelleri_kur(indeks):
    genai.configure(api_key=API_ANAHTARLARI[indeks])
    trafik_polisi = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
    cevaplayici = genai.GenerativeModel('gemini-2.5-flash')
    return trafik_polisi, cevaplayici

# İlk modelleri başlat
trafik_polisi_model, cevaplayici_model = modelleri_kur(aktif_anahtar_indeksi)

# Hata yakalama ve anahtar değiştirme fonksiyonu (Sorguları güvenle çalıştırmak için)
def guvenli_istek_yap(model, prompt, json_mi=False):
    global aktif_anahtar_indeksi, trafik_polisi_model, cevaplayici_model
    max_deneme = len(API_ANAHTARLARI) * 2
    
    for deneme in range(max_deneme):
        try:
            cevap = model.generate_content(prompt)
            return json.loads(cevap.text) if json_mi else cevap.text
        except Exception as e:
            hata_mesaji = str(e).lower()
            if "429" in hata_mesaji or "quota" in hata_mesaji or "exhausted" in hata_mesaji:
                print(f"\n[!] Limit Doldu! {aktif_anahtar_indeksi + 1}. Anahtar değiştiriliyor...")
                aktif_anahtar_indeksi = (aktif_anahtar_indeksi + 1) % len(API_ANAHTARLARI)
                trafik_polisi_model, cevaplayici_model = modelleri_kur(aktif_anahtar_indeksi)
                # Yeni anahtara geçilen modeli güncelle
                model = trafik_polisi_model if json_mi else cevaplayici_model
                time.sleep(3)
            else:
                print(f"Beklenmeyen Hata: {e}")
                time.sleep(5)
    return {} if json_mi else "Üzgünüm, API limitleri aşıldı ve yanıt üretilemedi."

# --- 2. VERİTABANI BAĞLANTISI ---

print("Veritabanı yükleniyor...")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./oyun_rag_veritabani")
koleksiyon = client.get_collection(name="crimson_desert_hatalar", embedding_function=ef)
print(f"Veritabanı hazır! Toplam belge: {koleksiyon.count()}")


# --- 3. TRAFİK POLİSİ (SORGUNUN ANALİZİ) ---

def sorguyu_analiz_et(kullanici_sorusu):
    prompt = f"""
    Sen bir veritabanı yönlendiricisisin. Görevin kullanıcının sorusunu analiz edip bir ChromaDB arama filtresi oluşturmak.
    Kullanıcı bir süreden bahsediyorsa 'oynama_suresi' (dakika) filtresini kullan. (1 saat = 60 dakika).
    Kullanıcı iade etmekten bahsediyorsa 'iade_edildi' (true) filtresini kullan.
    
    ÖNEMLİ CHROMADB KURALI: Eğer birden fazla filtre kullanman gerekiyorsa, KESİNLİKLE bunları "$and" operatörü içine bir liste olarak koymalısın.
    - Tek filtre örneği: {{"oynama_suresi": {{"$gt": 600}}}}
    - Çoklu filtre örneği: {{"$and": [{{"oynama_suresi": {{"$gt": 600}}}}, {{"iade_edildi": true}}]}}
    
    Aşağıdaki JSON formatında yanıt ver:
    {{
        "arama_metni": "Vektör araması yapılacak temel kavramlar",
        "filtreler": {{}}
    }}
    
    Kullanıcı Sorusu: "{kullanici_sorusu}"
    """
    return guvenli_istek_yap(trafik_polisi_model, prompt, json_mi=True)


# --- 4. ARAMA VE FINAL CEVAP ÜRETİMİ ---

def bot_soru_sor(kullanici_sorusu):
    print(f"\n" + "="*70)
    print(f"YÖNETİCİ SORGUSU: {kullanici_sorusu}")
    print("="*70)
    print("Trafik Polisi analiz ediyor...")
    
    analiz_sonucu = sorguyu_analiz_et(kullanici_sorusu)
    arama_metni = analiz_sonucu.get("arama_metni", "")
    filtreler = analiz_sonucu.get("filtreler", {})
    
    print(f"-> Çıkarılan Filtreler: {filtreler}")
    print(f"-> Arama Terimi: {arama_metni}")
    
    # 1. ADIM: Vektör Veritabanında Arama Yap (Çocukları / Tekil Yorumları Bul)
    if filtreler:
        sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5, where=filtreler)
    else:
        sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5)
        
    
    # --- YENİ: 2. ADIM (PARENT-CHILD / KÖKÜ BULMA MİMARİSİ) ---
    bulunan_kume_idleri = set()
    
    # Bulunan yorumların hangi kümelere ait olduğunu tespit et
    for meta in sonuclar['metadatas'][0]:
        k_id = meta.get('kume_id', -1)
        belge_tipi = meta.get('belge_tipi', '')
        if belge_tipi == 'tekil_yorum' and k_id != -1:
            bulunan_kume_idleri.add(k_id)
            
    # Tespit edilen kümelerin "Genel Özetlerini" (Ebeveynleri) veritabanından direkt ID ile çek!
    ekstra_ozet_metinleri = []
    for kume_id in bulunan_kume_idleri:
        ozet_sorgusu = koleksiyon.get(where={"$and": [{"belge_tipi": "kume_ozeti"}, {"kume_id": kume_id}]})
        if ozet_sorgusu['documents']:
            ekstra_ozet_metinleri.append(ozet_sorgusu['documents'][0])

            
    # --- ŞEFFAFLIK ADIMI: HANGİ VERİLER BULUNDU? ---
    print("\n🔍 VERİTABANINDAN ÇEKİLEN KAYNAKLAR (LLM'in Okuduğu Veriler):")
    bulunan_metinler_listesi = []
    
    # Önce o yorumların büyük resmini (Özetleri) bağlama ekleyelim
    if ekstra_ozet_metinleri:
        print("\n--- BÜYÜK RESİM (KÜME ÖZETLERİ) ---")
        for ozet in ekstra_ozet_metinleri:
            print(f"-> {ozet[:150]}...")
            bulunan_metinler_listesi.append(f"[KÜME GENEL ÖZETİ]: {ozet}\n")
    
    print("\n--- SPESİFİK YORUMLAR ---")
    for i in range(len(sonuclar['documents'][0])):
        metin = sonuclar['documents'][0][i]
        metadata = sonuclar['metadatas'][0][i]
        
        print(f"-> Metadata: {metadata}")
        print(f"-> Metin: {metin[:150]}...\n")
        
        kaynak_bilgisi = f"[Metadata: {metadata}]\nYorum: {metin}\n"
        bulunan_metinler_listesi.append(kaynak_bilgisi)

    bulunan_metinler_birlestirilmis = "\n".join(bulunan_metinler_listesi)
    
    
    # --- CEVAPLAYICI LLM (ASİSTAN PERSONASI) ---
    print("\nCevaplayıcı LLM İç Raporu hazırlıyor...")
    final_prompt = f"""
    Sen geliştirici ekibe içeriden yardım eden bir teknik destek asistanısın.
    Amacın, veritabanından çekilen aşağıdaki oyuncu geri bildirimlerini okuyup, ekibin neleri düzeltmesi gerektiğini özetlemektir.
    Eğer bağlamda [KÜME GENEL ÖZETİ] varsa, önce o büyük resmi belirterek konuya gir.
    
    Kurallar:
    1. Çok uzun destanlar veya raporlar yazma. Doğrudan sorunları söyle.
    2. Aşırı teknik jargona boğma, sade ve net ol.
    3. 'Değerli oyuncularımız' gibi müşteri hizmetleri ağzı veya 'Kime: Yönetim' gibi resmi şirket formatları KESİNLİKLE KULLANMA.
    4. Sadece kısa bir giriş yap, sorunları maddeler halinde özetle ve bitir.
    
    İncelemen Gereken Veritabanı Kayıtları:
    {bulunan_metinler_birlestirilmis}
    
    Ekibin Sorusu: {kullanici_sorusu}
    """
    
    final_cevap = guvenli_istek_yap(cevaplayici_model, final_prompt)
    print("\n" + "🟢 İÇ RAPOR (BOT YANITI):")
    print(final_cevap)
    print("="*70 + "\n")

# --- 5. TEST SORGULARI ---
if __name__ == "__main__":
    # Test 1
    bot_soru_sor("Oyundaki genel optimizasyon durumu ve performans sorunları nelerdir?")
    
    # İki sorgu arasına nefes payı (Ücretsiz API'nin kızmaması için)
    time.sleep(5) 
    
    # Test 2
    bot_soru_sor("Oyunu 10 saatten (600 dakika) fazla oynayan ve oyunu iade eden oyuncular en çok neden şikayetçi?")