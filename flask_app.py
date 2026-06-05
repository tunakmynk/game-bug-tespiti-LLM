import os
# Sessiz çökmeleri önleyen hayati ayarlar
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import json
import time

# .env dosyasındaki değişkenleri ortam değişkenlerine yükler
load_dotenv()

app = Flask(__name__)

# --- AYARLAR VE MODELLER ---
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

trafik_polisi_model, cevaplayici_model = modelleri_kur(aktif_anahtar_indeksi)

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
                aktif_anahtar_indeksi = (aktif_anahtar_indeksi + 1) % len(API_ANAHTARLARI)
                trafik_polisi_model, cevaplayici_model = modelleri_kur(aktif_anahtar_indeksi)
                model = trafik_polisi_model if json_mi else cevaplayici_model
                time.sleep(3)
            else:
                time.sleep(5)
    return {} if json_mi else "Üzgünüm, API limitleri aşıldı."

# --- VERİTABANI BAĞLANTISI (Uygulama başlarken bir kez yüklenir) ---
print("Veritabanı yükleniyor... Lütfen bekleyin.")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./oyun_rag_veritabani")
koleksiyon = client.get_collection(name="crimson_desert_hatalar", embedding_function=ef)
print("Veritabanı bağlandı! Web sunucusu hazır.")

def sorguyu_analiz_et(kullanici_sorusu):
    prompt = f"""
    Sen bir veritabanı yönlendiricisisin. Görevin kullanıcının sorusunu analiz edip bir ChromaDB arama filtresi oluşturmak.
    Kullanıcı bir süreden bahsediyorsa 'oynama_suresi' (dakika) filtresini kullan. (1 saat = 60 dakika).
    Kullanıcı iade etmekten bahsediyorsa 'iade_edildi' (true) filtresini kullan.
    
    Kullanıcı 'en kritik', 'en önemli' veya 'en kötü' gibi kelimeler kullanıyorsa, SADECE kümeleri getirme! Hem kritik kümeleri hem de dikkat çeken tekil yorumları beraber getirmek için KESİNLİKLE bir "$or" (VEYA) filtresi oluştur.
    Örnek Kritiklik Filtresi: {{"$or": [{{"kume_kritiklik_skoru": {{"$gt": 0.44}}}}, {{"yorum_degeri": {{"$gt": 0.70}}}}]}}
    
    ÖNEMLİ CHROMADB KURALI: Eğer birden fazla farklı tipte filtre kullanman gerekiyorsa, KESİNLİKLE bunları "$and" operatörü içine bir liste olarak koymalısın.
    
    Aşağıdaki JSON formatında yanıt ver:
    {{
        "arama_metni": "Vektör araması yapılacak temel kavramlar",
        "filtreler": {{}}
    }}

    Kullanıcı Sorusu: "{kullanici_sorusu}"
    """
    return guvenli_istek_yap(trafik_polisi_model, prompt, json_mi=True)

# --- WEB SUNUCUSU ROTALARI ---

@app.route("/")
def ana_sayfa():
    # Bu rota, birazdan oluşturacağımız HTML arayüzünü tarayıcıya gönderir
    return render_template("index.html")

@app.route("/api/soru-sor", methods=["POST"])
def soru_sor_api():
    veri = request.json
    kullanici_sorusu = veri.get("soru", "")
    
    # 1. Trafik Polisi
    analiz_sonucu = sorguyu_analiz_et(kullanici_sorusu)
    arama_metni = analiz_sonucu.get("arama_metni", "")
    filtreler = analiz_sonucu.get("filtreler", {})
    
    # 2. Veritabanı Araması
    if filtreler:
        sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5, where=filtreler)
    else:
        sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5)
        
    # Parent-Child (Küme) Mimarisi
    bulunan_kume_idleri = set()
    for meta in sonuclar['metadatas'][0]:
        k_id = meta.get('kume_id', -1)
        if meta.get('belge_tipi', '') == 'tekil_yorum' and k_id != -1:
            bulunan_kume_idleri.add(k_id)
            
    ekstra_ozet_metinleri = []
    for kume_id in bulunan_kume_idleri:
        ozet_sorgusu = koleksiyon.get(where={"$and": [{"belge_tipi": "kume_ozeti"}, {"kume_id": kume_id}]})
        if ozet_sorgusu['documents']:
            ekstra_ozet_metinleri.append(ozet_sorgusu['documents'][0])

    bulunan_metinler_listesi = []
    if ekstra_ozet_metinleri:
        for ozet in ekstra_ozet_metinleri:
            bulunan_metinler_listesi.append(f"[KÜME GENEL ÖZETİ]: {ozet}\n")
            
    for i in range(len(sonuclar['documents'][0])):
        metin = sonuclar['documents'][0][i]
        metadata = sonuclar['metadatas'][0][i]
        bulunan_metinler_listesi.append(f"[Metadata: {metadata}]\nYorum: {metin}\n")

    bulunan_metinler_birlestirilmis = "\n".join(bulunan_metinler_listesi)
    
    # 3. Asistan Cevabı
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
    
    # Tüm verileri web arayüzüne geri gönder
    return jsonify({
        "cevap": final_cevap,
        "arama_metni": arama_metni,
        "filtreler": filtreler,
        "kaynaklar": bulunan_metinler_birlestirilmis
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)