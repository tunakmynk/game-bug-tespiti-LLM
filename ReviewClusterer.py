import umap
import hdbscan
import numpy as np

class ReviewClusterer:
    def __init__(self, n_neighbors=30, n_components=5, min_cluster_size=25):
        """
        Modüler kümeleme sınıfı.
        n_components: UMAP'in vektörleri düşüreceği hedef boyut sayısı.
        min_cluster_size: HDBSCAN'in bir grubu 'küme' sayması için gereken minimum yorum sayısı.
        """
        # UMAP Yapılandırması (Boyut İndirgeme)
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine', # Metin vektörleri için en uygun mesafe ölçümü
            random_state=42  # Sonuçların tekrarlanabilir olması için
        )
        
        # HDBSCAN Yapılandırması (Kümeleme)
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean', # UMAP sonrası uzayda öklid kullanmak standarttır
            cluster_selection_method='eom' # Excess of Mass metodu
        )

    def fit_predict(self, embeddings):
        """
        Vektörleri alır, boyutlarını düşürür ve küme etiketlerini döndürür.
        """
        print("1. Boyut indirgeme (UMAP) uygulanıyor...")
        reduced_embeddings = self.umap_model.fit_transform(embeddings)
        
        print("2. Kümeleme (HDBSCAN) uygulanıyor...")
        cluster_labels = self.hdbscan_model.fit_predict(reduced_embeddings)
        
        return cluster_labels, reduced_embeddings