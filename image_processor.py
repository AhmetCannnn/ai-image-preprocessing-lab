import numpy as np

class ImageProcessor:
    """
    Görüntü İşleme Sınıfı
    ---------------------
    Bu sınıf, görüntü işleme için temel fonksiyonları içerir.
    Numpy kütüphanesini kullanarak sıfırdan geliştirilmiş işlemler sunar.
    
    Özellikler:
    - Tüm işlemler numpy kütüphanesi ile sıfırdan yazılmıştır
    - Harici görüntü işleme kütüphaneleri kullanılmamıştır (OpenCV, PIL, vb.)
    - Görüntüler numpy dizileri (ndarray) olarak temsil edilir
    - Pikseller 0-255 arası değerlere sahiptir (8-bit)
    - RGB görüntüler (h, w, 3) şeklinde, gri tonlamalı görüntüler (h, w) şeklindedir
    
    Sağlanan işlemler:
    - Temel dönüşümler: Gri tonlama, ikili (binary) dönüşüm, RGB->HSV
    - Geometrik işlemler: Döndürme, kırpma, yeniden boyutlandırma
    - Histogram işlemleri: Histogram eşitleme, kontrast ayarı
    - Aritmetik işlemler: Görüntü toplama, çarpma, parlaklık ayarı
    - Filtreleme: Gaussian bulanıklaştırma, ortalama filtre, medyan filtre
    - Kenar algılama: Sobel kenar algılama
    - Gürültü ekleme: Tuz & biber gürültüsü
    - Morfolojik işlemler: Genişletme, aşındırma, açma, kapama
    
    Kullanım:
    ```python
    # ImageProcessor sınıfını içe aktarma
    from models.image_processor import ImageProcessor
    
    # Sınıf örneği oluşturma
    processor = ImageProcessor()
    
    # Görüntü yükleme
    import numpy as np
    from PIL import Image
    image = np.array(Image.open('goruntu.jpg'))
    
    # İşlemleri uygulama
    gray_image = processor.to_grayscale(image)
    binary_image = processor.binary_conversion(image, threshold=150)
    blurred_image = processor.blur(image, kernel_size=5)
    ```
    """
    def __init__(self):
        """
        ImageProcessor sınıfının yapıcı metodu.
        """
        pass
    
    def to_grayscale(self, image):
        """
        RGB görüntüyü gri tonlamaya çevirir.
        
        Gri tonlama dönüşümü, R, G ve B kanallarının ağırlıklı toplamını alarak yapılır.
        Kullanılan ağırlıklar: R=0.299, G=0.587, B=0.114 (ITU-R BT.601 standardı)
        
        İşlem süreci:
        1. Görüntünün 3 kanallı (RGB) olup olmadığını kontrol eder
        2. RGB ise, her pikselin R, G ve B değerlerini belirli ağırlıklarla çarpar
        3. Bu ağırlıklı değerlerin toplamını alarak gri tonlama değerini hesaplar
        
        Not: İnsan gözü yeşil renge daha duyarlıdır, bu nedenle yeşil kanalın ağırlığı 
             (0.587) daha yüksektir. Bu ağırlıklar ITU-R BT.601 standardına dayanır ve 
             insan görsel algısına uygun bir gri tonlama elde edilmesini sağlar.
        
        Args:
            image (numpy.ndarray): RGB formatında görüntü
            
        Returns:
            numpy.ndarray: Gri tonlamalı görüntü (8-bit)
        """
        if len(image.shape) == 3:
            # RGB ağırlıkları: R=0.299, G=0.587, B=0.114
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return image
    
    def binary_conversion(self, image, threshold=127):
        """
        İkili (siyah-beyaz) görüntüye çevirir.
        
        Önce görüntü gri tonlamaya çevrilir, sonra belirtilen eşik değerine göre
        piksel değerleri 0 (siyah) veya 255 (beyaz) olarak atanır.
        
        İşlem süreci:
        1. Görüntüyü gri tonlamaya çevirir (eğer zaten gri değilse)
        2. Eşik değerinden (threshold) düşük olan pikselleri siyah (0) yapar
        3. Eşik değerinden yüksek olan pikselleri beyaz (255) yapar
        
        Not: Eşik değeri varsayılan olarak 127'dir (0-255 aralığının ortası).
             Bu değer, görüntünün özelliklerine göre ayarlanabilir. Karanlık
             bir görüntüde daha düşük eşik değeri, aydınlık bir görüntüde ise
             daha yüksek eşik değeri kullanmak daha iyi sonuç verebilir.
        
        Args:
            image (numpy.ndarray): Dönüştürülecek görüntü
            threshold (int): Eşik değeri (0-255 arası)
            
        Returns:
            numpy.ndarray: İkili (binary) görüntü
        """
        gray = self.to_grayscale(image)
        binary = np.zeros_like(gray)
        binary[gray > threshold] = 255
        return binary
    
    def rotate_image(self, image, angle):
        """
        Görüntüyü belirtilen açıda döndürür.
        
        Rotasyon matrisi oluşturularak her piksel için dönüşüm uygulanır.
        Bilinear interpolasyon kullanılmaz, en yakın komşu yöntemi uygulanır.
        
        Args:
            image (numpy.ndarray): Döndürülecek görüntü
            angle (float): Döndürme açısı (derece cinsinden)
            
        Returns:
            numpy.ndarray: Döndürülmüş görüntü
        """
        # Radyana çevir
        angle = np.deg2rad(angle)
        
        # Rotasyon matrisi
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        h, w = image.shape[:2]
        
        # Yeni boyutları hesapla
        new_h = int(abs(h * cos_a) + abs(w * sin_a))
        new_w = int(abs(w * cos_a) + abs(h * sin_a))
        
        # Yeni görüntü oluştur
        rotated = np.zeros((new_h, new_w, 3) if len(image.shape) == 3 else (new_h, new_w), dtype=np.uint8)
        
        # Merkez noktaları
        center_y, center_x = h // 2, w // 2
        new_center_y, new_center_x = new_h // 2, new_w // 2
        
        # Her piksel için dönüşüm uygula
        for y in range(new_h):
            for x in range(new_w):
                # Orijinal koordinatları bul
                orig_x = cos_a * (x - new_center_x) - sin_a * (y - new_center_y) + center_x
                orig_y = sin_a * (x - new_center_x) + cos_a * (y - new_center_y) + center_y
                
                if 0 <= orig_x < w and 0 <= orig_y < h:
                    rotated[y, x] = image[int(orig_y), int(orig_x)]
        
        return rotated
    
    def crop_image(self, image, x1, y1, x2, y2):
        """
        Görüntüyü belirtilen koordinatlardan kırpar.
        
        Args:
            image (numpy.ndarray): Kırpılacak görüntü
            x1 (int): Sol üst köşe x koordinatı
            y1 (int): Sol üst köşe y koordinatı
            x2 (int): Sağ alt köşe x koordinatı
            y2 (int): Sağ alt köşe y koordinatı
            
        Returns:
            numpy.ndarray: Kırpılmış görüntü
        """
        return image[y1:y2, x1:x2]
    
    def resize_image(self, image, scale_factor):
        """
        Görüntüyü yakınlaştırır/uzaklaştırır.
        
        En yakın komşu interpolasyon metodunu kullanarak görüntüyü yeniden
        boyutlandırır. Scale_factor > 1 ise büyütme, < 1 ise küçültme yapılır.
        
        Args:
            image (numpy.ndarray): Yeniden boyutlandırılacak görüntü
            scale_factor (float): Ölçekleme faktörü
            
        Returns:
            numpy.ndarray: Yeniden boyutlandırılmış görüntü
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        resized = np.zeros((new_h, new_w, 3) if len(image.shape) == 3 else (new_h, new_w), dtype=np.uint8)
        
        # Her piksel için en yakın komşu interpolasyonu
        for y in range(new_h):
            for x in range(new_w):
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                if orig_x < w and orig_y < h:
                    resized[y, x] = image[orig_y, orig_x]
        
        return resized
    
    def convert_grayscale_to_rgb(self, image):
        """
        Gri tonlamalı bir görüntüyü 3 kanallı RGB forma dönüştürür.
        
        Args:
            image (numpy.ndarray): Gri tonlamalı veya RGB görüntü
            
        Returns:
            numpy.ndarray: RGB görüntü (h, w, 3)
        """
        if len(image.shape) == 2:
            return np.stack([image, image, image], axis=-1).astype(np.uint8)
        return image
    
    def convert_rgb_to_grayscale(self, image):
        """
        RGB görüntüyü tek kanallı gri tonlamalı formata dönüştürür.
        
        Args:
            image (numpy.ndarray): RGB veya gri görüntü
            
        Returns:
            numpy.ndarray: Gri tonlamalı görüntü (h, w)
        """
        if len(image.shape) == 3:
            return self.to_grayscale(image)
        return image
    
    def resize_image_numpy(self, image, new_height, new_width):
        """
        Görüntüyü verilen yükseklik ve genişliğe yeniden boyutlandırır.
        
        En yakın komşu interpolasyonunu kullanır ve hem gri hem RGB görüntülerle çalışır.
        
        Args:
            image (numpy.ndarray): Yeniden boyutlandırılacak görüntü
            new_height (int): Hedef yükseklik
            new_width (int): Hedef genişlik
            
        Returns:
            numpy.ndarray: Yeniden boyutlandırılmış görüntü
        """
        if new_height <= 0 or new_width <= 0:
            return image
        
        h, w = image.shape[:2]
        resized = np.zeros((new_height, new_width, 3) if len(image.shape) == 3 else (new_height, new_width), dtype=np.uint8)
        
        scale_y = h / new_height
        scale_x = w / new_width
        
        for y in range(new_height):
            for x in range(new_width):
                orig_y = int(y * scale_y)
                orig_x = int(x * scale_x)
                
                if orig_y >= h:
                    orig_y = h - 1
                if orig_x >= w:
                    orig_x = w - 1
                
                resized[y, x] = image[orig_y, orig_x]
        
        return resized
    
    def rgb_to_hsv(self, image):
        """
        RGB'den HSV'ye dönüşüm.
        
        HSV renk uzayı: Hue (Renk tonu), Saturation (Doygunluk), Value (Parlaklık)
        Dönüşüm algoritması, RGB renk uzayından HSV'ye matematiksel dönüşüm yapar.
        
        Args:
            image (numpy.ndarray): RGB formatında görüntü
            
        Returns:
            numpy.ndarray: HSV formatında görüntü
        """
        if len(image.shape) != 3:
            return image
            
        # 0-1 aralığına normalize et
        r, g, b = image[..., 0] / 255.0, image[..., 1] / 255.0, image[..., 2] / 255.0
        
        cmax = np.max([r, g, b], axis=0)
        cmin = np.min([r, g, b], axis=0)
        diff = cmax - cmin
        
        # Hue hesapla
        h = np.zeros_like(cmax)
        mask = diff != 0
        mask_r = (cmax == r) & mask
        mask_g = (cmax == g) & mask
        mask_b = (cmax == b) & mask
        
        h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r] % 6)
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)
        
        # Saturation hesapla
        s = np.zeros_like(cmax)
        mask = cmax != 0
        s[mask] = diff[mask] / cmax[mask]
        
        # Value
        v = cmax
        
        return np.stack([h, s * 255, v * 255], axis=-1).astype(np.uint8)
    
    def histogram_equalization(self, image):
        """
        Histogram eşitleme.
        
        Bu işlem, görüntünün kontrastını artırır ve detayların daha iyi görünmesini sağlar.
        Kümülatif dağılım fonksiyonu (CDF) kullanılarak piksel değerleri yeniden dağıtılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            numpy.ndarray: Histogram eşitlenmiş görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        # Histogram hesapla
        hist = np.zeros(256)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1
                
        # Kümülatif dağılım fonksiyonu
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Yeni görüntü oluştur
        equalized = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                equalized[i, j] = cdf_normalized[image[i, j]]
                
        return equalized
    
    def add_images(self, image1, image2):
        """
        İki görüntüyü toplar.
        
        Pikselleri belirli ağırlıklarla toplar ve değerleri 0-255 aralığında keser.
        Görsel farkı daha belirgin hale getirmek için ağırlıklı toplama kullanılır.
        Görüntüler farklı boyutlarda ise, ikinci görüntü birinci görüntü boyutuna otomatik 
        olarak yeniden boyutlandırılır. Tamamı NumPy kullanılarak gerçekleştirilir.
        
        İşlem süreci:
        1. İkinci görüntüyü ilk görüntü boyutuna otomatik yeniden boyutlandırır (gerekirse)
        2. İki görüntünün piksellerini ağırlıklı olarak toplar (img1 * 0.7 + img2 * 0.3)
        3. Kontrast artırmak için bir faktör (1.3) ile çarpar
        4. Değerleri 0-255 aralığında sınırlar
        
        Not: Ağırlıkların farklı olması ve kontrast artırma faktörü, sonucun
             daha belirgin görünmesini sağlar. İki görüntü arasındaki fark
             böylece çıplak gözle daha kolay anlaşılabilir.
        
        Args:
            image1 (numpy.ndarray): İlk görüntü
            image2 (numpy.ndarray): İkinci görüntü
            
        Returns:
            numpy.ndarray: Toplama sonucu görüntü
        """
        # Görüntülerin boş olup olmadığını kontrol et
        if image1 is None:
            return image2
        if image2 is None:
            return image1
        if image1 is None and image2 is None:
            return None
            
        # İkinci görüntüyü ilk görüntü boyutuna getir
        if image1.shape != image2.shape:
            # Boyutları al
            height, width = image1.shape[:2]
            
            # Kanal sayısını kontrol et
            if len(image2.shape) == 2 and len(image1.shape) == 3:
                # Gri tonlama -> RGB dönüşümü
                image2 = self.convert_grayscale_to_rgb(image2)
            elif len(image2.shape) == 3 and len(image1.shape) == 2:
                # RGB -> Gri tonlama dönüşümü
                image2 = self.convert_rgb_to_grayscale(image2)
                
            # Boyutlandırma yap
            image2 = self.resize_image_numpy(image2, height, width)
            
        # Ağırlıklı toplama (0.7 ve 0.3 ağırlıkları)
        # Sonucu daha belirgin hale getirmek için kontrast artırma faktörü ekledik
        weight1, weight2 = 0.7, 0.3
        contrast_factor = 1.3
        
        result = np.clip(
            (image1.astype(np.float32) * weight1 + 
             image2.astype(np.float32) * weight2) * contrast_factor, 
            0, 255
        ).astype(np.uint8)
        
        return result
    
    def multiply_images(self, image1, image2):
        """
        İki görüntüyü çarpar.
        
        Pikselleri birebir çarpar ve sonucu daha belirgin hale getirmek için
        kontrast artırma faktörü uygular. Değerleri 0-255 aralığında keser.
        Görüntüler farklı boyutlarda ise, ikinci görüntü birinci görüntü boyutuna otomatik 
        olarak yeniden boyutlandırılır. Tamamı NumPy kullanılarak gerçekleştirilir.
        
        İşlem süreci:
        1. İkinci görüntüyü ilk görüntü boyutuna otomatik yeniden boyutlandırır (gerekirse)
        2. Görüntüleri 0-1 aralığına normalize eder (255'e bölerek)
        3. Normalize edilmiş değerleri birebir çarpar
        4. Kontrast artırmak için bir faktör (1.8) ile çarpar
        5. Sonucu tekrar 0-255 aralığına ölçekler
        6. Değerleri 0-255 aralığında sınırlar
        
        Not: İki görüntünün çarpımı genellikle karanlık sonuçlar üretir,
             çünkü 0-1 arasındaki değerlerin çarpımı daha küçük değerler verir.
             Bu nedenle kontrast faktörü (1.8) eklenerek sonucun daha belirgin
             olması sağlanır. Böylece işlemin etkisi gözle görülür hale gelir.
        
        Args:
            image1 (numpy.ndarray): İlk görüntü
            image2 (numpy.ndarray): İkinci görüntü
            
        Returns:
            numpy.ndarray: Çarpma sonucu görüntü
        """
        # Görüntülerin boş olup olmadığını kontrol et
        if image1 is None:
            return image2
        if image2 is None:
            return image1
        if image1 is None and image2 is None:
            return None
            
        # İkinci görüntüyü ilk görüntü boyutuna getir
        if image1.shape != image2.shape:
            # Boyutları al
            height, width = image1.shape[:2]
            
            # Kanal sayısını kontrol et
            if len(image2.shape) == 2 and len(image1.shape) == 3:
                # Gri tonlama -> RGB dönüşümü
                image2 = self.convert_grayscale_to_rgb(image2)
            elif len(image2.shape) == 3 and len(image1.shape) == 2:
                # RGB -> Gri tonlama dönüşümü
                image2 = self.convert_rgb_to_grayscale(image2)
                
            # Boyutlandırma yap
            image2 = self.resize_image_numpy(image2, height, width)
        
        # Görüntüleri normalize et (0-1 aralığı)
        img1_norm = image1.astype(np.float32) / 255.0
        img2_norm = image2.astype(np.float32) / 255.0
        
        # Çarpma işlemi
        # Kontrast artırma faktörü ekleyerek sonucu daha belirgin hale getiriyoruz
        # Çarpım sonucu karanlık olduğu için, daha büyük bir faktör (1.8) kullanıyoruz
        contrast_factor = 1.8
        result = np.clip(img1_norm * img2_norm * contrast_factor * 255.0, 0, 255).astype(np.uint8)
        
        return result
    
    def adjust_brightness(self, image, factor):
        """
        Parlaklık ayarı.
        
        Her piksele factor değeri eklenerek parlaklık ayarlanır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            factor (int): Parlaklık ayar faktörü (-255 ile 255 arası)
            
        Returns:
            numpy.ndarray: Parlaklığı ayarlanmış görüntü
        """
        return np.clip(image.astype(np.int16) + factor, 0, 255).astype(np.uint8)
    
    def gaussian_kernel(self, size, sigma):
        """
        Gaussian kernel oluşturur.
        
        Gaussian filtre için 2D Gaussian dağılım kerneli oluşturur.
        
        Args:
            size (int): Kernelin boyutu (tek sayı olmalı)
            sigma (float): Gaussian dağılımın standart sapması
            
        Returns:
            numpy.ndarray: Normalize edilmiş Gaussian kernel
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                
        return kernel / kernel.sum()
    
    def convolve(self, image, kernel):
        """
        Konvolüsyon işlemi uygular.
        
        Görüntüye verilen kernel ile konvolüsyon uygular. Bu işlem,
        gürültü giderme, kenar bulma, bulanıklaştırma gibi işlemlerin temelini oluşturur.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            kernel (numpy.ndarray): Konvolüsyon kerneli
            
        Returns:
            numpy.ndarray: Konvolüsyon sonucu görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # Padding uygula
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Çıktı görüntüsü
        output = np.zeros_like(image)
        
        # Konvolüsyon
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
                
        return output.astype(np.uint8)
    
    def adaptive_threshold(self, image, window_size=11, c=2):
        """
        Adaptif eşikleme.
        
        Her piksel için yerel bölgenin ortalama değerine göre eşik değeri belirler.
        Bu, değişken aydınlatma koşullarında daha iyi sonuç verir.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            window_size (int): Yerel bölge pencere boyutu (tek sayı olmalı)
            c (int): Ortalamadan çıkarılacak sabit değer
            
        Returns:
            numpy.ndarray: Adaptif eşiklenmiş ikili görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        h, w = image.shape
        pad = window_size // 2
        
        # Padding uygula
        padded = np.pad(image, pad, mode='edge')
        
        # Çıktı görüntüsü
        output = np.zeros_like(image)
        
        # Her piksel için yerel ortalama hesapla
        for i in range(h):
            for j in range(w):
                window = padded[i:i+window_size, j:j+window_size]
                threshold = np.mean(window) - c
                output[i, j] = 255 if image[i, j] > threshold else 0
                
        return output
    
    def sobel_edge_detection(self, image):
        """
        Sobel kenar bulma.
        
        Sobel operatörü kullanarak görüntüdeki kenarları tespit eder.
        Yatay ve dikey gradyanları hesaplayarak, kenar büyüklüğünü bulur.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            
        Returns:
            numpy.ndarray: Kenarları tespit edilmiş görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        # Sobel kernelleri
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Gradyanları hesapla
        grad_x = self.convolve(image, kernel_x)
        grad_y = self.convolve(image, kernel_y)
        
        # Gradyan büyüklüğü
        magnitude = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)
        
        return np.clip(magnitude, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, prob=0.02):
        """
        Tuz & biber gürültüsü ekler.
        
        Rastgele piksellere beyaz (tuz) ve siyah (biber) gürültü ekler.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            prob (float): Gürültü olasılığı (0-1 arası)
            
        Returns:
            numpy.ndarray: Gürültü eklenmiş görüntü
        """
        noisy = np.copy(image)
        h, w = image.shape[:2]
        
        # Tuz gürültüsü
        salt = np.random.random((h, w)) < prob
        noisy[salt] = 255
        
        # Biber gürültüsü
        pepper = np.random.random((h, w)) < prob
        noisy[pepper] = 0
        
        return noisy
    
    def mean_filter(self, image, kernel_size=3):
        """
        Ortalama filtre.
        
        Her pikseli çevresindeki piksellerin ortalaması ile değiştirir.
        Gürültü azaltma için kullanılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            kernel_size (int): Filtre pencere boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Filtrelenmiş görüntü
        """
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return self.convolve(image, kernel)
    
    def median_filter(self, image, kernel_size=3):
        """
        Medyan filtre.
        
        Her pikseli çevresindeki piksellerin medyanı ile değiştirir.
        Tuz ve biber gürültüsünü gidermede çok etkilidir.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            kernel_size (int): Filtre pencere boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Filtrelenmiş görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        h, w = image.shape
        pad = kernel_size // 2
        
        # Padding uygula
        padded = np.pad(image, pad, mode='edge')
        
        # Çıktı görüntüsü
        output = np.zeros_like(image)
        
        # Medyan hesapla
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(window)
                
        return output.astype(np.uint8)
    
    def blur(self, image, kernel_size=5):
        """
        Bulanıklaştırma işlemi.
        
        Ortalama filtre kullanan basit bir bulanıklaştırma işlemi uygular.
        Tüm kernel değerleri eşit ağırlıklıdır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü
            kernel_size (int): Filtre pencere boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Bulanıklaştırılmış görüntü
        """
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return self.convolve(image, kernel)
    
    def dilate(self, image, kernel_size=3):
        """
        Genişletme (Dilation) morfolojik işlemi.
        
        Beyaz nesneleri (255 değerli pikseller) genişleten bir işlemdir.
        Her piksel için komşuluk içindeki maksimum değer alınır.
        Nesneleri büyütmek, küçük boşlukları doldurmak için kullanılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü (tercihen ikili görüntü)
            kernel_size (int): Yapısal eleman boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Genişletilmiş görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        h, w = image.shape
        pad = kernel_size // 2
        
        # Padding uygula
        padded = np.pad(image, pad, mode='edge')
        
        # Çıktı görüntüsü
        output = np.zeros_like(image)
        
        # Her piksel için maksimum değeri al
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.max(window)
                
        return output
    
    def erode(self, image, kernel_size=3):
        """
        Aşındırma (Erosion) morfolojik işlemi.
        
        Beyaz nesneleri (255 değerli pikseller) küçülten bir işlemdir.
        Her piksel için komşuluk içindeki minimum değer alınır.
        Gürültü giderme, nesneleri inceltme için kullanılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü (tercihen ikili görüntü)
            kernel_size (int): Yapısal eleman boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Aşındırılmış görüntü
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        h, w = image.shape
        pad = kernel_size // 2
        
        # Padding uygula
        padded = np.pad(image, pad, mode='edge')
        
        # Çıktı görüntüsü
        output = np.zeros_like(image)
        
        # Her piksel için minimum değeri al
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.min(window)
                
        return output
    
    def opening(self, image, kernel_size=3):
        """
        Açma (Opening) morfolojik işlemi.
        
        Önce aşındırma, sonra genişletme uygulayarak gerçekleştirilir.
        Küçük nesneleri ve gürültüyü gidermek, ana nesneleri korurken
        ince çıkıntıları kaldırmak için kullanılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü (tercihen ikili görüntü)
            kernel_size (int): Yapısal eleman boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Açma işlemi uygulanmış görüntü
        """
        eroded = self.erode(image, kernel_size)
        return self.dilate(eroded, kernel_size)
    
    def closing(self, image, kernel_size=3):
        """
        Kapama (Closing) morfolojik işlemi.
        
        Önce genişletme, sonra aşındırma uygulayarak gerçekleştirilir.
        Küçük delikleri kapatmak, nesneleri birleştirmek ve konturları
        düzleştirmek için kullanılır.
        
        Args:
            image (numpy.ndarray): İşlenecek görüntü (tercihen ikili görüntü)
            kernel_size (int): Yapısal eleman boyutu (tek sayı olmalı)
            
        Returns:
            numpy.ndarray: Kapama işlemi uygulanmış görüntü
        """
        dilated = self.dilate(image, kernel_size)
        return self.erode(dilated, kernel_size) 

    def preprocess_for_model(
        self,
        image,
        denoise_type="none",
        kernel_size=3,
        sigma=1.0,
        use_hist_eq=False,
        brightness=0,
        color_mode="rgb",
        target_size=224,
    ):
        """
        Derin öğrenme modellerine girdi olacak görüntüler için esnek ön işleme.
        
        Bu fonksiyon mevcut ImageProcessor fonksiyonlarını kullanarak, seçilen
        ayarlara göre tipik bir AI preprocessing pipeline'ı uygular.
        
        İşlem sırası:
        1. Gürültü giderme (isteğe bağlı)
        2. Histogram eşitleme (isteğe bağlı)
        3. Parlaklık ayarı (isteğe bağlı)
        4. Renk uzayı seçimi (RGB / Grayscale / HSV-H)
        5. Hedef boyuta yeniden boyutlandırma (kare)
        
        Args:
            image (numpy.ndarray): Girdi görüntü
            denoise_type (str): "none", "gaussian", "mean" veya "median"
            kernel_size (int): Filtre çekirdek boyutu (tek sayı önerilir)
            sigma (float): Gaussian sigma değeri
            use_hist_eq (bool): Histogram eşitleme uygulanıp uygulanmayacağı
            brightness (int): Parlaklık ofseti (-255 ila 255)
            color_mode (str): "rgb", "grayscale" veya "hsv_h"
            target_size (int): Çıkış görüntüsünün hedef kenar boyutu (kare)
            
        Returns:
            numpy.ndarray: Ön işlenmiş görüntü
        """
        if image is None:
            return None
        
        img = image.copy()
        
        # Gürültü giderme
        k = int(kernel_size) if kernel_size is not None else 3
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        
        dt = (denoise_type or "none").lower()
        if dt == "gaussian":
            kernel = self.gaussian_kernel(k, float(sigma))
            img = self.convolve(img, kernel)
        elif dt == "mean":
            img = self.mean_filter(img, k)
        elif dt == "median":
            img = self.median_filter(img, k)
        
        # Histogram eşitleme
        if bool(use_hist_eq):
            img = self.histogram_equalization(img)
        
        # Parlaklık ayarı
        if brightness is not None:
            b = int(brightness)
            if b != 0:
                img = self.adjust_brightness(img, b)
        
        # Renk modu
        mode = (color_mode or "rgb").lower()
        if mode == "grayscale":
            img = self.to_grayscale(img)
        elif mode == "hsv_h":
            # Hue kanalını kullan
            hsv = self.rgb_to_hsv(img)
            if len(hsv.shape) == 3:
                img = hsv[..., 0].astype(np.uint8)
        
        # Hedef boyuta yeniden boyutlandırma (kare)
        if target_size is not None:
            ts = int(target_size)
            if ts > 0:
                img = self.resize_image_numpy(img, ts, ts)
        
        return img