# Bet AI Kupon Motoru Pro - Deploy ve Paylasim

Bu projeyi bilgisayar acik olmasa bile telefondan kullanmak ve arkadaslarla paylasmak icin en pratik yol buluta deploy etmektir.

## 1) Render ile tek URL (onerilen)

1. Projeyi GitHub'a push et.
2. Render'da `New +` -> `Web Service` -> repo sec.
3. `render.yaml` otomatik algilanir.
4. Environment Variables:
   - `ODDS_API_KEY`
   - `FOOTBALL_DATA_API_KEY`
   - `APP_PASSWORD` (zorunlu olmasi onerilir)
   - `REGION=eu`
5. Deploy bitince URL gelir: `https://...onrender.com`
6. Bu URL'yi telefondan acabilir ve arkadaslarla paylasabilirsin.

## 2) Railway ile deploy

1. Yeni proje olustur, GitHub repo bagla.
2. Start command:
   - `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`
3. Ayni env degiskenlerini ekle.
4. Deploy sonrasi verilen URL herkese acik olur.

## 3) Docker ile VPS/Sunucu

```bash
docker compose up -d --build
```

Sunucuda `8501` portunu ac ve domain bagla.

## Guvenlik Notu

- Public paylasim icin `APP_PASSWORD` mutlaka ayarla.
- API key'leri kod icine yazma, sadece env variable kullan.
- Render/Railway panelinden key yenileme kolaydir.
