# Arkadaslarla Paylasim Rehberi (Hic Bilmeyen Icin)

Bu rehberle uygulamayi link olarak paylasirsin. Arkadasin linke girer, kendi API keyini yazar ve kullanir.

## 1) Linki nasil alacaksin?

En kolay yol: Render'a deploy.

1. Projeyi GitHub'a yukle.
2. Render'da yeni `Web Service` olustur.
3. Repo'yu sec.
4. Deploy bitince sana bir URL verir:
   - Ornek: `https://bet-ai-odds.onrender.com`
5. Bu URL'yi arkadaslarina gonder.

## 2) Arkadasin ne yapacak?

1. Senin gonderdigin linki acacak.
2. Uygulamada su 2 keyi gorecek:
   - `The Odds API Key`
   - `Football-Data API Key`
3. Kendi keylerini alip kutulara yapistiracak.
4. `Kuponlari Uret` butonuna basacak.

## 3) Key nereden alinacak?

- The Odds API: [https://the-odds-api.com](https://the-odds-api.com)
- Football-Data: [https://www.football-data.org/client/register](https://www.football-data.org/client/register)

## 4) Benim keylerim gozukur mu?

Hayir. Bu sistem `REQUIRE_USER_KEYS=true` oldugu icin herkes kendi keyini girer.

## 5) Guvenlik

- Uygulama linki aciksa `APP_PASSWORD` kullan.
- API keyleri mesaja yazma, sadece uygulamadaki kutuya yapistir.
