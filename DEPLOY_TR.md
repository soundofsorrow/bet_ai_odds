# Bet AI Kupon Motoru Pro - Deploy ve Paylasim

Bu projeyi bilgisayar acik olmasa bile telefondan kullanmak ve uyelikli VIP Telegram sistemine cevirmek icin en pratik yol buluta deploy etmektir.

## 1) Render ile web + bot worker kurulumu (onerilen)

1. Projeyi GitHub'a push et.
2. Render'da `New +` -> `Blueprint` veya `Web Service` ac.
3. Bu repo icin environment variable'lari ekle:
   - `ODDS_API_KEY`
   - `FOOTBALL_DATA_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_ADMIN_IDS`
   - `TELEGRAM_PRIVATE_GROUP_ID`
   - `TELEGRAM_PAYMENT_FEE_TL=500`
   - `TELEGRAM_CONTACT=@senin_kullanici_adin`
4. Render process tipleri:
   - `web`: Streamlit panel
   - `worker_daily`: Gunluk kupon gonderimi
   - `worker_bot`: Odeme/onay ve kullanici yonetimi

## 2) Telegram tarafini hazirlama

1. `@BotFather` ile bot olustur ve token al.
2. Ozel bir Telegram grup olustur (private).
3. Botu bu gruba admin olarak ekle.
4. Bottan su yetkileri ac:
   - `Invite users via link`
   - `Post messages`
5. Grup chat id'sini al ve `TELEGRAM_PRIVATE_GROUP_ID` olarak kaydet.

## 3) Admin paneli komutlari (Telegram DM)

Botta admin komutlari:
- `/pending` -> bekleyen odeme talepleri
- `/approve <user_id>` -> kullaniciyi onaylar ve grup davet linki yollar
- `/reject <user_id> [not]` -> reddeder
- `/block <user_id> [not]` -> engeller
- `/unblock <user_id>` -> tekrar beklemeye alir
- `/approved` -> onayli liste
- `/who <user_id>` -> tekil kullanici detayi
- `/sendnow` -> kuponu anlik olarak private gruba yollar
- `/templates` -> ozellestirilebilir mesaj sablonlarini listeler
- `/ad` -> hazir reklam metni

## 4) Lokal test

Tek seferlik kupon gonderimi:

```bash
python3 telegram_bot.py --mode once
```

Gunluk otomatik gonderim:

```bash
python3 telegram_bot.py --mode daily
```

Odeme/onay botu:

```bash
python3 telegram_bot.py --mode bot
```

## 5) Guvenlik ve operasyon

- Grup private olmali.
- Kuponlar sadece private gruba gonderilir.
- Kullanici onaylandiginda tek kullanimlik davet linki alir.
- Onaysiz kullanici gruba giremez, kuponlari goremez.
- `TELEGRAM_ADMIN_IDS` disinda kimse onay komutu kullanamaz.
- Gunluk kupon gonderimi hata verirse adminlere Telegram DM ile hata metni gelir.
