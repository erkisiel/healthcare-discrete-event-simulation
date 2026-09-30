
import numpy as np
import simpy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
WARMUP_SURE    =  600

HASTA_KABUL_BAS=  600

SON_KABUL      = 29400

OGLE_BASLANGIC = 15000
OGLE_BITIS     = 18600

MAX_BITIS      = 33000

# ═══════════════════════════════════════════════════════════════════
# DAĞILIM PARAMETRELER BLOĞU
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# DAĞILIM PARAMETRELERİ
# ─────────────────────────────────────────────────────────────────

DIST_ILK_MUAYENE  = {"tip": "gamma", "params": (2.6972, 83.802)}

DIST_KONTROL      = {"tip": "gamma", "params": (2.5411, 99.03)}

DIST_XRAY_CEKIM_ODA = {
    0: {"tip": "gamma", "params": (2.216, 90.67)},
    1: {"tip": "gamma", "params": (2.443, 99.47)},
    2: {"tip": "gamma", "params": (5.107, 64.26)},
}

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
TRIANGULAR_GELIS_LOW  = -780
TRIANGULAR_GELIS_HIGH =  300
TRIANGULAR_GELIS_MODE = -300

# ─────────────────────────────────────────────────────────────────
# KAYIT BANKOS SÜRESİ
# ─────────────────────────────────────────────────────────────────
KAYIT_MIN = 20
KAYIT_MAX = 60

KAYIT_SLOT = {
    "ortopedi1": "kayit",
    "ortopedi2": "kayit",
    "tumor":     "kayit",
    "omurga":    "kayit",
    "el":        "kayit",
    "spor":      "kayit",
    "cocuk":     "kayit",
}

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def sure_cek(rnd, dist_dict):
    tip    = dist_dict["tip"]
    params = dist_dict["params"]
    if tip == "weibull":
        return rnd.weibullvariate(params[0], params[1])
    elif tip == "gamma":
        return rnd.gammavariate(params[0], params[1])
    elif tip == "exponential":
        return rnd.expovariate(1.0 / params[0])
    elif tip == "lognormal":
        return rnd.lognormvariate(params[0], params[1])
    else:
        raise ValueError(f"Bilinmeyen dağılım tipi: {tip}")

def travel_time():
    return rnd_seyahat.triangular(60, 180, 120)

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

rnd_walkin_ortopedi1 = random.Random(); rnd_walkin_ortopedi1.seed(1001)
rnd_walkin_el        = random.Random(); rnd_walkin_el.seed(1002)
rnd_walkin_cocuk     = random.Random(); rnd_walkin_cocuk.seed(1003)
rnd_walkin_ortopedi2 = random.Random(); rnd_walkin_ortopedi2.seed(1004)
rnd_walkin_tumor     = random.Random(); rnd_walkin_tumor.seed(1005)
rnd_walkin_omurga    = random.Random(); rnd_walkin_omurga.seed(1006)
rnd_walkin_spor      = random.Random(); rnd_walkin_spor.seed(1007)

rnd_servis_ortopedi1 = random.Random(); rnd_servis_ortopedi1.seed(2001)
rnd_servis_el        = random.Random(); rnd_servis_el.seed(2002)
rnd_servis_cocuk     = random.Random(); rnd_servis_cocuk.seed(2003)
rnd_servis_ortopedi2 = random.Random(); rnd_servis_ortopedi2.seed(2004)
rnd_servis_tumor     = random.Random(); rnd_servis_tumor.seed(2005)
rnd_servis_omurga    = random.Random(); rnd_servis_omurga.seed(2006)
rnd_servis_spor      = random.Random(); rnd_servis_spor.seed(2007)
rnd_servis_kontrol   = random.Random(); rnd_servis_kontrol.seed(2008)
rnd_servis_xray12    = random.Random(); rnd_servis_xray12.seed(2009)
rnd_servis_xray3     = random.Random(); rnd_servis_xray3.seed(2010)

rnd_gelis_sapma      = random.Random(); rnd_gelis_sapma.seed(3001)
rnd_xray_karar       = random.Random(); rnd_xray_karar.seed(3002)
rnd_xray_oda         = random.Random(); rnd_xray_oda.seed(3003)
rnd_refakatci        = random.Random(); rnd_refakatci.seed(3004)
rnd_kayit            = random.Random(); rnd_kayit.seed(3005)
rnd_xray_kayit       = random.Random(); rnd_xray_kayit.seed(3006)
rnd_seyahat          = random.Random(); rnd_seyahat.seed(3007)
rnd_xray_cekim_karar = random.Random(); rnd_xray_cekim_karar.seed(3008)

# ═══════════════════════════════════════════════════════════════════
# SİMÜLASYON PARAMETRELERİ
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
XRAY_PROB = {
    "ortopedi1": 0.624,
    "el":        0.433,
    "cocuk":     0.559,
    "ortopedi2": 0.559,
    "tumor":     0.240,
    "omurga":    0.619,
    "spor":      0.273,
}

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
XRAY_ODA_PROB = {
    "ortopedi1":  (0.621, 0.364, 0.015),
    "el":         (0.610, 0.314, 0.076),
    "cocuk":      (0.413, 0.481, 0.106),
    "ortopedi2":  (0.550, 0.427, 0.023),
    "tumor":      (0.571, 0.429, 0.000),
    "omurga":     (0.425, 0.448, 0.126),
    "spor":       (0.733, 0.000, 0.267),
}

P_XRAY_CEKIM = 0.9653

# ─────────────────────────────────────────────────────────────────
# WALK-IN HASTA GELİŞ YOĞUNLUĞU
# ─────────────────────────────────────────────────────────────────
WALKIN_LAMBDA_TOPLAM = [29.2, 55.8, 50.6, 37.8, 6.8, 29.8, 39.4, 23.6, 3.2]

WALKIN_ORANI   = 0.20

WALKIN_DAGILIM = {"ortopedi1": 0.00, "el": 0.15, "cocuk": 0.00, "ortopedi2": 0.25, "tumor": 0.30, "omurga": 0.20, "spor": 0.10}

def walkin_lambda(saat_idx, klinik):
    idx = min(saat_idx, len(WALKIN_LAMBDA_TOPLAM) - 1)
    return WALKIN_LAMBDA_TOPLAM[idx] * WALKIN_ORANI * WALKIN_DAGILIM[klinik]

MU_REFERANS_SN = 2495.14

# ═══════════════════════════════════════════════════════════════════
# RANDEVU ÇİZELGESİ OLUŞTURUCU
# ═══════════════════════════════════════════════════════════════════

def randevu_cizelgesi_olustur(schedule_no):
    hasta_listesi = []

    def aralik_ekle(baslangic_sn, bitis_sn, aralik_dk):
        t = baslangic_sn
        while t < bitis_sn:
            sapma    = rnd_gelis_sapma.triangular(TRIANGULAR_GELIS_LOW, TRIANGULAR_GELIS_HIGH, TRIANGULAR_GELIS_MODE)
            gelis_sn = max(0, t + sapma)
            hasta_listesi.append((gelis_sn, t))
            t += aralik_dk * 60

    if schedule_no == 1:
        aralik_ekle(600,  15000, 6)
        aralik_ekle(18600, 29400, 6)
    elif schedule_no == 2:
        aralik_ekle(600,  7800,  6)
        aralik_ekle(7800,  15000, 10)
        aralik_ekle(18600, 25800, 8)
        aralik_ekle(25800, 29400, 10)
    elif schedule_no == 3:
        aralik_ekle(600,  7800,  6)
        aralik_ekle(7800,  15000, 8)

    hasta_listesi.sort(key=lambda x: x[0])
    return hasta_listesi

# ═══════════════════════════════════════════════════════════════════
# SONUÇ YAPISI
# ═══════════════════════════════════════════════════════════════════

def bos_results():
    return {
        "sistem_sureleri":   [],
        "bekleme_sureleri":  [],
        "bekleme_ortopedi1": [],
        "bekleme_el":        [],
        "bekleme_cocuk":     [],
        "bekleme_ortopedi2": [],
        "bekleme_tumor":     [],
        "bekleme_omurga":    [],
        "bekleme_spor":      [],
        "kuyruk_ortopedi1":  [],
        "kuyruk_el":         [],
        "kuyruk_cocuk":      [],
        "kuyruk_ortopedi2":  [],
        "kuyruk_tumor":      [],
        "kuyruk_omurga":     [],
        "kuyruk_spor":       [],
        "kuyruk_xray1":      [],
        "kuyruk_xray2":      [],
        "kuyruk_xray3":      [],
        "xray1_mesgul":      0,
        "xray2_mesgul":      0,
        "xray3_mesgul":      0,
        "kuyruk_kayit":    [],
        "kuyruk_xray_kayit": [],
        "gun_sonu_hasta":    0,
        "toplam_hasta":      0,
        "xray_giden":        0,
        "gec_gelen":         0,
        "aktif_sure":        0,
    }

# ═══════════════════════════════════════════════════════════════════
# HASTA SINIFI
# ═══════════════════════════════════════════════════════════════════

class Hasta:
    _id = 0

    def __init__(self, randevulu, arrival, departman,
                 randevu_saati=None, gec_geldi=False, walkin_xray=False):
        Hasta._id       += 1
        self.id          = Hasta._id
        self.randevulu   = randevulu
        self.arrival     = arrival
        self.departman   = departman
        self.randevu_saati = randevu_saati
        self.gec_geldi   = gec_geldi   
        self.walkin_xray = walkin_xray  
        self.xray_durumu = False
        self.xray_istem  = False
        self.ikinci_giris= False
        self.xray_oda_no = None
        self.bekleme_sure= 0.0
        self.hizmet_sure = 0.0
        self.refakatci   = rnd_refakatci.choices(
            [0, 1, 2, 3], weights=[0.21, 0.35, 0.29, 0.15])[0]

# ═══════════════════════════════════════════════════════════════════
# GLOBAL SAYAÇLAR
# ═══════════════════════════════════════════════════════════════════

class Sayac:
    hasta_sayisi = 0
    toplam_kisi  = 0

DEPT_RNG = {
    "ortopedi1": rnd_servis_ortopedi1,
    "el":        rnd_servis_el,
    "cocuk":     rnd_servis_cocuk,
    "ortopedi2": rnd_servis_ortopedi2,
    "tumor":     rnd_servis_tumor,
    "omurga":    rnd_servis_omurga,
    "spor":      rnd_servis_spor,
}

# ═══════════════════════════════════════════════════════════════════
# HASTA AKIŞ FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════

def hasta_akis(env, hasta, kaynaklar, results):
    if env.now < hasta.arrival:
        yield env.timeout(hasta.arrival - env.now)

    Sayac.hasta_sayisi += 1
    Sayac.toplam_kisi  += (hasta.refakatci + 1)
    if hasta.departman == "cocuk":
        Sayac.toplam_kisi += 1

    # ── Kayıt bankosu ────────────────────────────────────────────
    slot = KAYIT_SLOT[hasta.departman]
    kayit_kaynak = kaynaklar[slot]
    enter_kayit = env.now
    with kayit_kaynak.request() as req:
        yield req
        ogle_bekleme_kayit = 0
        if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
            ogle_bekleme_kayit = OGLE_BITIS - env.now
            yield env.timeout(ogle_bekleme_kayit)
        hasta.bekleme_sure += env.now - enter_kayit - ogle_bekleme_kayit
        sure_kayit = rnd_kayit.uniform(KAYIT_MIN, KAYIT_MAX)
        yield env.timeout(sure_kayit)

    yield env.process(muayene(env, hasta, kaynaklar, results))

    hasta.hizmet_sure = env.now - hasta.arrival
    results["sistem_sureleri"].append(hasta.hizmet_sure)
    results["bekleme_sureleri"].append(hasta.bekleme_sure)
    results[f"bekleme_{hasta.departman}"].append(hasta.bekleme_sure)
    Sayac.hasta_sayisi -= 1
    Sayac.toplam_kisi  -= (hasta.refakatci + 1)
    if hasta.departman == "cocuk":
        Sayac.toplam_kisi -= 1


def muayene(env, hasta, kaynaklar, results):
    dept    = hasta.departman
    kaynak  = kaynaklar[dept]
    rng     = DEPT_RNG[dept]

    enter_time = env.now

    if (hasta.randevulu
            and hasta.randevu_saati is not None
            and env.now < hasta.randevu_saati
            and not hasta.ikinci_giris
            and not hasta.gec_geldi):
        yield env.timeout(hasta.randevu_saati - env.now)
        enter_time = env.now

    if hasta.ikinci_giris:
        oncelik = 0
    elif not hasta.randevulu:
        oncelik = 1
    else:
        oncelik = 1

    if isinstance(kaynak, simpy.PriorityResource):
        req_ctx = kaynak.request(priority=oncelik)
    else:
        req_ctx = kaynak.request()
    with req_ctx as req:
        yield req
        if env.now < HASTA_KABUL_BAS:
            yield env.timeout(HASTA_KABUL_BAS - env.now)
            enter_time = max(enter_time, HASTA_KABUL_BAS)
        ogle_bekleme = 0
        if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
            ogle_bekleme = OGLE_BITIS - env.now
            yield env.timeout(ogle_bekleme)
        hasta.bekleme_sure += env.now - enter_time - ogle_bekleme

        if not hasta.ikinci_giris:
            sure = sure_cek(rng, DIST_ILK_MUAYENE)
        else:
            sure = sure_cek(rnd_servis_kontrol, DIST_KONTROL)
        yield env.timeout(sure)

    xray_zorunlu  = hasta.walkin_xray and not hasta.xray_durumu
    xray_olasilik = (not hasta.walkin_xray
                     and not hasta.xray_durumu
                     and rnd_xray_karar.random() < XRAY_PROB[dept])

    if xray_zorunlu or xray_olasilik:
        hasta.xray_istem = True
        results["xray_giden"] += 1
        yield env.process(xray_akis(env, hasta, kaynaklar, results))


def xray_akis(env, hasta, kaynaklar, results):

    # ── Seyahat: muayene odasından x-ray bölümüne ───────────────
    seyahat = travel_time()
    if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
        yield env.timeout(OGLE_BITIS - env.now)
    yield env.timeout(seyahat)
    if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
        yield env.timeout(OGLE_BITIS - env.now)

    # ── X-ray kayıt bankosu (kapasite 1) ─────────────────────────
    enter_xray_kayit = env.now
    with kaynaklar["xray_kayit"].request() as req:
        yield req
        if env.now < HASTA_KABUL_BAS:
            yield env.timeout(HASTA_KABUL_BAS - env.now)
            enter_xray_kayit = max(enter_xray_kayit, HASTA_KABUL_BAS)
        ogle_bekleme_xkayit = 0
        if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
            ogle_bekleme_xkayit = OGLE_BITIS - env.now
            yield env.timeout(ogle_bekleme_xkayit)
        hasta.bekleme_sure += env.now - enter_xray_kayit - ogle_bekleme_xkayit
        sure_xray_kayit = rnd_xray_kayit.uniform(KAYIT_MIN, KAYIT_MAX)
        yield env.timeout(sure_xray_kayit)

    if not hasta.walkin_xray and rnd_xray_cekim_karar.random() > P_XRAY_CEKIM:
        return

    # ── Oda seçimi ve X-Ray bekleme + çekim ─────────────────────
    dept       = hasta.departman
    p1, p2, p3 = XRAY_ODA_PROB[dept]
    r          = rnd_xray_oda.random()
    oda_no     = 0 if r < p1 else (1 if r < p1 + p2 else 2)
    hasta.xray_oda_no = oda_no

    enter_xray = env.now

    if oda_no == 2:
        with kaynaklar["xray3"].request() as req:
            yield req
            ogle_bekleme_xray = 0
            if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
                ogle_bekleme_xray = OGLE_BITIS - env.now
                yield env.timeout(ogle_bekleme_xray)
            hasta.bekleme_sure += env.now - enter_xray - ogle_bekleme_xray
            cekim = sure_cek(rnd_servis_xray3, DIST_XRAY_CEKIM_ODA[2])
            yield env.timeout(cekim)
            results["xray3_mesgul"] += cekim
    else:
        oda_key = "xray1" if oda_no == 0 else "xray2"
        with kaynaklar[oda_key].request() as req:
            yield req
            ogle_bekleme_xray = 0
            if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
                ogle_bekleme_xray = OGLE_BITIS - env.now
                yield env.timeout(ogle_bekleme_xray)
            hasta.bekleme_sure += env.now - enter_xray - ogle_bekleme_xray
            cekim = sure_cek(rnd_servis_xray12, DIST_XRAY_CEKIM_ODA[oda_no])
            yield env.timeout(cekim)
            results[f"{oda_key}_mesgul"] += cekim

    hasta.xray_durumu = True

    # ── Seyahat: x-ray bölümünden muayene odasına geri ──────────
    if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
        yield env.timeout(OGLE_BITIS - env.now)
    yield env.timeout(travel_time())
    if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
        yield env.timeout(OGLE_BITIS - env.now)

    # ── Kontrol: hasta direkt kuyruğa giriyor (bekleme muayeneden gelir) ──
    hasta.ikinci_giris = True
    yield env.process(muayene(env, hasta, kaynaklar, results))

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def walkin_generator(env, departman, rnd, kaynaklar, results):
    WALKIN_SON = 25800   
    WALKIN_ARA_BAS = 11400  

    if env.now < HASTA_KABUL_BAS:
        yield env.timeout(HASTA_KABUL_BAS - env.now)

    while env.now < WALKIN_SON:
        if WALKIN_ARA_BAS <= env.now < OGLE_BASLANGIC:
            yield env.timeout(OGLE_BASLANGIC - env.now)
            continue

        if OGLE_BASLANGIC <= env.now < OGLE_BITIS:
            yield env.timeout(OGLE_BITIS - env.now)
            continue

        saat_idx  = min(int((env.now - WARMUP_SURE) // 3600), len(WALKIN_LAMBDA_TOPLAM) - 1)
        lam       = walkin_lambda(saat_idx, departman)

        if lam <= 0:
            yield env.timeout(300)
            continue

        lam_per_sn  = lam / 3600
        sonraki     = rnd.expovariate(lam_per_sn)
        yield env.timeout(sonraki)

        if env.now >= WALKIN_SON:
            break
        if WALKIN_ARA_BAS <= env.now < OGLE_BASLANGIC:
            continue

        hasta = Hasta(False, env.now, departman, walkin_xray=True)
        env.process(hasta_akis(env, hasta, kaynaklar, results))

# ═══════════════════════════════════════════════════════════════════
# RANDEVULU GENERATÖR
# ═══════════════════════════════════════════════════════════════════

def randevulu_generator(env, departman, schedule_no, kaynaklar, results):
    cizelge = randevu_cizelgesi_olustur(schedule_no)
    cizelge.append((10**9, 10**9))

    for i in range(len(cizelge) - 1):
        gelis_sn, randevu_sn = cizelge[i]

        if randevu_sn >= SON_KABUL:
            break

        gec_geldi = gelis_sn > randevu_sn

        hasta = Hasta(
            randevulu=True,
            arrival=gelis_sn,
            departman=departman,
            randevu_saati=randevu_sn,
            gec_geldi=gec_geldi
        )
        if gec_geldi:
            results["gec_gelen"] += 1

        env.process(hasta_akis(env, hasta, kaynaklar, results))

        bekle = cizelge[i + 1][0] - gelis_sn
        yield env.timeout(max(0, bekle))

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def kuyruk_monitor(env, kaynaklar, results):
    while True:
        yield env.timeout(60)
        t = env.now
        results["kuyruk_ortopedi1"].append( (t, len(kaynaklar["ortopedi1"].queue)))
        results["kuyruk_el"].append(        (t, len(kaynaklar["el"].queue)))
        results["kuyruk_cocuk"].append(     (t, len(kaynaklar["cocuk"].queue)))
        results["kuyruk_ortopedi2"].append( (t, len(kaynaklar["ortopedi2"].queue)))
        results["kuyruk_tumor"].append(     (t, len(kaynaklar["tumor"].queue)))
        results["kuyruk_omurga"].append(    (t, len(kaynaklar["omurga"].queue)))
        results["kuyruk_spor"].append(      (t, len(kaynaklar["spor"].queue)))
        results["kuyruk_xray1"].append(     (t, len(kaynaklar["xray1"].queue)))
        results["kuyruk_xray2"].append(     (t, len(kaynaklar["xray2"].queue)))
        results["kuyruk_xray3"].append(     (t, len(kaynaklar["xray3"].queue)))
        results["kuyruk_kayit"].append(     (t, len(kaynaklar["kayit"].queue)))
        results["kuyruk_xray_kayit"].append((t, len(kaynaklar["xray_kayit"].queue)))

# ═══════════════════════════════════════════════════════════════════
# ZAMAN DÖNÜŞTÜRÜCÜ
# ═══════════════════════════════════════════════════════════════════

def sn_to_saat(sn):
    return sn / 3600 + 7.5

# ═══════════════════════════════════════════════════════════════════
# GRAFİKLER
# ═══════════════════════════════════════════════════════════════════

def grafik_ciz(results_list):
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    KLINIK_RENKLER = {
        "ortopedi1": "#2E86AB", "el": "#A23B72", "cocuk": "#F18F01",
        "ortopedi2": "#C73E1D", "tumor": "#264653", "omurga": "#2A9D8F",
        "spor": "#E9C46A"
    }
    KLINIK_ISIMLER = {
        "ortopedi1": "Ortopedi1", "el": "El Cer.", "cocuk": "Çocuk Ort.",
        "ortopedi2": "Ortopedi2", "tumor": "Tümör", "omurga": "Omurga", "spor": "Spor Hek."
    }

    # ── 1. Sistem ve Bekleme Süresi Histogramı ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tum_sistem  = [s / 60 for r in results_list for s in r["sistem_sureleri"]]
    tum_bekleme = [s / 60 for r in results_list for s in r["bekleme_sureleri"]]

    sns.histplot(tum_sistem,  bins=40, kde=True, color="#2E86AB", ax=axes[0])
    axes[0].axvline(np.mean(tum_sistem), color="red", ls="--", lw=2,
                    label=f"Ort: {np.mean(tum_sistem):.1f} dk")
    axes[0].set_xlabel("Sistem Süresi (dk)")
    axes[0].set_ylabel("Frekans")
    axes[0].set_title("Sistem Süresi Dağılımı (30 Replikasyon)")
    axes[0].legend()

    sns.histplot(tum_bekleme, bins=40, kde=True, color="#C73E1D", ax=axes[1])
    axes[1].axvline(np.mean(tum_bekleme), color="navy", ls="--", lw=2,
                    label=f"Ort: {np.mean(tum_bekleme):.1f} dk")
    axes[1].set_xlabel("Bekleme Süresi (dk)")
    axes[1].set_ylabel("Frekans")
    axes[1].set_title("Bekleme Süresi Dağılımı (30 Replikasyon)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("grafik_sistem_bekleme.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 2. Klinik Bazlı Ortalama Bekleme Süresi ─────────────────
    klinikler = ["ortopedi1", "el", "cocuk", "ortopedi2", "tumor", "omurga", "spor"]
    isimler   = [KLINIK_ISIMLER[k] for k in klinikler]
    ort_beklemeler = []
    for k in klinikler:
        vals = [s / 60 for r in results_list for s in r[f"bekleme_{k}"]]
        ort_beklemeler.append(np.mean(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(11, 5))
    renkler_bar = [KLINIK_RENKLER[k] for k in klinikler]
    bars = ax.bar(isimler, ort_beklemeler, color=renkler_bar,
                  edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, ort_beklemeler):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Ortalama Bekleme Süresi (dk)")
    ax.set_title("Klinik Bazlı Ortalama Bekleme Süresi (30 Replikasyon)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafik_klinik_bekleme.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 3. X-Ray Doluluk Oranları ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    oda_isimleri = ["Oda-1", "Oda-2", "Oda-3"]
    oda_keys     = ["xray1_mesgul", "xray2_mesgul", "xray3_mesgul"]
    doluluk      = []
    for key in oda_keys:
        per_rep = [r[key] / r["aktif_sure"] * 100
                   for r in results_list if r["aktif_sure"] > 0]
        doluluk.append(np.mean(per_rep))

    renkler_xray = ["#2E86AB", "#A23B72", "#F18F01"]
    bars = ax.bar(oda_isimleri, doluluk, color=renkler_xray,
                  edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, doluluk):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"%{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Doluluk Oranı (%)")
    ax.set_title("X-Ray Odaları Doluluk Oranı")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafik_xray_doluluk.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 4. Kuyruk Zaman Serisi (İlk Replikasyon) ────────────────
    r0 = results_list[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for k in klinikler:
        key = f"kuyruk_{k}"
        if r0[key]:
            zamanlar = [sn_to_saat(t) for t, _ in r0[key]]
            degerler = [n for _, n in r0[key]]
            axes[0].plot(zamanlar, degerler, label=KLINIK_ISIMLER[k],
                         color=KLINIK_RENKLER[k], lw=1.5)
    axes[0].axvline(12, color="gray", ls="--", lw=1, alpha=0.6, label="Öğle")
    axes[0].axvline(16, color="red",  ls="--", lw=1, alpha=0.6, label="Son kabul")
    axes[0].set_xlabel("Saat")
    axes[0].set_ylabel("Kuyruk Uzunluğu")
    axes[0].set_title("Klinik Kuyruk Uzunlukları — Zaman Serisi (Rep. 1)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    xray_keys    = ["kuyruk_xray1", "kuyruk_xray2", "kuyruk_xray3"]
    xray_isimler = ["X-Ray Oda-1",  "X-Ray Oda-2",  "X-Ray Oda-3"]
    for key, isim, renk in zip(xray_keys, xray_isimler, renkler_xray):
        if r0[key]:
            zamanlar = [sn_to_saat(t) for t, _ in r0[key]]
            degerler = [n for _, n in r0[key]]
            axes[1].plot(zamanlar, degerler, label=isim, color=renk, lw=1.5)
    axes[1].axvline(12, color="gray", ls="--", lw=1, alpha=0.6, label="Öğle")
    axes[1].axvline(16, color="red",  ls="--", lw=1, alpha=0.6, label="Son kabul")
    axes[1].set_xlabel("Saat")
    axes[1].set_ylabel("Kuyruk Uzunluğu")
    axes[1].set_title("X-Ray Kuyruk Uzunlukları — Zaman Serisi (Rep. 1)")
    axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafik_kuyruk_zaman.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 5. Kayıt Bankosu Kuyruk Zaman Serisi (İlk Rep.) ─────────
    fig, ax = plt.subplots(figsize=(14, 5))
    kayit_keys   = ["kuyruk_kayit", "kuyruk_xray_kayit"]
    kayit_isimler= ["Kayıt", "X-Ray Kayıt"]
    kayit_renkler= ["#2E86AB", "#C73E1D", "#F18F01", "#264653"]
    for key, isim, renk in zip(kayit_keys, kayit_isimler, kayit_renkler):
        if r0[key]:
            zamanlar = [sn_to_saat(t) for t, _ in r0[key]]
            degerler = [n for _, n in r0[key]]
            ax.plot(zamanlar, degerler, label=isim, color=renk, lw=1.5)
    ax.axvline(12, color="gray", ls="--", lw=1, alpha=0.6, label="Öğle")
    ax.axvline(16, color="red",  ls="--", lw=1, alpha=0.6, label="Son kabul")
    ax.set_xlabel("Saat")
    ax.set_ylabel("Kuyruk Uzunluğu")
    ax.set_title("Kayıt Bankosu Kuyruk Uzunlukları — Zaman Serisi (Rep. 1)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafik_kayit_kuyruk.png", dpi=150, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# T-TEST & ÖZET
# ═══════════════════════════════════════════════════════════════════

def t_test_ve_ozet(ortalama_listesi, results_list):
    mu       = MU_REFERANS_SN
    n        = len(ortalama_listesi)
    ortalama = np.mean(ortalama_listesi)
    std      = np.std(ortalama_listesi, ddof=1)
    t_stat   = (ortalama - mu) / (std / np.sqrt(n))
    p_value  = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))

    print("\n" + "=" * 65)
    print("ÖZET SONUÇLAR — 30 REPLİKASYON")
    print("=" * 65)

    print(f"\n  Replikasyon bazlı ortalama sistem süreleri (dk):")
    for i, v in enumerate(ortalama_listesi):
        print(f"    Rep {i+1:2d}: {v/60:.2f} dk")

    tum_sistem  = [s/60 for r in results_list for s in r["sistem_sureleri"]]
    tum_bekleme = [s/60 for r in results_list for s in r["bekleme_sureleri"]]

    t_kritik  = t_dist.ppf(0.975, df=n - 1)
    hata_payi = t_kritik * std / np.sqrt(n)
    ci_alt    = (ortalama - hata_payi) / 60
    ci_ust    = (ortalama + hata_payi) / 60

    print(f"\n  ┌─ Sistem Süresi ────────────────────────────────────")
    print(f"  │  Genel ortalama  : {ortalama/60:.2f} dk")
    print(f"  │  Referans (μ)    : {mu/60:.2f} dk")
    print(f"  │  Std sapma       : {std/60:.2f} dk")
    print(f"  │  %95 Güven Aralığı: [{ci_alt:.2f}, {ci_ust:.2f}] dk")
    print(f"  │  Ort (tüm rep.)  : {np.mean(tum_sistem):.2f} dk")
    print(f"  │  Max             : {np.max(tum_sistem):.2f} dk")
    print(f"  └────────────────────────────────────────────────────")

    print(f"\n  ┌─ Bekleme Süresi ───────────────────────────────────")
    print(f"  │  Ort. bekleme    : {np.mean(tum_bekleme):.2f} dk")
    print(f"  │  Max bekleme     : {np.max(tum_bekleme):.2f} dk")
    print(f"  └────────────────────────────────────────────────────")

    print(f"\n  ┌─ t-Test ───────────────────────────────────────────")
    print(f"  │  t istatistiği   : {t_stat:.4f}")
    print(f"  │  p-değeri        : {p_value:.4f}")
    if p_value < 0.05:
        print(f"  │  → H0 REDDEDİLDİ: Simülasyon referanstan farklı")
    else:
        print(f"  │  → H0 reddedilemedi: Simülasyon referansla uyumlu ✓")
    print(f"  └────────────────────────────────────────────────────")

    print(f"\n  Klinik Bazlı Ortalama Bekleme Süreleri:")
    klinikler = ["ortopedi1","el","cocuk","ortopedi2","tumor","omurga","spor"]
    isimler   = ["Ortopedi1","El Cer.","Çocuk Ort.","Ortopedi2","Tümör","Omurga","Spor Hek."]
    for k, isim in zip(klinikler, isimler):
        vals = [s/60 for r in results_list for s in r[f"bekleme_{k}"]]
        if vals:
            print(f"    {isim:<15}: ort={np.mean(vals):.2f} dk  "
                  f"max={np.max(vals):.2f} dk  n={len(vals)}")

    print(f"\n  X-Ray Doluluk Oranları (aktif süreye göre):")
    for key, isim in [("xray1_mesgul","Oda-1"),
                      ("xray2_mesgul","Oda-2"),
                      ("xray3_mesgul","Oda-3")]:
        per_rep = [r[key] / r["aktif_sure"] * 100
                   for r in results_list if r["aktif_sure"] > 0]
        print(f"    {isim}: %{np.mean(per_rep):.1f}")

    print(f"\n  Diğer İstatistikler:")
    print(f"    Geç gelen randevulu (Rep 1) : "
          f"{results_list[0]['gec_gelen']}")
    print(f"    X-ray giden (Rep 1)         : "
          f"{results_list[0]['xray_giden']}")
    print(f"    Gün sonu kalan hasta (ort.) : "
          f"{np.mean([r['gun_sonu_hasta'] for r in results_list]):.1f}")

    print(f"\n  Ortalama Kayıt Bankosu Kuyruk Uzunlukları:")
    kayit_keys   = ["kuyruk_kayit", "kuyruk_xray_kayit"]
    kayit_isimler= ["Kayıt", "X-Ray Kayıt"]
    for key, isim in zip(kayit_keys, kayit_isimler):
        vals = [n for r in results_list for _, n in r[key]]
        if vals:
            print(f"    {isim:<20}: ort={np.mean(vals):.2f}  max={np.max(vals):.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot([v/60 for v in ortalama_listesi], bins=8, kde=True,
                 color="#2E86AB", ax=ax)
    ax.axvline(mu/60, color="red", ls="--", lw=2,
               label=f"Referans μ = {mu/60:.1f} dk")
    ax.axvline(ortalama/60, color="green", ls="-", lw=2,
               label=f"Sim. ort. = {ortalama/60:.1f} dk")
    ax.set_xlabel("Ortalama Sistem Süresi (dk)")
    ax.set_ylabel("Frekans")
    ax.set_title(f"t-Test | p-değeri = {p_value:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("grafik_ttest.png", dpi=150, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# ANA ÇALIŞMA — 30 REPLİKASYON
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# ANA ÇALIŞMA DÖNGÜSÜ — 30 REPLİKASYON
# ─────────────────────────────────────────────────────────────────
ortalama_listesi = []
tum_results      = []

print("=" * 65)
print("SİMÜLASYON BAŞLIYOR — 30 REPLİKASYON")
print(f"Dağılım tipi: {DIST_ILK_MUAYENE['tip'].upper()}")
print(f"t-test referansı: {MU_REFERANS_SN/60:.2f} dk")
print("=" * 65)

for rep in range(30):

    # ── Her replikasyonda tüm seed'leri güncelle ─────────────────
    rnd_walkin_ortopedi1.seed( 1001 + rep * 100)
    rnd_walkin_el.seed(        1002 + rep * 100)
    rnd_walkin_cocuk.seed(     1003 + rep * 100)
    rnd_walkin_ortopedi2.seed( 1004 + rep * 100)
    rnd_walkin_tumor.seed(     1005 + rep * 100)
    rnd_walkin_omurga.seed(    1006 + rep * 100)
    rnd_walkin_spor.seed(      1007 + rep * 100)
    rnd_servis_ortopedi1.seed( 2001 + rep * 100)
    rnd_servis_el.seed(        2002 + rep * 100)
    rnd_servis_cocuk.seed(     2003 + rep * 100)
    rnd_servis_ortopedi2.seed( 2004 + rep * 100)
    rnd_servis_tumor.seed(     2005 + rep * 100)
    rnd_servis_omurga.seed(    2006 + rep * 100)
    rnd_servis_spor.seed(      2007 + rep * 100)
    rnd_servis_kontrol.seed(   2008 + rep * 100)
    rnd_servis_xray12.seed(    2009 + rep * 100)
    rnd_servis_xray3.seed(     2010 + rep * 100)
    rnd_gelis_sapma.seed(      3001 + rep * 100)
    rnd_xray_karar.seed(       3002 + rep * 100)
    rnd_xray_oda.seed(         3003 + rep * 100)
    rnd_refakatci.seed(        3004 + rep * 100)
    rnd_kayit.seed(            3005 + rep * 100)
    rnd_xray_kayit.seed(       3006 + rep * 100)
    rnd_seyahat.seed(          3007 + rep * 100)
    rnd_xray_cekim_karar.seed( 3008 + rep * 100)    

    # ── Sayaçları sıfırla ────────────────────────────────────────
    Hasta._id          = 0
    Sayac.hasta_sayisi = 0
    Sayac.toplam_kisi  = 0

    # ── SimPy ortamını ve kaynakları oluştur ─────────────────────
    env = simpy.Environment()
    kaynaklar = {
        "ortopedi1":    simpy.PriorityResource(env, capacity=1),
        "el":           simpy.PriorityResource(env, capacity=1),
        "cocuk":        simpy.PriorityResource(env, capacity=1),
        "ortopedi2":    simpy.PriorityResource(env, capacity=1),
        "tumor":        simpy.PriorityResource(env, capacity=1),
        "omurga":       simpy.PriorityResource(env, capacity=1),
        "spor":         simpy.PriorityResource(env, capacity=1),
        "xray1":        simpy.Resource(env, capacity=1),
        "xray2":        simpy.Resource(env, capacity=1),
        "xray3":        simpy.Resource(env, capacity=1),
        "kayit":        simpy.Resource(env, capacity=1),
        "xray_kayit":   simpy.Resource(env, capacity=1),
    }
    results = bos_results()

    # ── Randevulu generatörler ───────────────────────────────────
    env.process(randevulu_generator(env, "ortopedi1",  1, kaynaklar, results))
    env.process(randevulu_generator(env, "el",         2, kaynaklar, results))
    env.process(randevulu_generator(env, "cocuk",      1, kaynaklar, results))
    env.process(randevulu_generator(env, "ortopedi2",  2, kaynaklar, results))
    env.process(randevulu_generator(env, "tumor",      1, kaynaklar, results))
    env.process(randevulu_generator(env, "omurga",     2, kaynaklar, results))
    env.process(randevulu_generator(env, "spor",       3, kaynaklar, results))

    # ── Walk-in generatörler ─────────────────────────────────────
    env.process(walkin_generator(env, "ortopedi1",
                                 rnd_walkin_ortopedi1, kaynaklar, results))
    env.process(walkin_generator(env, "el",
                                 rnd_walkin_el,        kaynaklar, results))
    env.process(walkin_generator(env, "cocuk",
                                 rnd_walkin_cocuk,     kaynaklar, results))
    env.process(walkin_generator(env, "ortopedi2",
                                 rnd_walkin_ortopedi2, kaynaklar, results))
    env.process(walkin_generator(env, "tumor",
                                 rnd_walkin_tumor,     kaynaklar, results))
    env.process(walkin_generator(env, "omurga",
                                 rnd_walkin_omurga,    kaynaklar, results))
    env.process(walkin_generator(env, "spor",
                                 rnd_walkin_spor,      kaynaklar, results))

    # ── Kuyruk monitör ───────────────────────────────────────────
    env.process(kuyruk_monitor(env, kaynaklar, results))

    # ── Simülasyonu 16:00'a (SON_KABUL) kadar çalıştır ──────────
    env.run(until=SON_KABUL)

    # ── 16:00-17:00: İçerideki hastaları bitir ───────────────────
    adim = 0
    while Sayac.hasta_sayisi > 0 and env.now < MAX_BITIS:
        env.run(until=min(env.now + 60, MAX_BITIS))
        adim += 1

    # ── Replikasyon sonuçlarını kaydet ───────────────────────────
    results["gun_sonu_hasta"] = Sayac.hasta_sayisi
    results["toplam_hasta"]   = Hasta._id
    results["aktif_sure"]     = env.now - WARMUP_SURE - (OGLE_BITIS - OGLE_BASLANGIC)

    # ── Korunum kontrolü (doğrulama) ────────────────────────────
    tamamlanan = len(results["sistem_sureleri"])
    kalan      = results["gun_sonu_hasta"]
    toplam     = results["toplam_hasta"]
    korunum_ok = (tamamlanan + kalan) == toplam
    if not korunum_ok:
        print(f"  [UYARI] Rep {rep+1}: Korunum hatası! "
              f"Tamamlanan({tamamlanan}) + Kalan({kalan}) = {tamamlanan+kalan} ≠ Toplam({toplam})")

    ort = (np.mean(results["sistem_sureleri"])
           if results["sistem_sureleri"] else 0)
    ortalama_listesi.append(ort)
    tum_results.append(results)

    bitis_saat = sn_to_saat(env.now)
    print(f"  Rep {rep+1:2d}: {Hasta._id:4d} hasta | "
          f"Ort. sistem {ort/60:.2f} dk | "
          f"X-ray {results['xray_giden']:4d} | "
          f"Geç gelen {results['gec_gelen']:3d} | "
          f"Bitiş {bitis_saat:.2f} | "
          f"Kalan {results['gun_sonu_hasta']}")

# ── Grafikler ve özet ────────────────────────────────────────────
grafik_ciz(tum_results)
t_test_ve_ozet(ortalama_listesi, tum_results)

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def tek_senaryo_calistir(n_rep=30,
                         xray_prob_override=None,
                         xray_oda_prob_override=None,
                         walkin_orani_override=None,
                         kapasite_override=None,
                         fifo=False,
                         seed_offset=0):
    # ── Geçici parametre geçersizleştirme ────────────────────────
    global XRAY_PROB, WALKIN_ORANI, XRAY_ODA_PROB

    orijinal_xray_prob    = XRAY_PROB.copy()
    orijinal_walkin_orani = WALKIN_ORANI
    orijinal_xray_oda     = {k: v for k, v in XRAY_ODA_PROB.items()}

    if xray_prob_override:
        for k, v in xray_prob_override.items():
            XRAY_PROB[k] = v
    if xray_oda_prob_override:
        for k, v in xray_oda_prob_override.items():
            XRAY_ODA_PROB[k] = v
    if walkin_orani_override is not None:
        WALKIN_ORANI = walkin_orani_override

    ort_listesi  = []
    res_listesi  = []

    for rep in range(n_rep):
        so = seed_offset * 10000
        rnd_walkin_ortopedi1.seed( 1001 + rep * 100 + so)
        rnd_walkin_el.seed(        1002 + rep * 100 + so)
        rnd_walkin_cocuk.seed(     1003 + rep * 100 + so)
        rnd_walkin_ortopedi2.seed( 1004 + rep * 100 + so)
        rnd_walkin_tumor.seed(     1005 + rep * 100 + so)
        rnd_walkin_omurga.seed(    1006 + rep * 100 + so)
        rnd_walkin_spor.seed(      1007 + rep * 100 + so)
        rnd_servis_ortopedi1.seed( 2001 + rep * 100 + so)
        rnd_servis_el.seed(        2002 + rep * 100 + so)
        rnd_servis_cocuk.seed(     2003 + rep * 100 + so)
        rnd_servis_ortopedi2.seed( 2004 + rep * 100 + so)
        rnd_servis_tumor.seed(     2005 + rep * 100 + so)
        rnd_servis_omurga.seed(    2006 + rep * 100 + so)
        rnd_servis_spor.seed(      2007 + rep * 100 + so)
        rnd_servis_kontrol.seed(   2008 + rep * 100 + so)
        rnd_servis_xray12.seed(    2009 + rep * 100 + so)
        rnd_servis_xray3.seed(     2010 + rep * 100 + so)
        rnd_gelis_sapma.seed(      3001 + rep * 100 + so)
        rnd_xray_karar.seed(       3002 + rep * 100 + so)
        rnd_xray_oda.seed(         3003 + rep * 100 + so)
        rnd_refakatci.seed(        3004 + rep * 100 + so)
        rnd_kayit.seed(            3005 + rep * 100 + so)
        rnd_xray_kayit.seed(       3006 + rep * 100 + so)
        rnd_seyahat.seed(          3007 + rep * 100 + so)
        rnd_xray_cekim_karar.seed( 3008 + rep * 100 + so)        

        Hasta._id          = 0
        Sayac.hasta_sayisi = 0
        Sayac.toplam_kisi  = 0

        env = simpy.Environment()

        kap = {
            "ortopedi1": 1, "el": 1, "cocuk": 1, "ortopedi2": 1,
            "tumor": 1, "omurga": 1, "spor": 1,
        }
        if kapasite_override:
            kap.update(kapasite_override)

        KlasKaynak = simpy.Resource if fifo else simpy.PriorityResource
        kaynaklar = {
            "ortopedi1": KlasKaynak(env, capacity=kap["ortopedi1"]),
            "el":        KlasKaynak(env, capacity=kap["el"]),
            "cocuk":     KlasKaynak(env, capacity=kap["cocuk"]),
            "ortopedi2": KlasKaynak(env, capacity=kap["ortopedi2"]),
            "tumor":     KlasKaynak(env, capacity=kap["tumor"]),
            "omurga":    KlasKaynak(env, capacity=kap["omurga"]),
            "spor":      KlasKaynak(env, capacity=kap["spor"]),
            "xray1":     simpy.Resource(env, capacity=1),
            "xray2":     simpy.Resource(env, capacity=1),
            "xray3":     simpy.Resource(env, capacity=1),
            "kayit":     simpy.Resource(env, capacity=1),
            "xray_kayit":simpy.Resource(env, capacity=1),
        }
        results = bos_results()

        env.process(randevulu_generator(env, "ortopedi1", 1, kaynaklar, results))
        env.process(randevulu_generator(env, "el",        2, kaynaklar, results))
        env.process(randevulu_generator(env, "cocuk",     1, kaynaklar, results))
        env.process(randevulu_generator(env, "ortopedi2", 2, kaynaklar, results))
        env.process(randevulu_generator(env, "tumor",     1, kaynaklar, results))
        env.process(randevulu_generator(env, "omurga",    2, kaynaklar, results))
        env.process(randevulu_generator(env, "spor",      3, kaynaklar, results))

        WALKIN_RNG_LIST = [
            ("ortopedi1", rnd_walkin_ortopedi1),
            ("el",        rnd_walkin_el),
            ("cocuk",     rnd_walkin_cocuk),
            ("ortopedi2", rnd_walkin_ortopedi2),
            ("tumor",     rnd_walkin_tumor),
            ("omurga",    rnd_walkin_omurga),
            ("spor",      rnd_walkin_spor),
        ]
        for dept, rng in WALKIN_RNG_LIST:
            env.process(walkin_generator(env, dept, rng, kaynaklar, results))

        env.process(kuyruk_monitor(env, kaynaklar, results))
        env.run(until=SON_KABUL)
        while Sayac.hasta_sayisi > 0 and env.now < MAX_BITIS:
            env.run(until=min(env.now + 60, MAX_BITIS))

        results["gun_sonu_hasta"] = Sayac.hasta_sayisi
        results["toplam_hasta"]   = Hasta._id
        results["aktif_sure"]     = env.now - WARMUP_SURE - (OGLE_BITIS - OGLE_BASLANGIC)

        # ── Hasta sayısı korunumu kontrolü ──────────────────────
        tamamlanan = len(results["sistem_sureleri"])
        kalan      = results["gun_sonu_hasta"]
        toplam     = results["toplam_hasta"]
        if (tamamlanan + kalan) != toplam:
            print(f"  [UYARI] Korunum hatası! "
                  f"Tamamlanan({tamamlanan}) + Kalan({kalan}) ≠ Toplam({toplam})")
        ort = np.mean(results["sistem_sureleri"]) if results["sistem_sureleri"] else 0
        ort_listesi.append(ort)
        res_listesi.append(results)

    # ── Parametreleri geri yükle ─────────────────────────────────
    XRAY_PROB    = orijinal_xray_prob
    WALKIN_ORANI = orijinal_walkin_orani
    XRAY_ODA_PROB = orijinal_xray_oda

    return ort_listesi, res_listesi

def ci95(degerler):
    arr = np.array(degerler) / 60
    n   = len(arr)
    mu  = np.mean(arr)
    s   = np.std(arr, ddof=1)
    t_k = t_dist.ppf(0.975, df=n - 1)
    hata = t_k * s / np.sqrt(n)
    return mu, mu - hata, mu + hata


# ═══════════════════════════════════════════════════════════════════
# DESIGN OF IMPROVEMENT — SENARYO ANALİZİ
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("DESIGN OF IMPROVEMENT — SENARYO ANALİZİ (Low/Med/High)")
print("10 Kombinasyon × 30 Replikasyon")
print("=" * 65)

N_REP_SEN = 30

# ── Faktör seviyeleri ─────────────────────────────────────────────
WI_LOW  = 0.15
WI_MED  = 0.20
WI_HIGH = 0.25

SV_LOW  = 0.9
SV_MED  = 1.0
SV_HIGH = 1.1

XR_LOW  = -0.05
XR_MED  =  0.0
XR_HIGH = +0.05

def xray_prob_delta(delta):
    return {
        k: float(np.clip(v + delta, 0.01, 0.99))
        for k, v in XRAY_PROB.items()
    }

def servis_params_carpan(carpan):
    return {
        "ILK_MUAYENE": (DIST_ILK_MUAYENE["params"][0] * carpan,
                        DIST_ILK_MUAYENE["params"][1]),
        "KONTROL":     (DIST_KONTROL["params"][0] * carpan,
                        DIST_KONTROL["params"][1]),
        "XRAY_CEKIM":  {k: (v["params"][0] * carpan, v["params"][1])
                        for k, v in DIST_XRAY_CEKIM_ODA.items()},
    }

# ── Kombinasyon tablosu ───────────────────────────────────────────
KOMBINASYONLAR = [
    ("BAZ", "Baz (Med-Med-Med)",      WI_MED,  SV_MED,  XR_MED,  "Referans — mevcut sistem"),
    ("D1",  "D1: Low-Med-Med",        WI_LOW,  SV_MED,  XR_MED,  "Walk-in ↓ (%15)"),
    ("D2",  "D2: High-Med-Med",       WI_HIGH, SV_MED,  XR_MED,  "Walk-in ↑ (%25)"),
    ("D3",  "D3: Med-Low-Med",        WI_MED,  SV_LOW,  XR_MED,  "Servis süresi ↓ (×0.9)"),
    ("D4",  "D4: Med-High-Med",       WI_MED,  SV_HIGH, XR_MED,  "Servis süresi ↑ (×1.1)"),
    ("D5",  "D5: Med-Med-Low",        WI_MED,  SV_MED,  XR_LOW,  "X-ray oranı ↓ (−5%)"),
    ("D6",  "D6: Med-Med-High",       WI_MED,  SV_MED,  XR_HIGH, "X-ray oranı ↑ (+5%)"),
    ("D7",  "D7: Low-Low-Low (Best)", WI_LOW,  SV_LOW,  XR_LOW,  "En iyi durum: tüm faktörler Low"),
    ("D8",  "D8: High-High-High (Worst)", WI_HIGH, SV_HIGH, XR_HIGH, "En kötü durum: tüm faktörler High"),
    ("D9",  "D9: Low-Low-High",       WI_LOW,  SV_LOW,  XR_HIGH, "Walk-in↓ + Servis↓ + X-ray↑"),
]

senaryo_ozet = []

for sen_idx, (kod, isim, wi, sv, xr, aciklama) in enumerate(KOMBINASYONLAR):

    if kod == "BAZ":
        ort_l = ortalama_listesi
        res_l = tum_results
    else:
        baz_ilk    = DIST_ILK_MUAYENE["params"]
        baz_kont   = DIST_KONTROL["params"]
        baz_xray_p = {k: v["params"] for k, v in DIST_XRAY_CEKIM_ODA.items()}

        yeni = servis_params_carpan(sv)
        DIST_ILK_MUAYENE["params"] = yeni["ILK_MUAYENE"]
        DIST_KONTROL["params"]     = yeni["KONTROL"]
        for k, p in yeni["XRAY_CEKIM"].items():
            DIST_XRAY_CEKIM_ODA[k]["params"] = p

        ort_l, res_l = tek_senaryo_calistir(
            n_rep                = N_REP_SEN,
            walkin_orani_override= wi,
            xray_prob_override   = xray_prob_delta(xr),
            seed_offset          = sen_idx,
        )

        DIST_ILK_MUAYENE["params"] = baz_ilk
        DIST_KONTROL["params"]     = baz_kont
        for k, p in baz_xray_p.items():
            DIST_XRAY_CEKIM_ODA[k]["params"] = p

    mu_dk, ci_alt, ci_ust = ci95(ort_l)
    ortalama_bekleme = np.mean([s/60 for r in res_l for s in r["bekleme_sureleri"]])
    xray_dol = [
        np.mean([r[k] / r["aktif_sure"] * 100
                 for r in res_l if r["aktif_sure"] > 0])
        for k in ["xray1_mesgul", "xray2_mesgul", "xray3_mesgul"]
    ]

    senaryo_ozet.append({
        "isim":       isim,
        "kod":        kod,
        "aciklama":   aciklama,
        "wi":         wi,
        "sv":         sv,
        "xr":         xr,
        "mu_dk":      mu_dk,
        "ci_alt":     ci_alt,
        "ci_ust":     ci_ust,
        "bekleme_dk": ortalama_bekleme,
        "xray_dol":   xray_dol,
        "ort_listesi": ort_l,
    })

    print(f"\n  [{kod}] {isim}")
    print(f"       {aciklama}")
    print(f"       Walk-in=%{wi*100:.0f}  Servis=×{sv:.1f}  X-ray Δ={xr:+.2f}")
    print(f"       Ort. sistem süresi  : {mu_dk:.2f} dk")
    print(f"       %95 CI              : [{ci_alt:.2f}, {ci_ust:.2f}] dk")
    print(f"       Ort. bekleme süresi : {ortalama_bekleme:.2f} dk")
    print(f"       X-Ray doluluk       : "
          f"Oda1=%{xray_dol[0]:.1f}  Oda2=%{xray_dol[1]:.1f}  Oda3=%{xray_dol[2]:.1f}")

# ── Karşılaştırma tablosu ─────────────────────────────────────────
print("\n\n  ┌─ DESIGN OF IMPROVEMENT — KARŞILAŞTIRMA TABLOSU ──────────────────────────────────────────────")
print(f"  │  {'Kod':<6} {'İsim':<28} {'WI%':>5} {'Sv×':>5} {'XrΔ':>6}  {'Ort.(dk)':>9}  {'%95 CI':^22}  {'Bkl(dk)':>8}  {'Fark':>7}")
print(f"  │  {'-'*6} {'-'*28} {'-'*5} {'-'*5} {'-'*6}  {'-'*9}  {'-'*22}  {'-'*8}  {'-'*7}")
baz_mu = senaryo_ozet[0]["mu_dk"]
for s in senaryo_ozet:
    fark    = s["mu_dk"] - baz_mu
    ci_str  = f"[{s['ci_alt']:.2f}, {s['ci_ust']:.2f}]"
    fark_str= f"{fark:+.2f}" if s["kod"] != "BAZ" else "  —"
    print(f"  │  {s['kod']:<6} {s['isim']:<28} {s['wi']*100:>5.0f} {s['sv']:>5.1f} {s['xr']:>+6.2f}  "
          f"{s['mu_dk']:>9.2f}  {ci_str:^22}  {s['bekleme_dk']:>8.2f}  {fark_str:>7}")
print(f"  └───────────────────────────────────────────────────────────────────────────────────────────────")

# ═══════════════════════════════════════════════════════════════════
# DESIGN OF IMPROVEMENT GRAFİKLERİ
# ═══════════════════════════════════════════════════════════════════
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.05)

renkler_s = [
    "#2E86AB",
    "#5DADE2",
    "#C73E1D",
    "#2A9D8F",
    "#E76F51",
    "#57CC99",
    "#F18F01",
    "#264653",
    "#8B0000",
    "#A23B72",
]

# ── Grafik 1: Bar chart — Ortalama sistem süresi + %95 CI ─────────
fig, ax = plt.subplots(figsize=(15, 6))
kodlar   = [s["kod"]   for s in senaryo_ozet]
ort_dk   = [s["mu_dk"] for s in senaryo_ozet]
ci_hata  = [s["mu_dk"] - s["ci_alt"] for s in senaryo_ozet]
bars = ax.bar(kodlar, ort_dk,
              yerr=ci_hata, capsize=5,
              color=renkler_s[:len(kodlar)], edgecolor="white", alpha=0.88,
              error_kw={"ecolor": "gray", "lw": 1.5})
ax.axhline(MU_REFERANS_SN/60, color="red", ls="--", lw=1.5,
           label=f"Referans μ = {MU_REFERANS_SN/60:.1f} dk")
ax.axhline(baz_mu, color="navy", ls=":", lw=1.2, alpha=0.7,
           label=f"Baz simülasyon = {baz_mu:.1f} dk")
for bar, val in zip(bars, ort_dk):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(ci_hata)*0.12,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(kodlar)))
ax.set_xticklabels(
    [f"{s['kod']}\n(WI={int(s['wi']*100)}% S={s['sv']:.1f} X={s['xr']:+.2f})"
     for s in senaryo_ozet],
    fontsize=8
)
ax.set_ylabel("Ortalama Sistem Süresi (dk)")
ax.set_title("Design of Improvement — Ortalama Sistem Süresi (%95 CI)\n"
             "Faktörler: A=Walk-in Oranı | B=Servis Süresi | C=X-Ray Oranı")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("grafik_doi_sistem.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Grafik 2: Boxplot — replikasyon dağılımı ─────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
bp_data = [[v/60 for v in s["ort_listesi"]] for s in senaryo_ozet]
bp = ax.boxplot(bp_data, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
for patch, renk in zip(bp["boxes"], renkler_s[:len(senaryo_ozet)]):
    patch.set_facecolor(renk); patch.set_alpha(0.7)
ax.set_xticklabels(kodlar)
ax.axhline(MU_REFERANS_SN/60, color="red", ls="--", lw=1.5,
           label=f"Referans μ = {MU_REFERANS_SN/60:.1f} dk")
ax.axhline(baz_mu, color="navy", ls=":", lw=1.2, alpha=0.7,
           label=f"Baz = {baz_mu:.1f} dk")
ax.set_ylabel("Ortalama Sistem Süresi (dk)")
ax.set_title("Design of Improvement — Replikasyon Dağılımı (Boxplot)")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("grafik_doi_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Grafik 3: Faktör Etkisi — Ana Etkiler (Main Effects) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Design of Improvement — Ana Faktör Etkileri (Baz = Med-Med-Med)",
             fontsize=12, fontweight="bold")

wi_levels  = [WI_LOW, WI_MED, WI_HIGH]
wi_mu = []
for wi_val in wi_levels:
    for s in senaryo_ozet:
        if abs(s["wi"] - wi_val) < 1e-9 and abs(s["sv"] - SV_MED) < 1e-9 and abs(s["xr"] - XR_MED) < 1e-9:
            wi_mu.append(s["mu_dk"]); break
wi_labels = [f"Low\n(%{int(WI_LOW*100)})", f"Med\n(%{int(WI_MED*100)})", f"High\n(%{int(WI_HIGH*100)})"]
axes[0].plot(wi_labels, wi_mu, "o-", color="#2E86AB", lw=2.5, ms=10)
for x_i, (lbl, val) in enumerate(zip(wi_labels, wi_mu)):
    axes[0].annotate(f"{val:.1f}", (x_i, val), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
axes[0].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[0].set_title("Faktör A: Walk-in Oranı")
axes[0].set_ylabel("Ort. Sistem Süresi (dk)")
axes[0].legend(fontsize=8)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

sv_levels = [SV_LOW, SV_MED, SV_HIGH]
sv_mu = []
for sv_val in sv_levels:
    for s in senaryo_ozet:
        if abs(s["wi"] - WI_MED) < 1e-9 and abs(s["sv"] - sv_val) < 1e-9 and abs(s["xr"] - XR_MED) < 1e-9:
            sv_mu.append(s["mu_dk"]); break
sv_labels = [f"Low\n(×{SV_LOW})", f"Med\n(×{SV_MED})", f"High\n(×{SV_HIGH})"]
axes[1].plot(sv_labels, sv_mu, "o-", color="#C73E1D", lw=2.5, ms=10)
for x_i, (lbl, val) in enumerate(zip(sv_labels, sv_mu)):
    axes[1].annotate(f"{val:.1f}", (x_i, val), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
axes[1].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[1].set_title("Faktör B: Servis Süresi Çarpanı")
axes[1].set_ylabel("Ort. Sistem Süresi (dk)")
axes[1].legend(fontsize=8)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

xr_levels = [XR_LOW, XR_MED, XR_HIGH]
xr_mu = []
for xr_val in xr_levels:
    for s in senaryo_ozet:
        if abs(s["wi"] - WI_MED) < 1e-9 and abs(s["sv"] - SV_MED) < 1e-9 and abs(s["xr"] - xr_val) < 1e-9:
            xr_mu.append(s["mu_dk"]); break
xr_labels = [f"Low\n({XR_LOW:+.2f})", f"Med\n({XR_MED:+.2f})", f"High\n({XR_HIGH:+.2f})"]
axes[2].plot(xr_labels, xr_mu, "o-", color="#2A9D8F", lw=2.5, ms=10)
for x_i, (lbl, val) in enumerate(zip(xr_labels, xr_mu)):
    axes[2].annotate(f"{val:.1f}", (x_i, val), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
axes[2].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[2].set_title("Faktör C: X-Ray Gönderme Oranı Δ")
axes[2].set_ylabel("Ort. Sistem Süresi (dk)")
axes[2].legend(fontsize=8)
axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("grafik_doi_ana_etkiler.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Grafik 4: Heatmap — Sistem süresi matrisi (A vs B, C=Med) ────
import pandas as pd

wi_vals = [WI_LOW, WI_MED, WI_HIGH]
sv_vals = [SV_LOW, SV_MED, SV_HIGH]
wi_lbls = ["Low\n(15%)", "Med\n(20%)", "High\n(25%)"]
sv_lbls = ["Low\n(×0.9)", "Med\n(×1.0)", "High\n(×1.1)"]

mat = np.full((3, 3), np.nan)
for s in senaryo_ozet:
    if abs(s["xr"] - XR_MED) < 1e-9:
        wi_i = [abs(s["wi"] - w) < 1e-9 for w in wi_vals].index(True) if True in [abs(s["wi"] - w) < 1e-9 for w in wi_vals] else None
        sv_i = [abs(s["sv"] - v) < 1e-9 for v in sv_vals].index(True) if True in [abs(s["sv"] - v) < 1e-9 for v in sv_vals] else None
        if wi_i is not None and sv_i is not None:
            mat[sv_i, wi_i] = s["mu_dk"]

df_mat = pd.DataFrame(mat, index=sv_lbls, columns=wi_lbls)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
               vmin=np.nanmin(mat)*0.97, vmax=np.nanmax(mat)*1.03)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Ort. Sistem Süresi (dk)", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(wi_lbls)
ax.set_yticks(range(3)); ax.set_yticklabels(sv_lbls)
ax.set_xlabel("Faktör A: Walk-in Oranı")
ax.set_ylabel("Faktör B: Servis Süresi")
ax.set_title("Heatmap — Sistem Süresi (dk)\nFaktör C (X-Ray Oranı) = Med (Δ=0)")
for i in range(3):
    for j in range(3):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i,j]:.1f}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if mat[i,j] > (np.nanmax(mat)+np.nanmin(mat))/2 else "black")
plt.tight_layout()
plt.savefig("grafik_doi_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Grafik 5: Best / Worst / Baz karşılaştırması (Boxplot odaklı) ─
fig, ax = plt.subplots(figsize=(10, 5))
secilen = [s for s in senaryo_ozet if s["kod"] in ("BAZ", "D7", "D8", "D9")]
bp_data2 = [[v/60 for v in s["ort_listesi"]] for s in secilen]
renkler2 = {"BAZ": "#2E86AB", "D7": "#264653", "D8": "#8B0000", "D9": "#A23B72"}
bp2 = ax.boxplot(bp_data2, patch_artist=True, notch=False,
                 medianprops=dict(color="white", lw=2.5),
                 whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
for patch, s in zip(bp2["boxes"], secilen):
    patch.set_facecolor(renkler2[s["kod"]]); patch.set_alpha(0.85)
ax.set_xticklabels(
    [f"{s['kod']}: {s['aciklama']}" for s in secilen],
    fontsize=9
)
ax.axhline(MU_REFERANS_SN/60, color="red", ls="--", lw=1.5,
           label=f"Referans μ = {MU_REFERANS_SN/60:.1f} dk")
for i, s in enumerate(secilen):
    ax.text(i+1, s["mu_dk"]/60 + 0.3, f"{s['mu_dk']:.1f}dk",
            ha="center", fontsize=8.5, color="black")
ax.set_ylabel("Ortalama Sistem Süresi (dk)")
ax.set_title("Best / Worst / Baz Karşılaştırması")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("grafik_doi_bestworst.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Faktör etkisi özet tablosu ────────────────────────────────────
print("\n  ┌─ FAKTÖR ETKİSİ ÖZETİ (Low → High değişimi, diğerleri Med) ────────────────")
print(f"  │  {'Faktör':<35} {'Low (dk)':>9}  {'Med (dk)':>9}  {'High (dk)':>9}  {'Etki (dk)':>10}")
print(f"  │  {'-'*35} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")
for label, vals in [
    ("A — Walk-in Oranı", wi_mu),
    ("B — Servis Süresi Çarpanı", sv_mu),
    ("C — X-Ray Gönderme Oranı Δ", xr_mu),
]:
    if len(vals) == 3:
        etki = vals[2] - vals[0]
        print(f"  │  {label:<35} {vals[0]:>9.2f}  {vals[1]:>9.2f}  {vals[2]:>9.2f}  {etki:>+10.2f}")
print(f"  └──────────────────────────────────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("DUYARLILIK ANALİZİ — OAT — 3 PARAMETRE")
print("=" * 65)

N_REP_DA = 30

# ── P1: Walk-in oranı ────────────────────────────────────────────
print("\n  [P1] Walk-in Oranı")
p1_oranlar  = [0.15, 0.20, 0.25, 0.30]
p1_sonuclar = []

for ci_i, oran in enumerate(p1_oranlar):
    ort_l, _ = tek_senaryo_calistir(
        n_rep=N_REP_DA,
        walkin_orani_override=oran,
        seed_offset=0,
    )
    mu_dk, ci_alt, ci_ust = ci95(ort_l)
    p1_sonuclar.append((oran * 100, mu_dk, ci_alt, ci_ust))
    etiket = " ← BAZ" if abs(oran - WALKIN_ORANI) < 0.001 else ""
    print(f"    Walk-in %{oran*100:.0f}{etiket}: {mu_dk:.2f} dk  CI:[{ci_alt:.2f},{ci_ust:.2f}]")

# ── P2: Tüm servis süreleri ±%10 (scale çarpanı) ─────────────────
print("\n  [P2] Tüm Servis Süreleri (scale çarpanı ±%10)")
p2_carpanlar = [0.8, 0.9, 1.0, 1.1, 1.2]
p2_sonuclar  = []

baz_params = {
    "ILK_MUAYENE":    list(DIST_ILK_MUAYENE["params"]),
    "KONTROL":        list(DIST_KONTROL["params"]),
    "XRAY_CEKIM_ODA": {k: list(v["params"]) for k, v in DIST_XRAY_CEKIM_ODA.items()},
}

for ci_i, carpan in enumerate(p2_carpanlar):
    DIST_ILK_MUAYENE["params"] = (baz_params["ILK_MUAYENE"][0] * carpan, baz_params["ILK_MUAYENE"][1])
    DIST_KONTROL["params"]     = (baz_params["KONTROL"][0]     * carpan, baz_params["KONTROL"][1])
    for oda_no, p in baz_params["XRAY_CEKIM_ODA"].items():
        DIST_XRAY_CEKIM_ODA[oda_no]["params"] = (p[0] * carpan, p[1])

    ort_l, _ = tek_senaryo_calistir(n_rep=N_REP_DA, seed_offset=0)
    mu_dk, ci_alt, ci_ust = ci95(ort_l)
    p2_sonuclar.append((carpan, mu_dk, ci_alt, ci_ust))
    etiket = " ← BAZ" if carpan == 1.0 else ""
    print(f"    Scale ×{carpan:.1f}{etiket}: {mu_dk:.2f} dk  CI:[{ci_alt:.2f},{ci_ust:.2f}]")

DIST_ILK_MUAYENE["params"] = tuple(baz_params["ILK_MUAYENE"])
DIST_KONTROL["params"]     = tuple(baz_params["KONTROL"])
for oda_no, p in baz_params["XRAY_CEKIM_ODA"].items():
    DIST_XRAY_CEKIM_ODA[oda_no]["params"] = tuple(p)

# ── P3: X-ray gönderme oranı ±0.1 ───────────────────────────────
print("\n  [P3] X-Ray Gönderme Oranı (tüm klinikler ±0.1)")
p3_deltalar  = [-0.1, 0.0, 0.1]
p3_sonuclar  = []

baz_xray_prob = XRAY_PROB.copy()

for ci_i, delta in enumerate(p3_deltalar):
    xray_override = {
        k: float(np.clip(v + delta, 0.01, 0.99))
        for k, v in baz_xray_prob.items()
    }
    ort_l, _ = tek_senaryo_calistir(
        n_rep=N_REP_DA,
        xray_prob_override=xray_override,
        seed_offset=0,
    )
    mu_dk, ci_alt, ci_ust = ci95(ort_l)
    isaret = f"{delta:+.1f}" if delta != 0 else " 0.0 ← BAZ"
    p3_sonuclar.append((delta, mu_dk, ci_alt, ci_ust))
    print(f"    X-ray oranı {isaret}: {mu_dk:.2f} dk  CI:[{ci_alt:.2f},{ci_ust:.2f}]")

# ── Duyarlılık Grafikleri ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Duyarlılık Analizi — OAT (One-At-a-Time)", fontsize=13, fontweight="bold")

p1_x  = [x[0] for x in p1_sonuclar]
p1_y  = [x[1] for x in p1_sonuclar]
p1_lo = [x[2] for x in p1_sonuclar]
p1_hi = [x[3] for x in p1_sonuclar]
axes[0].plot(p1_x, p1_y, "o-", color="#2E86AB", lw=2, ms=8)
axes[0].fill_between(p1_x, p1_lo, p1_hi, alpha=0.20, color="#2E86AB")
axes[0].axvline(WALKIN_ORANI*100, color="gray", ls="--", lw=1.2, label="Baz")
axes[0].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[0].set_xlabel("Walk-in Oranı (%)")
axes[0].set_ylabel("Ort. Sistem Süresi (dk)")
axes[0].set_title("P1: Walk-in Oranı")
axes[0].legend(fontsize=8)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

p2_x  = [x[0] for x in p2_sonuclar]
p2_y  = [x[1] for x in p2_sonuclar]
p2_lo = [x[2] for x in p2_sonuclar]
p2_hi = [x[3] for x in p2_sonuclar]
axes[1].plot(p2_x, p2_y, "o-", color="#C73E1D", lw=2, ms=8)
axes[1].fill_between(p2_x, p2_lo, p2_hi, alpha=0.20, color="#C73E1D")
axes[1].axvline(1.0, color="gray", ls="--", lw=1.2, label="Baz (×1.0)")
axes[1].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[1].set_xlabel("Servis Süresi Scale Çarpanı")
axes[1].set_ylabel("Ort. Sistem Süresi (dk)")
axes[1].set_title("P2: Servis Süreleri ±%10")
axes[1].legend(fontsize=8)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

p3_x  = [x[0] for x in p3_sonuclar]
p3_y  = [x[1] for x in p3_sonuclar]
p3_lo = [x[2] for x in p3_sonuclar]
p3_hi = [x[3] for x in p3_sonuclar]
axes[2].plot(p3_x, p3_y, "o-", color="#2A9D8F", lw=2, ms=8)
axes[2].fill_between(p3_x, p3_lo, p3_hi, alpha=0.20, color="#2A9D8F")
axes[2].axvline(0.0, color="gray", ls="--", lw=1.2, label="Baz (Δ=0)")
axes[2].axhline(MU_REFERANS_SN/60, color="red", ls=":", lw=1, label="Referans")
axes[2].set_xlabel("X-Ray Oranı Değişimi (Δ)")
axes[2].set_ylabel("Ort. Sistem Süresi (dk)")
axes[2].set_title("P3: X-Ray Gönderme Oranı ±0.1")
axes[2].legend(fontsize=8)
axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("grafik_duyarlilik.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Duyarlılık özet ───────────────────────────────────────────────
def etki_araligi(sonuclar):
    return max(x[1] for x in sonuclar) - min(x[1] for x in sonuclar)

print("\n  ┌─ DUYARLILIK ÖZET ─────────────────────────────────────")
print(f"  │  {'Parametre':<35} {'Etki Aralığı (dk)'}")
print(f"  │  {'-'*50}")
for isim, sonuc in [
    ("P1 — Walk-in Oranı [0.15‒0.30]",         p1_sonuclar),
    ("P2 — Servis Süreleri [×0.8‒×1.2]",       p2_sonuclar),
    ("P3 — X-Ray Gönderme Oranı [±0.1]",       p3_sonuclar),
]:
    print(f"  │  {isim:<38}: {etki_araligi(sonuc):.2f} dk")
print(f"  └────────────────────────────────────────────────────────")

print("\nTÜM ANALİZLER TAMAMLANDI.")

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

try:
    from output_analiz import cagir_output_analizi
    cagir_output_analizi(
        ortalama_listesi = ortalama_listesi,
        tum_results      = tum_results,
        senaryo_ozet     = senaryo_ozet,
        p1_sonuclar      = p1_sonuclar,
        p2_sonuclar      = p2_sonuclar,
        p3_sonuclar      = p3_sonuclar,
    )
except ImportError:
    print("\n[Uyarı] output_analiz.py bulunamadı. Output analizi atlandı.")
    print("        output_analiz.py'yi aynı klasöre kopyalayın ve tekrar çalıştırın.")